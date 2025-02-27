from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
from lib.test.utils.hann import hann2d
from lib.models.mcitrack import build_mcitrack
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box


class MCITracker(BaseTracker):
    def __init__(
        self,
        params,
        update_threshold=0.8,
        update_h_t=0.88,
        update_intervals=70,
        memory_bank=500,
        device="cuda",
    ):
        super().__init__(params)
        network = build_mcitrack(params.cfg)
        network.load_state_dict(
            torch.load(self.params.checkpoint, map_location="cpu")["net"], strict=True
        )
        self.cfg = params.cfg
        self.network = network.to(device)
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            self.output_window = hann2d(
                torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True
            ).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.frame_id = 0

        self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS

        self.update_threshold = update_threshold
        self.update_h_t = update_h_t
        self.update_intervals = update_intervals
        self.memory_bank = memory_bank

    def initialize(self, image, info: dict):
        # get the initial templates
        z_patch_arr, resize_factor = sample_target(
            image,
            info["init_bbox"],
            self.params.template_factor,
            output_sz=self.params.template_size,
        )
        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        self.state = info["init_bbox"]
        prev_box_crop = transform_image_to_crop(
            torch.tensor(info["init_bbox"]),
            torch.tensor(info["init_bbox"]),
            resize_factor,
            torch.Tensor([self.params.template_size, self.params.template_size]),
            normalize=True,
        )
        self.template_anno_list = [
            prev_box_crop.to(template.device).unsqueeze(0)
        ] * self.num_template
        self.frame_id = 0
        self.memory_template_list = self.template_list.copy()
        self.memory_template_anno_list = self.template_anno_list.copy()

    def update_settings(self, settings):
        self.update_threshold = settings.get("update_threshold", 0.8)
        self.update_h_t = settings.get("update_h_t", 0.88)
        self.update_intervals = settings.get("update_intervals", 70)
        self.memory_bank = settings.get("memory_bank", 500)

    def track(self, image):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(
            image,
            self.state,
            self.params.search_factor,
            output_sz=self.params.search_size,
        )  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        search_list = [search]

        # run the encoder
        with torch.no_grad():
            enc_opt = self.network.forward_encoder(
                self.template_list, search_list, self.template_anno_list
            )

        # run the time neck
        with torch.no_grad():
            hidden_state = self.h_state.copy()
            encoder_out, out_neck, h = self.network.forward_neck(enc_opt, hidden_state)
        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=out_neck)

        # add hann windows
        pred_score_map = out_dict["score_map"]
        if self.cfg.TEST.WINDOW == True:  # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if "size_map" in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response,
                out_dict["size_map"],
                out_dict["offset_map"],
                return_score=True,
            )
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict["offset_map"], return_score=True
            )
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (
            pred_boxes.mean(dim=0) * self.params.search_size / resize_factor
        ).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(
            self.map_box_back(pred_box, resize_factor), H, W, margin=10
        )
        # update hiden state
        self.h_state = h
        if conf_score.item() < self.update_h_t:
            self.h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS

        # update the template
        if self.num_template > 1:
            if conf_score > self.update_threshold:
                z_patch_arr, resize_factor = sample_target(
                    image,
                    self.state,
                    self.params.template_factor,
                    output_sz=self.params.template_size,
                )
                template = self.preprocessor.process(z_patch_arr)
                self.memory_template_list.append(template)
                prev_box_crop = transform_image_to_crop(
                    torch.tensor(self.state),
                    torch.tensor(self.state),
                    resize_factor,
                    torch.Tensor(
                        [self.params.template_size, self.params.template_size]
                    ),
                    normalize=True,
                )
                self.memory_template_anno_list.append(
                    prev_box_crop.to(template.device).unsqueeze(0)
                )
                if len(self.memory_template_list) > self.memory_bank:
                    self.memory_template_list.pop(0)
                    self.memory_template_anno_list.pop(0)
        if self.frame_id % self.update_intervals == 0:
            assert len(self.memory_template_anno_list) == len(self.memory_template_list)
            len_list = len(self.memory_template_anno_list)
            interval = len_list // self.num_template
            for i in range(1, self.num_template):
                idx = interval * i
                if idx > len_list:
                    idx = len_list
                self.template_list.append(self.memory_template_list[idx])
                self.template_list.pop(1)
                self.template_anno_list.append(self.memory_template_anno_list[idx])
                self.template_anno_list.pop(1)
        assert len(self.template_list) == self.num_template
        return {"target_bbox": self.state, "best_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = (
            self.state[0] + 0.5 * self.state[2],
            self.state[1] + 0.5 * self.state[3],
        )
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
