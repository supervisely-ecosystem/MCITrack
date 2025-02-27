import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lib.test.parameter.mcitrack import parameters
from lib.train.admin import create_default_local_file_ITP_train
from lib.test.evaluation import create_default_local_file_ITP_test
from supervisely_integration.src.tracker import MCITracker
import supervisely as sly
from dotenv import load_dotenv
import os
from supervisely.nn.inference import BBoxTracking
from supervisely.nn.prediction_dto import PredictionBBox
from typing import Any, Dict, Literal, Optional, Union
from pathlib import Path
import numpy as np
import torch
from supervisely.sly_logger import logger
from supervisely.io import env
from supervisely.nn.inference.inference import Inference


# load credentials
load_dotenv("supervisely.env")
api = sly.Api()

# set smart cache parameters
os.environ["SMART_CACHE_TTL"] = str(5 * 60)
os.environ["SMART_CACHE_SIZE"] = str(512)

# get models data path
root_source_path = str(Path(__file__).parents[2])
models_data_path = os.path.join(
    root_source_path, "supervisely_integration", "models.json"
)


class SlyMCITracker(BBoxTracking):

    def __init__(
        self,
        model_dir: Optional[str] = None,
        custom_inference_settings: Optional[Union[Dict[str, Any], str]] = None,
    ):
        Inference.__init__(
            self,
            model_dir,
            custom_inference_settings,
            sliding_window_mode=None,
            use_gui=True,
        )

        # try:
        #     self.load_on_device(model_dir, "cuda")
        # except RuntimeError:
        #     self.load_on_device(model_dir, "cpu")
        #     logger.warning("Failed to load model on CUDA device.")

        logger.debug(
            "Smart cache params",
            extra={"ttl": env.smart_cache_ttl(), "maxsize": env.smart_cache_size()},
        )

    def get_models(self, mode="table"):
        model_data = sly.json.load_json_file(models_data_path)
        if mode == "table":
            for element in model_data:
                del element["weights"]
            return model_data
        elif mode == "info":
            models_data_processed = {}
            for element in model_data:
                models_data_processed[element["Model"]] = {
                    "backbone": element["weights"]["backbone"],
                    "checkpoint": element["weights"]["checkpoint"],
                }
            return models_data_processed

    def support_custom_models(self):
        return False

    def get_weights_path(self):
        models_data = self.get_models(mode="info")
        selected_model = self.gui.get_checkpoint_info()["Model"]
        backbone_path = models_data[selected_model]["backbone"]
        checkpoint_path = models_data[selected_model]["checkpoint"]
        if sly.is_development():
            backbone_path = root_source_path + backbone_path
            checkpoint_path = root_source_path + checkpoint_path
        return backbone_path, checkpoint_path

    def get_parameters(self):
        selected_model = self.gui.get_checkpoint_info()["Model"]
        if selected_model == "MCITrack-T224":
            params = parameters("mcitrack_t224")
        elif selected_model == "MCITrack-S224":
            params = parameters("mcitrack_s224")
        elif selected_model == "MCITrack-B224":
            params = parameters("mcitrack_b224")
        elif selected_model == "MCITrack-L224":
            params = parameters("mcitrack_l224")
        return params

    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cuda",
    ):
        backbone_path, checkpoint_path = self.get_weights_path()
        model_params = self.get_parameters()
        model_params.checkpoint = checkpoint_path
        model_params.cfg["MODEL"]["ENCODER"]["PRETRAIN_TYPE"] = backbone_path
        self.tracker = MCITracker(params=model_params, device=device)

    def initialize(self, init_rgb_image: np.ndarray, target_bbox: PredictionBBox):
        top, left, bottom, right = target_bbox.bbox_tlbr
        width = right - left + 1
        height = bottom - top + 1
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        init_bbox = [center_x, center_y, width, height]
        init_info = {"init_bbox": init_bbox}
        self.tracker.initialize(init_rgb_image, init_info)

    def predict(
        self,
        rgb_image: np.ndarray,
        settings: Dict[str, Any],
        prev_rgb_image: np.ndarray,
        target_bbox: PredictionBBox,
    ) -> PredictionBBox:
        torch.set_grad_enabled(False)
        class_name = target_bbox.class_name
        self.tracker.update_settings(settings)
        output = self.tracker.track(rgb_image)
        center_x, center_y, width, height = [int(s) for s in output["target_bbox"]]
        top = center_y - (height / 2)
        left = center_x - (width / 2)
        bottom = center_y + (height / 2)
        right = center_x + (width / 2)
        tlbr = [int(top), int(left), int(bottom), int(right)]
        return PredictionBBox(class_name, tlbr, None)


create_default_local_file_ITP_train(workspace_dir="./", data_dir="./")
create_default_local_file_ITP_test(workspace_dir="./", data_dir="./", save_dir="./")

model = SlyMCITracker(
    custom_inference_settings="supervisely_integration/inference_settings.yaml"
)
model.gui._models_table.select_row(2)
model.serve()
