FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir \
    packaging==22.0 \
    PyYAML==6.0.2 \
    easydict==1.13 \
    cython==3.0.12

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgeos-dev \
    libsm6 \
    libxext6 \
    libexiv2-dev \
    libxrender-dev \
    libboost-all-dev \
    wget

RUN pip3 install --no-cache-dir \
    opencv-python \
    pandas==2.1.4 \
    pycocotools==2.0.8 \
    jpeg4py==0.1.4 \
    tikzplotlib==0.10.1 \
    lmdb==1.6.2 \
    scipy==1.15.2 \
    visdom==0.2.4 \
    timm==1.0.14 \
    yacs==0.1.8 \
    scikit-image==0.25.2 \
    thop==0.1.1.post2209072238 \
    tb-nightly==2.20.0a20250219

RUN mkdir -p /weights
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_tiny_1600e_1k.pt -O /weights/fast_itpn_tiny_1600e_1k.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_small_1600e_1k.pt -O /weights/fast_itpn_small_1600e_1k.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_base_clipl_e1600.pt -O /weights/fast_itpn_base_clipl_e1600.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/fast_itpn_large_1600e_1k.pt -O /weights/fast_itpn_large_e1600_1k.pt
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_tiny.pth.tar -O /weights/MCITRACK_tiny.pth.tar
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_small.pth.tar -O /weights/MCITRACK_small.pth.tar
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_base.pth.tar -O /weights/MCITRACK_base.pth.tar
RUN wget https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/MCITRACK_large.pth.tar -O /weights/MCITRACK_large.pth.tar

RUN python3 -m pip install supervisely==6.73.483
LABEL python_sdk_version=6.73.483
