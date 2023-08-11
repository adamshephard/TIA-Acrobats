FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-de -y \
    && apt-get install python3.8-dev -y \
    && apt-get install build-essential -y \
    && apt-get clean \
    && :

# Get other packages
RUN apt-get install -y gcc
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get update && apt-get install -y libopenjp2-7-dev libopenjp2-tools openslide-tools
RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get update && apt-get install -y libpixman-1-dev

# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Libraries
RUN python -m pip install -U pip
RUN pip install tiatoolbox
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install albumentations 
RUN pip install pycm 
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install segmentation-models-pytorch
RUN pip install click

# RUN apt install nvidia-modprobe -y

# FOLDERS and PERMISSIONS
USER root
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /input/images /input/annos \
    /configs \
    && chown -R algorithm:algorithm /opt/algorithm /input /output \
    /input/images /input/annos

WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# add scripts and models 
COPY --chown=algorithm:algorithm main.py /opt/algorithm/
COPY --chown=algorithm:algorithm tissue_segmentation.py /opt/algorithm/
COPY --chown=algorithm:algorithm registration.py /opt/algorithm/
COPY --chown=algorithm:algorithm landmark_registration.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm wsi_registration_local.py /opt/algorithm/
COPY --chown=algorithm:algorithm models/unet-acrobat.pth /opt/algorithm/models/

# ENTRYPOINT python -u -m main $0 $@s
ENTRYPOINT ["python", "./main.py"]
# ENTRYPOINT ["/bin/bash", "-c"]