# syntax=docker/dockerfile:1.3
# TODO: Consider swapping base image depending on availability of graphics device.
ARG BASE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu20.04

FROM ${BASE_IMAGE} as base

# NOTE: Avoid questions during build process
ENV DEBIAN_FRONTEND=noninteractive
# NOTE: Avoid images getting bigger than necessary
ENV PIP_NO_CACHE_DIR=1

# NOTE: Re-declare `USE_CUDA` to bring within build scope.
ARG USE_CUDA=on
ARG USERNAME=root
# ARG UID=1000
# ARG GID=1000

# Install Core / networking packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    sudo \
    gnupg2 \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists*

# Install Packages for C++ compilations & Python.
# NOTE: `python3-dev` may be required for pybind deps?
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    g++ \
    gdb \
    python3-dev \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists*

# Install packages for building C++ sub-modules.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libopencv-dev \
    libfmt-dev \
    && rm -rf /var/lib/apt/lists*

# TODO: libnvidia-gl-470
# it's needed for eGL rendering and maybe vulkan?
# but the issue is I don't want to lock down the version number...
# libnvidia-gl-470 \
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    freeglut3-dev  \
    && rm -rf /var/lib/apt/lists*

# Install Isaac Gym dependencies.
# Mostly display drivers et al.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    gcc-8 \
    g++-8 \
    vulkan-utils \
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1 \
    git-lfs

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


# Setup GUI-access enabled user.
# FIXME: Perhaps unnecessary since we need to run with
# --privileged option anyways (for network access).
# RUN echo "useradd -m -s /bin/bash ${USERNAME}"
# RUN [ $(getent group ${GID}) ] || groupadd -g ${GID} ${USERNAME} && \
#     useradd -m -s /bin/bash ${USERNAME} -u ${UID} -g ${GID} && \
#     usermod -aG sudo ${USERNAME} && \
#     echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} && \
#     chmod 440 /etc/sudoers.d/${USERNAME} && \
#     usermod -u ${UID} ${USERNAME} && \
#     usermod -g ${GID} ${USERNAME}

# Add to `video` group (possibly needed to access /dev/dri)
# RUN usermod -aG video ${USERNAME}

# USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Add ~/.local/bin to path.
ENV HOME="/home/root"
ENV PATH="${PATH}:${HOME}/.local/bin"

# Update python deps.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade wheel setuptools Cython

# Install additional python packages for building our package.
# WARN: cmake python package probably conflicts with
# the system-wide cmake version... watch out for unexpected errors.
RUN python3 -m pip install --upgrade \
    cmake \
    cmake-build-extension \
    pybind11 \
    shapely \
    chama \
    pandas \
    scikit-image \
    triangle

# Configure PATH variable by default.
# TODO: Is this the right way to go?
RUN echo 'export PATH="$HOME/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
# export `CUDA_HOME` env variable to .bashrc.
RUN printf 'export CUDA_HOME=/usr/local/cuda\nexport LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"\nexport PATH="${PATH}:${CUDA_HOME}/bin"' >> "${HOME}/.bashrc"

# NOTE: Install torch.
# setuptools version lock is required due to
# comptibility between pytorch and tensorboard.
RUN python3 -m pip install --upgrade \
    setuptools==59.5.0 \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    tensorboard \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install python utilities, RL,
# geometry processing / IO, rendering,
# deep-learning and vision related packages.
RUN python3 -m pip install --upgrade \
    gym \
    tqdm \
    trimesh \
    open3d \
    pybullet_rendering \
    pyrender \
    einops \
    opencv-python


# NOTE: extra packages for 3D geometry processing.
RUN python3 -m pip install --upgrade \
    rtree fvcore iopath
RUN PY_VERSION_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')" && \
    python3 -m pip install --no-index --no-cache --upgrade \ 
    -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3${PY_VERSION_MINOR}_cu113_pyt1110/download.html" \
    pytorch3d

# NOTE: pytorch-points3d
RUN python3 -m pip install --upgrade --no-index \
    -f 'https://data.pyg.org/whl/torch-1.11.0+cu113.html' \
    torch-scatter torch-sparse torch-cluster torch-spline-conv

# pytorch-geometric should be installed separately after build docker
# pytorch-geometric
# RUN python3 -m pip install --upgrade \
#     torch-geometric

# NOTE: below lines require compilation, so
# it does not work without `nvidia-container-runtime`.
# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# Reference:
# https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime
# https://github.com/NVIDIA/nvidia-docker/wiki/Advanced-topics#default-runtime
ENV TORCH_CUDA_ARCH_LIST='Ampere'
# RUN FORCE_CUDA=1 python3 -m pip install \
# #     torch-points-kernels
# RUN python3 -m pip install --upgrade \
#     torch-points3d

# Install `nvdiffrast` for fast mesh rasterization.
RUN python3 -m pip install --upgrade \
    git+https://github.com/NVlabs/nvdiffrast.git

# Install PySDF for signed-distance function generations.
# Install torchmcubes for visualization.
# Install yourdfpy for loading URDF files.
RUN python3 -m pip install --upgrade \
    yourdfpy \
    pysdf 
# Not required for now
# RUN python3 -m pip install --upgrade \
#     git+https://github.com/tatsy/torchmcubes.git

RUN python3 -m pip install --upgrade \
    hydra-core \
    transformers \
    wandb

# Profiling ...
RUN python3 -m pip install --upgrade \
    line_profiler \
    cachetools \
    nvtx

# Dataset ...
RUN python3 -m pip install --upgrade \
    webdataset \
    graspnetAPI

# # Extra package (MVP)
RUN python3 -m pip install --upgrade \
    git+https://github.com/ir413/mvp

# Upgrade Open3D for visualization
RUN python3 -m pip install open3d --upgrade

RUN python3 -m pip install --upgrade 'omegaconf>=2.3.0'
RUN python3 -m pip install --upgrade 'hydra-core>=1.3.2'
RUN python3 -m pip install kornia
RUN python3 -m pip install "flash-attn==1.0.4"

# QoL packages + pyro-ppl for distribution
RUN python3 -m pip install --upgrade \
    autopep8 \
    pyflakes \
    mypy \
    icecream \
    torchviz \
    gdown

# Install moviepy for video logging.
RUN python3 -m pip install --upgrade \
    moviepy

# Extra packages for training scripts
RUN python3 -m pip install --upgrade numpy e3nn jax flax
# For pretraining
RUN python3 -m pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
# For policy training
RUN python3 -m pip install --upgrade opt_einsum coacd pymeshlab

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

CMD ["/bin/bash"]
