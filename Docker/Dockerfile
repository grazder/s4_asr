FROM nvcr.io/nvidia/pytorch:22.11-py3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN sed -i "s|archive.ubuntu.com|ru.archive.ubuntu.com|g" /etc/apt/sources.list
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV HOME="/home/$(whoami)"
RUN mkdir -p "/home/$(whoami)" && chmod 777 "/home/$(whoami)"
WORKDIR "/home/$(whoami)"

ONBUILD COPY dotfiles ./dotfiles
ONBUILD RUN cd ~/dotfiles && stow bash zsh tmux && sudo chsh -s /usr/bin/zsh $(whoami)
# nvcr pytorch image sets SHELL=/bin/bash
ONBUILD ENV SHELL=/bin/zsh

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    libsndfile1 \
    ffmpeg

RUN pip install -U --upgrade pip

RUN pip install -U Cython
RUN pip install -U nemo_toolkit['all']==1.15.0

RUN pip install -U pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops opt_einsum fvcore gsutil cmake pykeops zstandard psutil h5py twine gdown eeghdf \
    && python -m spacy download en_core_web_sm

RUN pip install -U hydra-core==1.3.1 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 pyrootutils rich
RUN pip install -U transformers==4.26.0 datasets==2.9.0 pytorch-lightning==1.9.0 triton==2.0.0.dev20221202 wandb==0.13.9 timm==0.6.12 torchmetrics==0.11.1

RUN pip install -U jupyterlab ipywidgets jupyterlab-lsp python-lsp-server jupyterlab_execute_time jupyterlab_code_formatter

RUN pip install autopep8 \
                aquirdturtle_collapsible_headings \
                lckr-jupyterlab-variableinspector

COPY initialize.sh .
RUN chmod +x initialize.sh

ENTRYPOINT ./initialize.sh
