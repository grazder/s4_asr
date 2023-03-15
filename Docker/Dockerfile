FROM nvcr.io/nvidia/pytorch:22.05-py3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN sed -i "s|archive.ubuntu.com|ru.archive.ubuntu.com|g" /etc/apt/sources.list
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONUNBUFFERED 1

ONBUILD COPY dotfiles ./dotfiles
ONBUILD RUN cd ~/dotfiles && stow bash zsh tmux && sudo chsh -s /usr/bin/zsh "$(whoami)"
# nvcr pytorch image sets SHELL=/bin/bash
ONBUILD ENV SHELL=/bin/zsh

ENV PIP_NO_CACHE_DIR=1

RUN apt update --fix-missing -y && apt upgrade -y
RUN apt-get install -y \
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
    ffmpeg \
    openssh-client \
    sshfs && \
    apt -yqq install ssh

RUN pip install --upgrade pip

RUN pip install Cython
RUN pip install nemo_toolkit[asr]==1.15.0

RUN pip install pytest matplotlib jupyter ipython ipdb gpustat scikit-learn spacy munch einops opt_einsum fvcore gsutil cmake pykeops zstandard psutil h5py twine gdown eeghdf \
    && python -m spacy download en_core_web_sm

RUN pip install jupyterlab ipywidgets jupyterlab-lsp python-lsp-server jupyterlab_execute_time jupyterlab_code_formatter

RUN pip install autopep8 \
                aquirdturtle_collapsible_headings \
                lckr-jupyterlab-variableinspector

RUN git clone https://github.com/HazyResearch/safari.git && \
    cd safari/csrc/fftconv && pip install . && cd ../../../ && rm -rf safari

COPY initialize.sh .
RUN chmod +x initialize.sh

ENTRYPOINT ./initialize.sh
