FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0+PTX"
ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Set the working directory
WORKDIR /workspace/spatialgen

# inatall library dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 python3 python3-pip -y

# Install some basic utilities.
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt-get install -y \
  curl \
  ca-certificates \
  sudo \
  git \
  bzip2 \
  libx11-6 \
  tmux \
  wget \
  build-essential \
  zsh \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/spatialgen/

# Upgrade pip
RUN python3 -m pip install --upgrade pip
# install pytorch and torchvision
RUN pip3 install -i https://download.pytorch.org/whl/cu121 -U torch==2.3.1 torchvision==0.18.1
# install xformers
RUN pip3 install -i https://download.pytorch.org/whl/cu121 -U xformers==0.0.27
# install other dependencies
RUN pip3 install -U -r /workspace/spatialgen/requirements.txt
RUN pip3 install iopath
RUN pip3 install --no-index --no-cache-dir pytorch3d -f  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt231/download.html
RUN pip install deepspeed==0.15.4 recordclass mapbox-earcut diptest==0.6.1

ENV HOME=/root
RUN mkdir $HOME/.ssh && chmod -R 0755 $HOME

# set zsh
RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
  && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# setup ssh
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]