FROM ubuntu:22.04

# Set user arguments (being username, echo $UID, and id -g)
ARG USERNAME=dbutterfield3
ARG USER_UID=3332128
ARG USER_GID=2626

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y sudo build-essential libboost-all-dev git-all cmake libblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with matching UID/GID
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add a custom .bashrc with warm colors and fixed hostname
RUN echo 'export PS1="\\[\\e[1;31m\\]\\u@roman_container\\[\\e[0m\\]:\\[\\e[1;33m\\]\\w\\[\\e[0m\\]\\$ "' > /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

# Set default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Install GTSAM
RUN git clone https://github.com/borglab/gtsam.git && \
    cd gtsam && \
    git checkout 686e16aaae26c9a4f23d4af7f2d4a504125ec9c3 && \
    mkdir build && \
    cd build && \
    cmake .. -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON && \
    sudo make install

# Install additional requirements
RUN sudo apt update && sudo apt install -y libeigen3-dev python3-pip curl

# Install additional libraries
RUN sudo apt install -y wmctrl wget lsb-release python3-opencv
RUN pip install numpy==1.23 scipy scikit-image matplotlib wmctrl

# Install g2o for access to g2o_viewer
RUN sudo apt install cmake libeigen3-dev libspdlog-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5 -y && \
    git clone https://github.com/RainerKuemmerle/g2o.git && \
    cd g2o && mkdir build && cd build && cmake ../ && make && export PATH="$HOME/g2o/build/bin:$PATH"

# Default shell
SHELL ["/bin/bash", "-c"]

# Set the User environment variables
RUN echo "export USER=${USERNAME}" >> ~/.bashrc

# Install graphviz
RUN sudo apt-get install graphviz graphviz-dev -y

# Install Rerun and dependencies
RUN pip install rerun-sdk && \
    sudo apt install vulkan-tools && \
    sudo apt update && \
    sudo apt install mesa-vulkan-drivers

# Add user to proper groups so Rerun has permission to access GPU
RUN sudo usermod -aG input ${USERNAME} && \
    sudo usermod -aG video ${USERNAME}

# Install Nvidia Toolkit so we can run CuPy
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
    sudo apt-get update && \
    sudo apt-get -y install cuda-toolkit-12-5

# Uninstall matplotlib installed by OS package manager as it conflicts with pip installed version
RUN sudo apt remove python3-matplotlib -y

# Update certain libraries to avoid dependency mismatches
RUN pip install --upgrade pip && \
    pip install --upgrade requests urllib3 charset-normalizer

# Install gdown for use with test cases
RUN pip install gdown

# Setup a symbolic link from python to python3 (for use with WandB Sweeps)
RUN sudo ln -s /usr/bin/python3 /usr/bin/python