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

# Install tmuxp
RUN sudo apt install tmux && pip install --upgrade tmuxp && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && \
    echo 'setw -g mouse on' > ~/.tmux.conf 

# Install additional libraries
RUN sudo apt install -y wmctrl wget lsb-release python3-opencv
RUN pip install numpy==1.23 scipy scikit-image matplotlib wmctrl

# Install ros2 Humble
RUN sudo apt update && sudo apt install locales -y && \
    sudo locale-gen en_US en_US.UTF-8 && \
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    sudo apt install software-properties-common -y && \
    sudo add-apt-repository universe && \
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    sudo apt update && \
    sudo apt upgrade && \
    sudo apt install -y ros-humble-desktop ros-dev-tools python3-vcstool && \
    sudo apt install -y ros-humble-tf-transformations ros-humble-diagnostic-common-diagnostics ros-humble-rqt-runtime-monitor && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install g2o for access to g2o_viewer
RUN sudo apt install cmake libeigen3-dev libspdlog-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5 -y && \
    git clone https://github.com/RainerKuemmerle/g2o.git && \
    cd g2o && mkdir build && cd build && cmake ../ && make && export PATH="$HOME/g2o/build/bin:$PATH"

# Default shell
SHELL ["/bin/bash", "-c"]

# Set the User environment variables
RUN echo "export USER=${USERNAME}" >> ~/.bashrc

# Install graphviz
RUN sudo apt-get install graphviz graphviz-dev
