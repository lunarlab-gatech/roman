DATA_DIR='/media/dbutterfield3/T73'
REPO_DIR='/home/dbutterfield3/Research/ROMAN'

docker run -it \
    --name="roman_baseline" \
    --net="host" \
    --privileged \
    --gpus="all" \
    --device /dev/dri \
    --workdir="/home/$USER/roman" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=/tmp/.Xauthority" \
    --env="XDG_RUNTIME_DIR=/tmp/runtime-$USER" \
    --env="USER_ID=$(id -u)" \
    --env="GROUP_ID=$(id -g)" \
    --volume="$REPO_DIR:/home/$USER/roman" \
    --volume="$DATA_DIR:/home/$USER/data" \
    --volume="/home/$USER/.bash_aliases:/home/$USER/.bash_aliases" \
    --volume="/home/$USER/.ssh:/home/$USER/.ssh:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume /tmp/runtime-$USER:/tmp/runtime-$USER \
    --volume="$XAUTHORITY:/tmp/.Xauthority:ro" \
    roman_baseline \
    /bin/bash
