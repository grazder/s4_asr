#/bin/bash

JUPYTER_PORT=31212
SSH_PORT=31213
TENSORBOARD_PORT=31214

## Check variables
if [ -z "$JUPYTER_PORT" ] || [ -z "$SSH_PORT" ] || [ -z "$TENSORBOARD_PORT" ]; then
    echo "Please set up correct ports for jupyter web interface, ssh and tensorboard in start.sh.";
    exit 1;
fi
 
CONTAINER_NAME="$(whoami)"

if [ "$1" == "--it" ]; then
  MODE="bash"
  RUN_ARGS=(-e SSH_AUTH_SOCK=/ssh-agent  -v $SSH_AUTH_SOCK:/ssh-agent -it --rm)
else
  MODE="lab"
  RUN_ARGS=(-d)
fi


docker run --runtime=nvidia \
-e NVIDIA_VISIBLE_DEVICES=all \
-e LOGIN=$(whoami) \
-e LOGIN_UID=$(id -u) \
-e LOGIN_GID=$(id -g) \
-e JUPYTER_PORT=$JUPYTER_PORT \
-e MODE=$MODE \
--privileged \
-v $HOME/.ssh/authorized_keys:/mnt/authorized_keys:ro \
-v $HOME/work:/work \
-v /home/data/:/data \
-p $JUPYTER_PORT:$JUPYTER_PORT \
-p $SSH_PORT:22 \
-p $TENSORBOARD_PORT:6006 \
--shm-size 64G \
"${RUN_ARGS[@]}" \
--name $CONTAINER_NAME pytorch22.05:latest

if [ "$MODE" == "lab" ]; then
  echo -e "Container started\n" && \
  echo -e "Web interface: http://127.0.0.1:$JUPYTER_PORT/\n" && \
  echo -e "SSH interface: ssh $(whoami)@$(cat /proc/sys/kernel/hostname).sys-msk.kontur-extern.ru -p $SSH_PORT"
fi
