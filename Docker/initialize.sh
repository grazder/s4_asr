#!/usr/bin/env bash

## Check variables
if [[ -z ${LOGIN+x} ]] || [[ -z ${LOGIN_UID+x} ]] || [[ -z ${LOGIN_GID+x} ]] || [[ -z ${JUPYTER_PORT+x} ]]; then
    echo "[ error ] env variables LOGIN, LOGIN_UID, LOGIN_GID or JUPYTER_PORT unseted.";
    exit 1;
fi

# Make ALL users sudo
echo "ALL ALL = NOPASSWD: ALL" > /etc/sudoers.d/ALL;

# This need for run command in sudo
sed -i '/secure_path=/d' /etc/sudoers;

# Clone user name, UID and GID from host
groupadd -r -g $LOGIN_GID $LOGIN && \
useradd -m -u $LOGIN_UID -g $LOGIN_GID -G root -s /bin/bash $LOGIN;

[ -d "/home/$LOGIN" ] && chown -R $LOGIN:$LOGIN /home/$LOGIN;

# Clone authorized_keys from host. It is necessary for direct ssh access to container
if [[ -f /mnt/authorized_keys ]]; then
    runuser -l $LOGIN -c "mkdir /home/$LOGIN/.ssh";
    runuser -l $LOGIN -c "cat /mnt/authorized_keys > /home/$LOGIN/.ssh/authorized_keys";
    service ssh start;
fi

# Clone PATH value from root user. PATH value for root is set in dockerfile.
runuser -l $LOGIN -c "echo PATH=$PATH >> /home/$LOGIN/.bashrc";

# Create aliases for Jupyter tools
JUPYTER_ARGS="--port $JUPYTER_PORT --no-browser --ip=\"0.0.0.0\" --allow-root --NotebookApp.iopub_msg_rate_limit=1000000.0 --NotebookApp.iopub_data_rate_limit=100000000.0"
runuser -l $LOGIN -c "echo 'alias run-jupyter=\"jupyter notebook $JUPYTER_ARGS\"' >> /home/$LOGIN/.bashrc"
runuser -l $LOGIN -c "echo 'alias run-jupyter-lab=\"jupyter-lab --NotebookApp.notebook_dir=/ --LabApp.token='' $JUPYTER_ARGS\"' >> /home/$LOGIN/.bashrc"
runuser -l $LOGIN -c ". /home/$LOGIN/.bashrc"

if [ "$MODE" == "bash" ]; then
  su $LOGIN;
else
  runuser -l $LOGIN -c "/opt/conda/bin/jupyter-lab --NotebookApp.notebook_dir=/ --LabApp.token='' $JUPYTER_ARGS"
fi