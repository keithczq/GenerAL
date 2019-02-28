#!/usr/bin/env bash
source /opt/ros/$ROS_DISTRO/setup.bash
export PYTHONPATH=$PYTHONPATH:/usr/lib
export PATH=$PATH:$HOME/.local/bin
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion


export ROS_HOSTNAME=$(hostname -f).local
export CUDA_VISIBLE_DEVICES=1
cd ~/Desktop/ws && touch ~/.graspit/images/tmp.jpg && rm -rf ~/.graspit/images/* && python run.py