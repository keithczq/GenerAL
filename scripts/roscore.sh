#!/bin/bash
source /opt/ros/indigo/setup.bash
export PYTHONPATH=$PYTHONPATH:/usr/lib
export PATH=$PATH:$HOME/.local/bin

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion


export ROS_HOSTNAME=$(hostname -f).local
cd ~/Desktop/ws/staubli_barrett_ws
source devel/setup.bash
cd ~/Desktop/ws
clear
screen -ls
export ROS_MASTER_URI=http://localhost:$1
echo $ROS_MASTER_URI
roscore -p $1