#!/bin/bash
if [ $1 = "11311" ]; then
    source /opt/ros/indigo/setup.bash
else
    source /opt/ros/kinetic/setup.bash
fi

export PYTHONPATH=$PYTHONPATH:/usr/lib
export PATH=$PATH:$HOME/.local/bin

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion


export ROS_HOSTNAME=$(hostname -f).local
cd ~/Desktop/ws/staubli_barrett_ws
source devel/setup.bash
clear
export ROS_MASTER_URI=http://localhost:$1
if [ $1 = "11311" ]; then
    echo "this is master"
    roslaunch staubli_barretthand_moveit_config move_group.launch id:=$1 master:="true"
else
    echo "this is slave"
    cd ~/Desktop/ws
    source devel/setup.bash
    source simple_barrett_ws/devel/setup.bash
    roslaunch staubli_barretthand_moveit_config demo.launch id:=$1 master:="false"
fi