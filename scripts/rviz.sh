#!/bin/bash
source /opt/ros/indigo/setup.bash
export PYTHONPATH=$PYTHONPATH:/usr/lib
export PATH=$PATH:$HOME/.local/bin

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion


export ROS_HOSTNAME=$(hostname -f).local
if [ $1 = "barrett" ]; then
    cd ~/Desktop/ws/staubli_barrett_ws
else
    cd ~/Desktop/ws/ur5_seed_commander
fi
source devel/setup.bash
clear
export ROS_MASTER_URI=http://localhost:$1
if [ $2 = "barrett" ]; then
    rosrun rviz rviz -d /home/bohan/.rviz/grasping_rl.rviz
elif [ $1 = "11311" ]; then
    rosrun rviz rviz -d /home/bohan/Desktop/ws/ur5_seed_commander/src/ur5seed_moveit_config/launch/moveit.rviz
elif [ $1 = "11312" ]; then
    rosrun rviz rviz -d /home/bohan/Desktop/ws/ur5_seed_commander/src/ur5seed_moveit_config/launch/moveit.rviz
fi

