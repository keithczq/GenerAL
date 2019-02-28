#!/bin/bash
source /opt/ros/kinetic/setup.bash
cd ~/Desktop/ws
export PATH=$PATH:~/Desktop/ws:~/Desktop/clion-2018.2.3/bin:~/Desktop/idea-IU-182.4323.46/bin:/usr/local/bin
export LD_LIBRARY_PATH=/usr/lib:/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:~/Desktop/ws/models
export GRASPIT=~/.graspit
export COINDIR="/usr/local"
export INCLUDE="/usr/local/include:/usr/local/Trolltech/Qt-4.8.6/include"
source devel/setup.bash
screen -wipe
source simple_barrett_ws/devel/setup.bash
clear
screen -ls
nvidia-smi
export ROS_HOSTNAME=$(hostname -f).local
if [ $1 = "11311" ] && [ $2 = "barrett" ]; then
    export ROS_MASTER_URI=http://192.168.50.183:$1
    echo "this is master"
    roslaunch grasping_project everything.launch master:="true" hand:="barrett"
elif [ $1 != "11311" ] && [ $2 = "barrett" ]; then
    export ROS_MASTER_URI=http://localhost:$1
    echo "this is slave"
    roslaunch grasping_project everything.launch master:="false" hand:="barrett"
elif [ $1 = "11311" ] && [ $2 = "seed" ]; then
    export ROS_MASTER_URI=http://localhost:$1
    echo "this is master"
    roslaunch grasping_project everything.launch master:="true" hand:="seed"
else
    export ROS_MASTER_URI=http://localhost:$1
    echo "this is slave"
    roslaunch grasping_project everything.launch master:="false" hand:="seed"
fi