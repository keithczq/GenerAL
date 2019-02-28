#!/usr/bin/env bash

cd ~/Desktop/ws/ur5_seed_commander
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:$1
if [ $1 = "11311" ]; then
    echo "this is master"
    roslaunch ur5seed_moveit_config ur5_moveit_planning_execution.launch sim:=false id:=$1 master:="true"
else
    echo "this is slave"
    roslaunch ur5seed_moveit_config demo.launch sim:=false id:=$1 master:="false"
fi