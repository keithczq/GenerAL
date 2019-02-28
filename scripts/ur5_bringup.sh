#!/usr/bin/env bash

cd ~/Desktop/ws/ur5_seed_commander
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.50.81