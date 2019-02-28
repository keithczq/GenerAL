#!/usr/bin/env bash

cd ~/Desktop/ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
roslaunch freenect_launch freenect.launch depth_registration:=true