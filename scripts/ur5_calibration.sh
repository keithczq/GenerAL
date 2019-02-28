#!/usr/bin/env bash

cd ~/calibration_ws
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
roslaunch interactive_calib interactive_marker.launch