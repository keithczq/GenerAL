#!/usr/bin/env bash
cd ~/Desktop/ws/ur5_seed_commander
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
cd src
python my_seed_hand/motion_controller/action_client/simple_client.py