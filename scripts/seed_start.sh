#!/usr/bin/env bash
cd ~/Desktop/ws/ur5_seed_commander
source devel/setup.bash
export ROS_HOSTNAME=$(hostname -f).local
export ROS_MASTER_URI=http://localhost:11311
roslaunch my_seed_hand start_meta_controller.launch