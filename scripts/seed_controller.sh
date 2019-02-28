#!/usr/bin/env bash
cd ~/Desktop/ws/
cd ur5_seed_commander
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
sudo chmod 777 /dev/ttyUSB0 && sudo chmod 777 /dev/ttyUSB1
roslaunch my_seed_hand controller_manager.launch