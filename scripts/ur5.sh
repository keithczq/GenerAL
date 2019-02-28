#!/usr/bin/env bash
cd ~/Desktop/ws/
cd ur5_seed_commander
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
cd ..
make reall