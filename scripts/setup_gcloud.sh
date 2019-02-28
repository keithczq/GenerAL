#!/usr/bin/env bash
sudo apt-get -y update \
&& sudo apt-get -y upgrade \
&& sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
&& sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
&& sudo apt-get -y update \
&& sudo apt-get install -y ros-kinetic-desktop-full \
    screen vim git htop \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential \
    python-catkin-tools \
    build-essential \
    python-pip \
    python3-pip \
    gnupg-curl \
&& sudo rosdep init \
&& rosdep update \
&& sudo apt install mesa-utils \
&& cd ~/Desktop/ws \
&& make pip \
&& sudo pip install --no-dependencies tensorflow-gpu \
&& sudo pip install keras_applications keras_preprocessing pyquaternion ray tensorflow-probability\
&& sudo pip uninstall keyring \
&& mkdir -p ~/.graspit/images \
&& echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc \
&& source ~/.bashrc \
&& cd ~ \
&& sudo sh cuda_10.0.130_410.48_linux.run
