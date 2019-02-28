#!/usr/bin/env bash
sudo apt-get -y update \
&& sudo apt-get -y upgrade \
&& sudo apt-get -y install broadcom-sta-dkms \
&& sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
&& sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 \
&& sudo apt-get -y update \
&& sudo apt-get install -y ros-kinetic-desktop-full \
    screen vim git htop \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential \
    libqt4-dev \
    libqt4-opengl-dev \
    libqt4-sql-psql \
    libcoin80-dev \
    libsoqt4-dev \
    libblas-dev \
    liblapack-dev \
    libqhull-dev \
    libeigen3-dev \
    libglfw3-dev \
    python-catkin-tools \
    dkms \
    synaptic \
    emacs \
    build-essential \
    python-pip \
&& sudo rosdep init \
&& rosdep update \
&& sudo apt install mesa-utils \
&& cd ~/Desktop/graspit \
&& export GRASPIT=$PWD \
&& mkdir build \
&& cd build \
&& cmake .. \
&& make -j5 \
&& sudo make install \
&& cd ~/Desktop/ws \
&& make pip \
&& source ~/.bashrc \
&& catkin_make \
&& sudo chmod -R 777 ~/.graspit/ \
&& sudo chmod -R 777 ~/.ros/ \
&& source ../devel/setup.bash \
&& sudo ln -s /home/bohan/Desktop/sublime_text_3/sublime_text /usr/local/bin/subl \
&& git config --global user.email "bw2505@columbia.edu" \
&& git config --global user.name "Bohan Wu"
&& pip install --upgrade tensorflow-probability-gpu --user \
&& sudo pip uninstall -y tensorflow \
&& sudo pip install --upgrade tensorflow-gpu --user \
&& sudo add-apt-repository ppa:graphics-drivers/ppa \
&& sudo apt-get update \
&& sudo apt-get install -y nvidia-387 \
&& cd \
&& wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run \
&& chmod +x cuda_9.0.176_384.81_linux-run \
&& ./cuda_9.0.176_384.81_linux-run --extract=$HOME \
&& sudo ./cuda-samples.9.0.176-22781540-linux.run \
&& sudo bash -c "echo /usr/local/cuda-9.0/lib64/ > /etc/ld.so.conf.d/cuda.conf" \
&& sudo ldconfig \
&& cd ~/Downloads/ \
&& sudo dpkg -i libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb \
&& sudo dpkg -i libcudnn7-dev_7.3.1.20-1+cuda9.0_amd64.deb \
&& sudo dpkg -i libcudnn7-doc_7.3.1.20-1+cuda9.0_amd64.deb \
&& sudo reboot \
&& cd /usr/local/cuda-9.0/samples \
&& sudo make
