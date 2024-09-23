#!/bin/bash

# Add ROS repository to sources list
echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# Add the ROS key
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Update package list and install ROS Noetic packages
apt update
apt install -y ros-noetic-ros-core ros-noetic-camera-info-manager \
    ros-noetic-diagnostic-updater \
    ros-noetic-roslint ros-noetic-rqt-image-view \
    ros-noetic-tf2-eigen ros-noetic-tf2 ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs ros-noetic-rviz \
    python3-catkin-tools python3-osrf-pycommon \
    python3-rosdep ros-noetic-rqt-multiplot \
    ros-noetic-imu-filter-madgwick

# Setup environment
# Check if .bash_aliases file exists, create it if it doesn't
if [ ! -f "/home/asl/.bash_aliases" ]; then
    touch /home/asl/.bash_aliases
fi

echo "source /opt/ros/noetic/setup.bash" >> /home/asl/.bash_aliases
echo 'if [ -f "/workspaces/paper/ros_workspace/devel/setup.bash" ]; then source "/workspaces/paper/ros_workspace/devel/setup.bash"; fi' >> /home/asl/.bash_aliases
