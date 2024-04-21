# Mobile_Robotics_Project
EECE5550: Mobile Robotics Project Spring 2024 by Adnan Amir, Febin Wilson and Ramez Mubarak

Please refer to the src folder to see the ROS packages used in this project. 

Our m-MPPI test recordings are in the Video Folder. To run these tests you will need a fully setup ROS Noetic workspace with Gazebo and RVIZ installed. Once those are setup please add the mppi package to your catkin_ws/src and call catkin_make. 

Afterward, run the following commands in order, each in a different terminal. 

1) roslaunch turtlebot3_gazebo turtlebot3_house.launch
2) roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
3) rosrun mppi controller.py
