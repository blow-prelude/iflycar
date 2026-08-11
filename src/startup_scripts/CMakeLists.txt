cmake_minimum_required(VERSION 3.0.2)
project(startup_scripts)

find_package(catkin REQUIRED COMPONENTS rospy std_srvs)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/managed_nodes_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
