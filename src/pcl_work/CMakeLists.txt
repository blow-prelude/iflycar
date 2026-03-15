cmake_minimum_required(VERSION 3.0.2)
project(pcl_work)
 
find_package(catkin REQUIRED COMPONENTS
  laser_geometry
  roscpp
  rospy
  std_msgs
  pcl_ros
  pcl_conversions 
  message_generation
  sensor_msgs
)   

find_package(PCL 1.9 REQUIRED)

include_directories( ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_message_files(
  FILES
  ultrasound.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES pcl_work
 CATKIN_DEPENDS laser_geometry roscpp rospy std_msgs  message_runtime 
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(laser2plc src/laser2plc.cpp)
add_executable(ultrasound src/ultrasound.cpp)

add_dependencies(ultrasound pcl_work_generate_messages_cpp)

target_link_libraries(laser2plc
  ${catkin_LIBRARIES}
)

target_link_libraries(ultrasound
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${PCL_COMMON_LIBRARIES} 
  ${PCL_IO_LIBRARIES}
)

