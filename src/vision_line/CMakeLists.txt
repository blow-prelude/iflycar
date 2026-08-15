cmake_minimum_required(VERSION 3.0.2)
project(vision_line)

# # Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# # Find catkin macros and libraries
# # if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
# # is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  std_srvs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)

# RGA 用于硬件色域转换。该库和目标板架构相关，请在目标板或对应交叉编译环境中使用。
set(RGA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/rga)
set(RGA_INCLUDE_DIR ${RGA_ROOT}/include)
# Use the static RGA archive: the bundled shared object requires
# exp@GLIBC_2.29, while the target system provides glibc 2.28.
set(RGA_LIBRARY ${RGA_ROOT}/libs/librga.a)

if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  message(STATUS "OpenCV library status:")
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "OpenCV library not found")
endif()

# # System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# # Uncomment this if the package has a setup.py. This macro ensures
# # modules and global scripts declared therein get installed
# # See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

# ###############################################
# # Declare ROS messages, services and actions ##
# ###############################################

# # To declare and build messages, services or actions from within this
# # package, follow these steps:
# # * Let MSG_DEP_SET be the set of packages whose message types you use in
# #   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
# # * In the file package.xml:
# #   * add a build_depend tag for "message_generation"
# #   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
# #   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
# #     but can be declared for certainty nonetheless:
# #     * add a exec_depend tag for "message_runtime"
# # * In this file (CMakeLists.txt):
# #   * add "message_generation" and every package in MSG_DEP_SET to
# #     find_package(catkin REQUIRED COMPONENTS ...)
# #   * add "message_runtime" and every package in MSG_DEP_SET to
# #     catkin_package(CATKIN_DEPENDS ...)
# #   * uncomment the add_*_files sections below as needed
# #     and list every .msg/.srv/.action file to be processed
# #   * uncomment the generate_messages entry below
# #   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

# # Generate messages in the 'msg' folder
# add_message_files(
# FILES
# Message1.msg
# Message2.msg
# )

# # Generate services in the 'srv' folder
# add_service_files(
# FILES
# Service1.srv
# Service2.srv
# )

# # Generate actions in the 'action' folder
# add_action_files(
# FILES
# Action1.action
# Action2.action
# )

# # Generate added messages and services with any dependencies listed here
# generate_messages(
# DEPENDENCIES
# std_msgs
# )

# ###############################################
# # Declare ROS dynamic reconfigure parameters ##
# ###############################################

# # To declare and build dynamic reconfigure parameters within this
# # package, follow these steps:
# # * In the file package.xml:
# #   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
# # * In this file (CMakeLists.txt):
# #   * add "dynamic_reconfigure" to
# #     find_package(catkin REQUIRED COMPONENTS ...)
# #   * uncomment the "generate_dynamic_reconfigure_options" section below
# #     and list every .cfg file to be processed

# # Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
# cfg/DynReconf1.cfg
# cfg/DynReconf2.cfg
# )

# ##################################
# # catkin specific configuration ##
# ##################################
# # The catkin_package macro generates cmake config files for your package
# # Declare things to be passed to dependent projects
# # INCLUDE_DIRS: uncomment this if your package contains header files
# # LIBRARIES: libraries you create in this project that dependent projects also need
# # CATKIN_DEPENDS: catkin_packages dependent projects also need
# # DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include

  # LIBRARIES vision_line
  # CATKIN_DEPENDS roscpp rospy std_msgs
  # DEPENDS system_lib
)

# ##########
# # Build ##
# ##########

# # Specify additional locations of header files
# # Your package locations should be listed before other locations
# include_directories(
# include
# ${catkin_INCLUDE_DIRS}
# )

# # Declare a C++ library
# add_library(${PROJECT_NAME}
# src/${PROJECT_NAME}/vision_line.cpp
# )

# # Add cmake target dependencies of the library
# # as an example, code may need to be generated before libraries
# # either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# # Declare a C++ executable
# # With catkin_make all packages are built within a single CMake context
# # The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node src/vision_line.cpp)
add_executable(${PROJECT_NAME}_node_fixed src/vision_line_fixed.cpp)
add_executable(${PROJECT_NAME}_node_fixed_pro src/vision_line_fixed_pro.cpp)
add_executable(find_way src/find_way.cpp)
add_executable(find_way_ros src/find_way_ros.cpp)
add_executable(find_way_ros_pro src/find_way_ros_pro.cpp)
add_executable(camera_test src/camera_test.cpp)

add_library(camera_capture src/camera_capture.cpp)
add_library(image_process src/image_process.cpp)

# # Rename C++ executable without prefix
# # The above recommended prefix causes long target names, the following renames the
# # target back to the shorter version for ease of user use
# # e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

# # Add cmake target dependencies of the executable
# # same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# # Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}_node_fixed
  ${catkin_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}_node_fixed_pro
  ${catkin_LIBRARIES}
)
target_link_libraries(find_way
  ${catkin_LIBRARIES}
  image_process
  camera_capture
)
target_link_libraries(find_way_ros
  ${catkin_LIBRARIES}
  image_process
)
target_link_libraries(find_way_ros_pro
  ${catkin_LIBRARIES}
  image_process
)
target_link_libraries(camera_capture
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
target_link_libraries(image_process
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${RGA_LIBRARY}
)

target_link_libraries(camera_test
  ${OpenCV_LIBS}
  camera_capture
)

target_include_directories(${PROJECT_NAME}_node
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(${PROJECT_NAME}_node_fixed
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(${PROJECT_NAME}_node_fixed_pro
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(find_way
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(camera_capture
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(image_process
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${RGA_INCLUDE_DIR}
)
target_include_directories(find_way_ros
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(find_way_ros_pro
  PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(camera_test
  PUBLIC
  include
  ${OpenCV_INCLUDE_DIRS})

add_library(ground_perspective src/ground_perspective.cpp)
target_link_libraries(ground_perspective ${OpenCV_LIBS})
target_include_directories(ground_perspective PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(find_way ground_perspective)
target_link_libraries(find_way_ros ground_perspective)
target_link_libraries(find_way_ros_pro ground_perspective)
target_compile_options(ground_perspective PRIVATE -std=c++11)

add_executable(dynamic_ground_perspective_cpp
  src/dynamic_ground_perspective.cpp)
target_link_libraries(dynamic_ground_perspective_cpp
  camera_capture
  ground_perspective
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS})
target_include_directories(dynamic_ground_perspective_cpp PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
target_compile_options(dynamic_ground_perspective_cpp PRIVATE -std=c++11)

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_ground_perspective
    test/test_ground_perspective.cpp)
  if(TARGET test_ground_perspective)
    target_link_libraries(test_ground_perspective
      gtest_main
      ground_perspective
      ${OpenCV_LIBS})
    target_include_directories(test_ground_perspective PUBLIC
      include
      ${catkin_INCLUDE_DIRS}
      ${OpenCV_INCLUDE_DIRS})
    target_compile_options(test_ground_perspective PRIVATE -std=c++11)
  endif()

  catkin_add_gtest(test_image_process
    test/test_image_process.cpp)
  if(TARGET test_image_process)
    target_link_libraries(test_image_process
      gtest_main
      image_process
      ${OpenCV_LIBS})
    target_include_directories(test_image_process PUBLIC
      include
      ${catkin_INCLUDE_DIRS}
      ${OpenCV_INCLUDE_DIRS})
    target_compile_options(test_image_process PRIVATE -std=c++11)
  endif()
endif()

# ############
# # Install ##
# ############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# # Mark executable scripts (Python etc.) for installation
# # in contrast to setup.py, you can choose the destination

# # Mark executables for installation
# # See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
# RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # Mark libraries for installation
# # See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
# ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

# # Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
# DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# FILES_MATCHING PATTERN "*.h"
# PATTERN ".svn" EXCLUDE
# )

# # Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
# # myfile1
# # myfile2
# DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

# ############
# # Testing ##
# ############

# # Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_vision_line.cpp)
# if(TARGET ${PROJECT_NAME}-test)
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

# # Add folders to be run by python nosetests
# catkin_add_nosetests(test)
