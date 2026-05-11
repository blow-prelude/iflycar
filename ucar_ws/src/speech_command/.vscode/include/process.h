#include <ros/ros.h>
#include <unistd.h>
#include "rknn_yolov5_demo/index_names.h"
#include "rknn_yolov5_demo/yolov5_results.h"
#include <iostream>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <math.h>
#include <stdio.h>
#include <set>
#include "bridge/align.h"
#include "bridge/com.h"
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

#include <sensor_msgs/LaserScan.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

#include <boost/thread.hpp>
#include <string>
#include <vector>
#include <map>
#include <algorithm> // 包含std::remove算法
#include <numeric> 

using namespace std;

class Process {
    public:
    Process();
    ~Process()=default;


    void SendPos(move_base_msgs::MoveBaseGoal goal);

    bool in_range(float x, pair<float, float> range);


    bool in_room_range(pair<pair<float, float>, pair<float, float>> range, pair<float, float>);

    //void amcl_pose_CallBack(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg);

    void room_judge();

    pair<float,float> get_pos();

    void yolo_callback(const rknn_yolov5_demo::yolov5_results::ConstPtr &msg_p);

    void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_msg);

    void logic_judge();

    void cancel_plan();

    void reach_in_1_sec();

    void PID_realize();

    int find_majority(vector<int> values);

    map<int, int> classification(vector<int> values);



    void pitch_process();

    void F_classification(vector<int> values);

    void audio_play();

    void complement();
    
    pair<vector<string>, int> map_check();

    vector<float> get_ratio(pair<vector<string>, int> vec_room);

    pair<int, int> fruit_count();






    public:
    move_base_msgs::MoveBaseGoal goalB;
    move_base_msgs::MoveBaseGoal goalC;
    move_base_msgs::MoveBaseGoal goalD;
    move_base_msgs::MoveBaseGoal goalE;
    move_base_msgs::MoveBaseGoal goal_F;
    move_base_msgs::MoveBaseGoal goalF;
    move_base_msgs::MoveBaseGoal goalF_2;
    move_base_msgs::MoveBaseGoal goalhome;
    //move_base_msgs::MoveBaseGoal empty_goal;
    pair<pair<float, float>, pair<float, float>> range_B;
    pair<pair<float, float>, pair<float, float>> range_C;
    pair<pair<float, float>, pair<float, float>> range_D;
    pair<pair<float, float>, pair<float, float>> range_E;

    pair<pair<float, float>, pair<float, float>> range_F;

    pair<pair<float, float>, pair<float, float>> range_O;

    vector<int> vecB;
    vector<int> vecC;
    vector<int> vecD;
    vector<int> vecE;
    vector<int> vecF_1;
    vector<int> vecF_2;


    map<string,int> map_1;

    map<int, string> map_audio;

    map<int, string> map_;

    map<int, pair<int,int>> map_F;

    set<int> fruit_count_set;

    pair<map<int, int>, set<int>> set_F; //first记录每种个数，second记录名录
    pair<map<int, int>, set<int>> set_F2;
    //rknn_yolov5_demo::yolov5_results rslt;



    public:
    int vecF1_cnt = 2;

    bool break_flag=false, rcv_flag =false, F_rotation_flag = false;
    string room,last_room, goal_;

    float dst_L,dst_H;

    const float tgt_H,tgt_L;

    bool pitch_flag = false, cnt_flag = false, reach_flag = false;

    geometry_msgs::PoseWithCovarianceStamped poseF;
    
    
    ros::NodeHandle nh;
    //actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac;
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac;
    ros::Subscriber yolo_sub_;
    ros::Subscriber laser_sub_;
    ros::Publisher vel_pub_ ;
    ros::Publisher initial_pose_ori_pub_;

    ros::ServiceClient align_client_;
    ros::ServiceClient rotation_client_;
    ros::ServiceClient param_reload_client_;

    tf::TransformListener listener_;
    tf::StampedTransform transform_;

    geometry_msgs::Twist cmd_vel_;

    bridge::com angle;
    bridge::align al;
    
    private:
    boost::thread* judgeThread_;
    boost::thread* logicThread_;

};