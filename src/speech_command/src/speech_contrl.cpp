// 播放音频文件
#include "ros/ros.h"
#include "speech_contrl.h"
#include "std_srvs/Empty.h"

bool is_first[32];  
// 用于记录每个命令是否已经播放过音频。 
// 数组大小为 32，表示最多支持 32 种命令。
// 0：没播放过。
// 1：已经播放。

/*
 音频播放函数 ，用于根据传入的命令编号 cmd 播放对应的音频文件，并在播放完成后设置 ROS 参数 audio 为 1。
 0：成功播放音频。
 -1：播放失败。
*/
int play_audio(int cmd)
{
    
    int res ;
    if(is_first[cmd])  // 如果命令已经播放过音频，则直接返回。
        return 0;
    is_first[cmd]=1;   // 设置该命令已经播放过音频。
    
    switch(cmd)
    {
        case CMD_Fruits:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/fruit.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("ben ci cai gou ren wu wei Fruits\n");
            ros::Duration(3).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
            
            break;

        case CMD_Vegetables:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/vegetable.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("ben ci cai gou ren wu wei Vegetables\n");
            ros::Duration(3).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;

        case CMD_sweet:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/sweet.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("ben ci cai gou ren wu wei sweet\n");
            ros::Duration(3).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
            
            break;
        
        case CMD_Apple:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/apple.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao Apple\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;
        
        case CMD_Banana:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/banana.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao Banana\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;
        
        case CMD_Watermelon:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/watermelon.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao Watermelon\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;
        
        case CMD_pepper:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/pepper.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao pepper\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;
        
        case CMD_Tomato:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/tomato.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao Tomato\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;
        
        case CMD_Potato:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/potato.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao Potato\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;
        
        case CMD_Milk:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/milk.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao Milk\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
        
            break;

        case CMD_Cake:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/cake.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao Cake\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;

        case CMD_coke:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/coke.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi qu dao coke \n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;

        case CMD_Gazebo1:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/room_1.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("fang zhen ren wu yi wan cheng, mu biao huo wu wei yu 1 fang jian \n");
            ros::Duration(5.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
            
        case CMD_Gazebo2:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/room_2.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("fang zhen ren wu yi wan cheng, mu biao huo wu wei yu 2 fang jian \n");
            ros::Duration(5.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Gazebo3:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/room_3.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("fang zhen ren wu yi wan cheng, mu biao huo wu wei yu 3 fang jian\n");
            ros::Duration(5.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;

        case CMD_Intersection1:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/road_1.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("lu kou 1 ke tong guo\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        
        case CMD_Intersection2:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/road_2.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("lu kou 2 ke tong guo\n");
            ros::Duration(1.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;

        case CMD_Apple_Apple:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/apple_apple.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu ,ben ci cai gou huo wu wei Apple_Apple,zong ji hua fei 8 yuan , xu zhao ling 12 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Apple_Banana:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/apple_banana.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu ,ben ci cai gou huo wu wei Apple_Banana,zong ji hua fei 6 yuan , xu zhao ling 14 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Apple_Watermelon:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/apple_watermelon.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Apple_Watermelon,zong ji hua fei 9 yuan, xu zhao ling 11 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Banana_Banana:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/banana_banana.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Banana_Banana,zong ji hua fei 4 yuan, xu zhao ling 16 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Banana_Watermelon:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/banana_watermelon.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Banana_Watermelon,zong ji hua fei 7 yuan, xu zhao ling 14 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Watermelon_Watermelon:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/watermelon_watermelon.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Watermelon_Watermelon,zong ji hua fei 10 yuan, xu zhao ling 10 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_pepper_pepper:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/pepper_pepper.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei pepper_pepper,zong ji hua fei 4 yuan, xu zhao ling 16 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_pepper_Tomato:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/pepper_tomato.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei pepper_Tomato,zong ji hua fei 7 yuan, xu zhao ling 13 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_pepper_Potato:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/pepper_potato.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei pepper_Potato,zong ji hua fei 4 yuan, xu zhao ling 16 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Tomato_Tomato:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/tomato_tomato.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Tomato_Tomato,zong ji hua fei 10 yuan, xu zhao ling 10 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Tomato_Potato:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/tomato_potato.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Tomato_Potato,zong ji hua fei 7 yuan, xu zhao ling 13 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Potato_Potato:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/potato_potato.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Potato_Potato,zong ji hua fei 4 yuan, xu zhao ling 16 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Milk_Milk:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/milk_milk.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Milk_Milk,zong ji hua fei 10 yuan, xu zhao ling 10 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Milk_Cake:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/milk_cake.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Milk_Cake,zong ji hua fei 15 yuan, xu zhao ling 5 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Milk_coke:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/mile_coke.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Milk_coke,zong ji hua fei 8 yuan, xu zhao ling 12 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Cake_Cake:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/cake_cake.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Cake_Cake,zong ji hua fei 20 yuan, xu zhao ling 0 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_Cake_coke:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/cake_coke.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei Cake_coke,zong ji hua fei 13 yuan, xu zhao ling 7 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        case CMD_coke_coke:
            res = system("aplay  /home/ucar/ucar_ws/src/speech_command/audio/task/coke_coke.wav");  // 使用 system("aplay ...") 调用系统命令播放音频。
            printf("wo yi wan cheng huo qu cai gou ren wu,ben ci cai gou huo wu wei coke_coke,zong ji hua fei 6 yuan, xu zhao ling 14 yuan\n");
            ros::Duration(9.5).sleep();  // 使用 ros::Duration(...).sleep() 暂停程序，确保音频播放完成。  时间根据语音时长进行调节
    
            break;
        
        default :
            printf("ERROR: Play audio fail !!! (player.cpp)\n");
            return -1;
    }
    
    
    ros::NodeHandle nh;
    nh.setParam("audio",1);  // 设置 ROS 参数 audio 为 1，表示音频播放完成。
    return 0;
}

/*
这段代码实现了一个 ROS 服务处理函数 play_flag_srv_handle，用于处理 /play_flag_srv 服务的请求。
该服务的主要功能是根据 ROS 参数服务器中的 target_tool 参数，调用 play_audio 函数播放对应的音频文件
true：服务处理成功。
false：服务处理失败。
*/
bool play_flag_srv_handle(std_srvs::Empty::Request& req,
                         std_srvs::Empty::Response& res)
{
    printf("start play_handle() ...\n");

    ros::NodeHandle nh;
    int task_id, target_id, cmd;
    

    // task_id = nh.param("task",0);
    target_id = nh.param("target_tool",0);
    // target_id = nh.param("colorRGBA",0);

    printf("task_id:%d, target_id:%d \n",task_id,target_id);

    if(target_id>0)
        play_audio(target_id);
        // play_audio( (task_id==4?(task_id<<3):((task_id<<3) + target_id)));
    else{
        printf("ERROR: GET CMD Fail !!! (player.cpp)\n");
        printf("task_id>0:%d --- target_id>0:%d  ---->  %d \n",task_id>0,target_id>0,(task_id>0 && target_id>0));
        return 1;
    }
    printf("语音播报完成！\n");
    //reset
    // nh.setparam("play_flag",0);
    return 1;
}

int main(int argc, char** argv)
 {
    ros::init(argc, argv, "speech_contrl_node");

    ros::NodeHandle nh;
    ros::ServiceServer play_flag_srv = nh.advertiseService("/play_flag_srv",play_flag_srv_handle);
    //listen
    while(!nh.param("awake",0));   //如果 awake 为 0，则继续等待；否则，退出循环。  该逻辑用于等待系统唤醒信号。
    system("rosnode kill /speech_command_node");   // 使用系统命令 rosnode kill 杀死名为 /speech_command_node 的 ROS 节点。   该操作可能是为了释放资源或确保系统状态正确。
    printf("Awake !\n");

    
    printf("play_flag_srv on!\n");
    ros::spin();

    return 0;
}
