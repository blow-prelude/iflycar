#ifndef SPEECH_CONTRL_H
#define SPEECH_CONTRL_H

/*
param:
    task 
    target
    play_flag
*/



// 定义物品类别的ID。
#define CMD_Fruits 1170   // 水果
#define CMD_Vegetables 1171   // 蔬菜
#define CMD_sweet 1172 // 甜品

// 定义具体物品ID
// 水果
#define CMD_Apple 530   // 苹果
#define CMD_Banana 609   // 香蕉
#define CMD_Watermelon 1086   // 西瓜

#define CMD_Apple_Apple 1060   // 苹果+苹果
#define CMD_Apple_Banana 1139   // 苹果+香蕉
#define CMD_Apple_Watermelon 1616   // 苹果+西瓜
#define CMD_Banana_Banana 1218   // 香蕉+香蕉
#define CMD_Banana_Watermelon 1695   // 香蕉+西瓜
#define CMD_Watermelon_Watermelon 2172   // 西瓜+西瓜

// 蔬菜 
#define CMD_pepper 652   // 辣椒
#define CMD_Tomato 660   // 西红柿
#define CMD_Potato 663    // 土豆

#define CMD_pepper_pepper 1304   // 辣椒+辣椒
#define CMD_pepper_Tomato 1312   // 辣椒+西红柿
#define CMD_pepper_Potato 1315   // 辣椒+土豆
#define CMD_Tomato_Tomato 1320   // 西红柿+西红柿
#define CMD_Tomato_Potato 1323   // 西红柿+土豆
#define CMD_Potato_Potato 1326   // 土豆+土豆

// 甜品 
#define CMD_Milk 429    // 牛奶
#define CMD_Cake 404    // 蛋糕 
#define CMD_coke 418   // 可乐

#define CMD_Milk_Milk 858   // 牛奶+牛奶
#define CMD_Milk_Cake 833   // 牛奶+蛋糕
#define CMD_Milk_coke 847   // 牛奶+可乐
#define CMD_Cake_Cake 808   // 蛋糕+蛋糕
#define CMD_Cake_coke 822   // 蛋糕+可乐
#define CMD_coke_coke 836   // 可乐+可乐

// 仿真
#define CMD_Gazebo1 851   // 1房间
#define CMD_Gazebo2 852   // 2房间
#define CMD_Gazebo3 853   // 3房间

//路口
#define CMD_Intersection1 951   // 路口1
#define CMD_Intersection2 952   // 路口2

#define CMD_OVER                ((4<<3)+0)

#define PATH_TERRORIST_ONE       /src/test_speech/audio/terrorist_one.wav
#define PATH_TERRORIST_TWO       /src/test_speech/audio/terrorist_two.wav
#define PATH_TERRORIST_THREE     /src/test_speech/audio/terrorist_three.wav
#define PATH_TOOL_ONE            /src/test_speech/audio/get_one.wav
#define PATH_TOOL_TWO            /src/test_speech/audio/get_two.wav
#define PATH_TOOL_THREE          /src/test_speech/audio/get_three.wav
#define PATH_OVER                /src/test_speech/audio/over.wav

int play_audio(int cmd);
int play_flag_srv_handle();

#endif //SPEECH_CONTRL_H
