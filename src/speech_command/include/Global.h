#ifndef GLOBAL_AIUI_H_
#define GLOBAL_AIUI_H_

#include <iostream>
#include "aiui/AIUI.h"
#include "FileUtil.h"
#include <AudioPlayer.h>
#include <map>
#include <serial/serial.h>

#define POS_THREAD 1
#define RPY_THREAD 10


extern char awake_words[30];

/************************初始化参数或固定参数，非必要可不修改**********************************/
extern IAIUIAgent* globalAgent;
extern AudioPlayer* globalAudioPlayer;
extern char* package_path;
extern char* package_path1;                                                        // 用于存储功能包的绝对路径
extern serial::Serial _serial;                                                  // 初始化参数,用于串口设置和读写
extern int angle;                                                               // 初始化参数
extern char question[128];                                                      // 初始化参数
extern char answer[128];                                                        // 初始化参数
extern int sign_conversation_cloud;                                         // 初始化参数,当识别结果来源于在线识别引擎时为1,否则为0
extern int sign_conversation_local;                                         // 初始化参数,当识别结果来源于离线识别引擎时为1,否则为0
extern char* TEST_ROOT_DIR;                                         // 初始化参数
//配置文件打的路径，里面是客户端设置的参数
extern char* TEST_AUDIO_PATH;            // 测试音频的路径
extern char* LOG_DIR;                               // 识别引擎log保存路径
extern char* TEST_RECORD_PATH;
extern char* CONFIG_FILE_PATH;
extern char* SOURCE_FILE_PATH;
extern char* PCM_FILE_PATH;                 // 若选择保存音频到本地，则保存到此指定路径,为降噪后的音频
extern char* ORIPCM_FILE_PATH;               // 若选择保存音频到本地，则保存到此指定路径,为原始音频，多通道音频交错分布
extern char* WAKEUP_RESPONSE_WAV;
extern char* NO_INTERNET_RESPONSE_WAV;        // 检测没有网络的警告声音


extern std::map<std::string,std::string> QA_list_;
extern std::map<std::string,std::string> QA_list_doc;
extern bool wait_for_awake_word;                                        // 等待用户说出唤醒词
extern bool no_tts;                                                     // 是否需要在线合成
extern bool sign_angle;                                                 // 麦克风阵列是否被唤醒

extern int read_flag;
extern int wakeupflag;
extern int i;
extern int err;
extern snd_pcm_t *capture_handle;// 一个指向PCM设备的句柄
extern snd_pcm_hw_params_t *hw_params; //此结构包含有关硬件的信息，可用于指定PCM流的配置
extern char* device;

extern char rcv_buf[1];
extern int data_len;
extern int fd;
extern int serial_error;
/*********************************用户可修改参数*******************************************/
#define DEV_ID "/dev/ttyS3"                                            // 外接设备的串口号,若用语音控制机器人运动,则该串口号为机器人下位机
#define BAUD_RATE 115200  

#define PCM_MSG_LEN 1024
#define ORI_PCM_MSG_LEN 16384

#define AudioFormat SND_PCM_FORMAT_S16_LE  //指定音频的格式,其他常用格式：SND_PCM_FORMAT_U24_LE、SND_PCM_FORMAT_U32_LE
#define AUDIO_CHANNEL_SET   1         //1单声道   2立体声
#define AUDIO_RATE_SET 16000   //音频采样率,常用的采样频率: 44100Hz 、16000HZ、8000HZ



#define RECV_BUF_LEN     12
#define MSG_NORMAL_LEN   4
#define MSG_EXTRA_LEN    8
#define PACKET_LEN_BIT   4
#define SYNC_HEAD        0xa5
#define SYNC_HEAD_SECOND 0x01

// rosnode kill /speech_command_node
extern char *pcm_name;                                              // 选择使用的设备型号
//char  *pcm_name = "hw:2,0";                                              // 选择使用的设备型号
extern int buffer_frames;
extern char *buffer1;
extern bool if_print_proc_log;                                          // 是否打印log,调试用
extern bool if_save_record_file;                                         // 是否保存音频到本地
extern std::string USER_CONFIG_PATH;                 // 离线问答对保存的地址
extern std::string appid;                                               // 仅用于在线AIUI，用户可修改，普通用户每天500次
extern std::string key;                         // AIUI引擎Appid对应的AppKey
extern std::string offline_appid;                     //仅用于离线命令词识别I，用户可修改.
extern bool offline_mode;                                                // 若为true则开启离线识别引擎，否则仅为在线识别引擎
extern char* GRAMMAR_FILE_PATH;                     // 离线引擎的语法文件,该变量的路径可不修改，但对应的bnf文件需要用户根据自己期望的待识别词语进行修改，可参看bnf语法
extern char* CFG_FILE_PATH;                // 语音识别配置文件地址,可根据需要修改该cfg文件里的参数
#endif /* GLOBAL_AIUI_H_ */
