#include <Global.h>

char awake_words[30] = "小飞小飞";

IAIUIAgent *globalAgent = NULL;
AudioPlayer *globalAudioPlayer = NULL;
char *package_path = NULL;
char *package_path1 = NULL;
serial::Serial _serial;
int angle = 0;
char question[128] = {};
char answer[128] = {};
int sign_conversation_cloud = 0;
int sign_conversation_local = 0;
char *TEST_ROOT_DIR = const_cast<char *>("");
char *TEST_AUDIO_PATH = const_cast<char *>("/config/AIUI/audio/test.pcm");
char *LOG_DIR = const_cast<char *>("/config/AIUI/log");
char *TEST_RECORD_PATH = const_cast<char *>("/config/AIUI/audio/record.pcm");
char *CONFIG_FILE_PATH = const_cast<char *>("/tmp/config.txt");
char *SOURCE_FILE_PATH = const_cast<char *>("/tmp/system.tar");
char *PCM_FILE_PATH = const_cast<char *>("/audio/hid_aiui_deno.pcm");
char *ORIPCM_FILE_PATH = const_cast<char *>("/audio/hid_aiui_ori.pcm");
char *WAKEUP_RESPONSE_WAV = const_cast<char *>("/audio/wakeup.mp3");
char *NO_INTERNET_RESPONSE_WAV = const_cast<char *>("/audio/no_internet.wav");

std::map<std::string, std::string> QA_list_;
std::map<std::string, std::string> QA_list_doc;
bool wait_for_awake_word = false;
bool no_tts = false;
bool sign_angle = false;

int read_flag = 0;
int wakeupflag = 0;
int i = 0;
int err = 0;
snd_pcm_t *capture_handle = NULL;
snd_pcm_hw_params_t *hw_params = NULL;
char *device = const_cast<char *>("2,0");

char rcv_buf[1] = {'0'};
int data_len = 8;
int fd = 0;
int serial_error = 0;

char *pcm_name = const_cast<char *>("hw:XFMDPV0018");
int buffer_frames = 512;
char *buffer1 = NULL;
bool if_print_proc_log = false;
bool if_save_record_file = true;
std::string USER_CONFIG_PATH = "/config/offline_QA.txt";
std::string appid = "d75c9c5d";
std::string key = "8899082b0b85ace67cd84337c2cf0085";
std::string offline_appid = "appid = d75c9c5d";
bool offline_mode = true;
char *GRAMMAR_FILE_PATH = const_cast<char *>("/config/call.bnf");
char *CFG_FILE_PATH = const_cast<char *>("/config/AIUI/cfg/aiui.cfg");
