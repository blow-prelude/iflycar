#include <AIUITester.h>
#include <vector>
#include <algorithm>
#define MAX_BUFFER 2097152

void gWakeup();
void gSleep();
void clearAudioFile(char *fileName);
static RingBuffer buffer_source(MAX_BUFFER);
unsigned int rate = AUDIO_RATE_SET;

void play()
{
	snd_pcm_t *pcm_handle;
	snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
	snd_pcm_set_params(pcm_handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 2, 44100, 1, 500000);
	FILE *fp = fopen("tts_sample.wav", "rb");
	int buf_size = 1000000;
	char buf[buf_size];
	while (1)
	{
		int size = fread(buf, 1, buf_size, fp);
		if (size == 0)
		{
			break;
		}
		int ret = snd_pcm_writei(pcm_handle, buf, size);
		if (ret == -EPIPE)
		{
			snd_pcm_prepare(pcm_handle);
		}
	}
	snd_pcm_drain(pcm_handle);
	snd_pcm_close(pcm_handle);
}

void *read_and_play1()
{
	if (buffer_source.get_length() > 0)
	{
		read_flag = 1;
		char data_to_speech[200000];
		int res1 = buffer_source.RingBuff_Tx(data_to_speech, buffer_source.get_length());
		read_flag = 0;
		if (res1 != -1)
		{
			globalAudioPlayer->Write((unsigned char *)data_to_speech, res1);
		}
	}
}

void set_awake_word_once(std::string awake_word)
{
	for (int i = 0; i < 5; i++)
	{
		awake_word.pop_back();
	}
	awake_word.erase(0, 1);
	set_awake_word((char *)awake_word.c_str());
}

int AudioRecorder::business_proc_callback(business_msg_t businessMsg)
{
	return 0;
}

void setParams()
{
	char *setParams = "{\"audioparams\":{\"vcn\":\"x2_xiaojuan\"}}";
	IAIUIMessage *setMsg = IAIUIMessage::create(AIUIConstant::CMD_SET_PARAMS, 0, 0, setParams, NULL);
	globalAgent->sendMessage(setMsg);
}

void TestListener::onEvent(const IAIUIEvent &event) const
{
	switch (event.getEventType())
	{
	case AIUIConstant::EVENT_STATE:
		break;
	case AIUIConstant::EVENT_WAKEUP:
		cout << ">>>>>麦克风已唤醒，可进行对话" << endl;
		break;
	case AIUIConstant::EVENT_SLEEP:
		cout << ">>>>>麦克风准备进入休眠模式，将休眠" << endl;
		break;
	case AIUIConstant::EVENT_VAD:
		break;
	case AIUIConstant::EVENT_RESULT:
		break;
	case AIUIConstant::EVENT_CMD_RETURN:
		break;
	case AIUIConstant::EVENT_ERROR:
		cout << "EVENT_ERROR:" << dec << event.getArg1() << endl;
		cout << " ERROR info is " << event.getInfo() << endl;
		break;
	}
}

void AIUITester::createAgent()
{
	Json::Value paramJson;
	Json::Value appidJson;

	appidJson["appid"] = appid;
	appidJson["key"] = key;
	string fileParam = FileUtil::readFileAsString(CFG_FILE_PATH);
	Json::Reader reader;
	if (reader.parse(fileParam, paramJson, false))
	{
		paramJson["login"] = appidJson;
		string wakeup_mode = paramJson["speech"]["wakeup_mode"].asString();
		string engine_type = paramJson["speech"]["intent_engine_type"].asString();

		if (engine_type != "cloud")
		{
			string lgiparams = offline_appid;
			int ret = MSPLogin(NULL, NULL, lgiparams.c_str());
		}
		if (wakeup_mode == "ivw")
		{
			string ivw_res_path = paramJson["ivw"]["res_path"].asString();
			if (!ivw_res_path.empty())
			{
				ivw_res_path = "fo|" + ivw_res_path;
				paramJson["ivw"]["res_path"] = ivw_res_path;
			}
			string ivw_lib_path = "libmsc.so";
			paramJson["ivw"]["msc_lib_path"] = ivw_lib_path;
		}
		Json::FastWriter writer;
		string paramStr = writer.write(paramJson);
		agent = IAIUIAgent::createAgent(paramStr.c_str(), &listener);
		globalAgent = agent;
		globalAudioPlayer = new AudioPlayer();
		setParams();
	}
}

void AIUITester::wakeup()
{
	if (NULL != agent)
	{
		IAIUIMessage *wakeupMsg = IAIUIMessage::create(AIUIConstant::CMD_WAKEUP);
		agent->sendMessage(wakeupMsg);
		wakeupMsg->destroy();
	}
}

void gWakeup()
{
	if (NULL != globalAgent)
	{
		IAIUIMessage *wakeupMsg = IAIUIMessage::create(AIUIConstant::CMD_WAKEUP);
		globalAgent->sendMessage(wakeupMsg);
		wakeupMsg->destroy();
	}
}

void gSleep()
{
	if (NULL != globalAgent)
	{
		IAIUIMessage *sleepMsg = IAIUIMessage::create(AIUIConstant::CMD_RESET_WAKEUP);
		globalAgent->sendMessage(sleepMsg);
		sleepMsg->destroy();
	}
	globalAudioPlayer->Clear_Write();
}

void gTTS(string text)
{
	if (NULL != globalAgent)
	{
		Buffer *textData = Buffer::alloc(text.length());
		text.copy((char *)textData->data(), text.length());
		string paramStr = "vcn=x2_xiaojuan";
		paramStr += ",speed=40";
		paramStr += ",pitch=50";
		paramStr += ",volume=80";
		paramStr += ",aue=speex-wb;7";

		IAIUIMessage *ttsMsg = IAIUIMessage::create(AIUIConstant::CMD_TTS, AIUIConstant::START, 0, paramStr.c_str(), textData);
		globalAgent->sendMessage(ttsMsg);
		ttsMsg->destroy();
	}
}

void AIUITester::start()
{
	if (NULL != agent)
	{
		IAIUIMessage *startMsg = IAIUIMessage::create(AIUIConstant::CMD_START);
		agent->sendMessage(startMsg);
		startMsg->destroy();
	}
}

void AIUITester::stop()
{
	if (NULL != agent)
	{
		IAIUIMessage *stopMsg = IAIUIMessage::create(AIUIConstant::CMD_STOP);
		agent->sendMessage(stopMsg);
		stopMsg->destroy();
	}
}

void AIUITester::reset()
{
	if (NULL != agent)
	{
		IAIUIMessage *resetMsg = IAIUIMessage::create(AIUIConstant::CMD_RESET);
		agent->sendMessage(resetMsg);
		resetMsg->destroy();
	}
}

void AIUITester::destory()
{
	if (audioRecorder != NULL)
	{
		audioRecorder->~AudioRecorder();
		audioRecorder = NULL;
	}

	if (NULL != agent)
	{
		agent->destroy();
		agent = NULL;
	}
}

void AIUITester::recorder_creat()
{
	if (agent == NULL)
		return;
	if (audioRecorder == NULL)
	{
		audioRecorder = new AudioRecorder(TEST_RECORD_PATH);
	}
}

void AIUITester::recorder_start()
{
	bool first_log = false;
	while (!AudioRecorder::if_success_boot)
	{
		if (AudioRecorder::if_awake)
			break;
		if (!first_log)
		{
			cout << ">>>>>请使用唤醒词唤醒\n"
				 << endl;
			first_log = true;
		}
		sleep(1);
	}
}

void AIUITester::recorder_stop()
{
	audioRecorder->stopRecord();
}

void clearAudioFile(char *fileName)
{
	FILE *pFile = fopen(fileName, "w+");
	unsigned char *buf;
	fwrite(buf, sizeof(char), 0, pFile);
	fclose(pFile);
	pFile = NULL;
}

AIUITester::AIUITester() : agent(NULL), audioRecorder(NULL), audioPlayer(NULL) {}

AIUITester::~AIUITester()
{
	if (agent != NULL)
	{
		agent->destroy();
		agent = NULL;
	}
}

void AIUITester::buildGrammar() {}
void AIUITester::updateLocalLexicon() {}
void AIUITester::readCmd() {}

static string MakeMsgPacket(unsigned short sid, MsgType type, const string &content)
{
	const unsigned short size = content.size();
	string data;
	data += (char)0xA5;
	data += (char)0x01;
	data += (char)type;
	data += (char)(size & 0xff);
	data += (char)((size >> 8) & 0xff);
	data += (char)(sid & 0xff);
	data += (char)((sid >> 8) & 0xff);
	data += content;
	int sum = std::accumulate(data.cbegin(), data.cend(), 0);
	data += (char)((~sum + 1) & 0xff);
	return data;
}

static bool UnPackMsgPacket(const string &content, MsgPacket &data)
{
	if (content.size() < 7 || ((unsigned char)content.at(0) != 0xA5))
		return false;
	data.uid = content[1] & 0xff;
	data.type = content[2] & 0xff;
	data.size = ((content[3] & 0xff) | (content[4] << 8 & 0xff00));
	data.sid = ((content[5] & 0xff) | (content[6] << 8 & 0xff00));

	switch ((MsgType)data.type)
	{
	case MsgType::AIUI_MSG:
	{
		string info = content.substr(7, data.size);
		data.bytes = info;
		return true;
	}
	break;
	default:
		break;
	}
	return false;
}

static string MakeComfirm(short sid)
{
	string data;
	data += (char)0xA5;
	data += (char)0x00;
	data += (char)0x00;
	data += (char)0x00;
	return MakeMsgPacket(sid, MsgType::CONFIRM, data);
}

void process_recv(const unsigned char *buf, int len)
{
	if (buf[2] == 0xff)
		return;

	int sum = std::accumulate(buf, buf + len - 1, 0);
	if (((~sum + 1) & 0xff) != buf[len - 1])
		return;

	string data = MakeComfirm(((buf[5] & 0xff) | (buf[6] << 8 & 0xff00)));
	_serial.write(data);

	MsgPacket pkg;
	if (UnPackMsgPacket(string((char *)buf, len), pkg))
	{
		if ((MsgType)pkg.type == MsgType::AIUI_MSG)
		{
			Json::Value WakeupJson;
			Json::Reader reader;

			if (!reader.parse(pkg.bytes, WakeupJson, false))
				return;
			Json::Value content = WakeupJson["content"];

			if (content.isMember("eventType"))
			{
				if (content["eventType"].asInt() == 4)
				{
					string info = content["info"].asString();
					string result = content["result"].asString();
					Json::Reader reader;
					Json::Value root;
					if (reader.parse(info, root))
					{
						angle = root["ivw"]["angle"].asFloat();

						ros::NodeHandle nh;
						int awake = 1;
						nh.param("awake", awake, 0);
						// ================== 核心修改区 ==================
						if (awake == 0) // 继续使用 awake 变量
						{
							// system("aplay /home/ucar/ucar_ws/src/speech_command/audio/wakeup.wav");
							nh.setParam("awake", 1); // 唤醒后，将 awake 设为 1，通知 Python 接管
							cout << "\n==============================================" << endl;
							cout << "✅ 唤醒成功！已释放底层拦截，通知 Python 开始录音..." << endl;
							cout << "==============================================\n"
								 << endl;
						}
						else
						{
							cout << "正在等待 Python 处理任务中..." << endl;
						}
						// ================================================
						printf("awake_angle: %d\n", angle);
					}
				}
			}
		}
	}
}

void uart_rec(const unsigned char *msg, unsigned int msglen)
{
	static int recv_index = 0;
	unsigned char recv_buf[RECV_BUF_LEN];
	static unsigned int big_buf_len = 0;
	static unsigned int big_buf_index = 0;
	static unsigned char *big_buf = NULL;

	if (big_buf == NULL && recv_index + msglen >= 2)
	{
		if (recv_index == 0)
		{
			if (((unsigned char *)msg)[0] != SYNC_HEAD || ((unsigned char *)msg)[1] != SYNC_HEAD_SECOND)
				return;
		}
		else if (recv_index == 1)
		{
			if (recv_buf[0] != SYNC_HEAD || ((unsigned char *)msg)[0] != SYNC_HEAD_SECOND)
			{
				recv_index = 0;
				return;
			}
		}
	}

	int copy_len;
	if (big_buf != NULL)
	{
		copy_len = big_buf_len - big_buf_index < msglen ? big_buf_len - big_buf_index : msglen;
		memcpy(big_buf + big_buf_index, msg, copy_len);
		big_buf_index += copy_len;
		if (big_buf_index < big_buf_len)
			return;
	}
	else
	{
		copy_len = RECV_BUF_LEN - recv_index < msglen ? RECV_BUF_LEN - recv_index : msglen;
		memcpy(recv_buf + recv_index, msg, copy_len);
		if ((recv_index + copy_len) > PACKET_LEN_BIT)
		{
			unsigned int content_len = recv_buf[PACKET_LEN_BIT] << 8 | recv_buf[PACKET_LEN_BIT - 1];
			if (content_len != MSG_NORMAL_LEN)
			{
				big_buf_index = 0;
				big_buf_len = content_len + MSG_EXTRA_LEN;
				big_buf = (unsigned char *)malloc(big_buf_len);
				memset(big_buf, '\0', big_buf_len);
				memcpy(big_buf, recv_buf, recv_index);
				big_buf_index += recv_index;
				recv_index = 0;
				return uart_rec(msg, msglen);
			}
		}
		recv_index += copy_len;
		if (recv_index < RECV_BUF_LEN)
			return;
	}

	if (big_buf != NULL)
	{
		process_recv(big_buf, big_buf_len);
		big_buf_len = 0;
		big_buf_index = 0;
		free(big_buf);
		big_buf = NULL;
	}
	else
	{
		process_recv(recv_buf, RECV_BUF_LEN);
		recv_index = 0;
	}

	if (copy_len < msglen)
	{
		uart_rec(msg + copy_len, msglen - copy_len);
	}
}

void exit_sighandler(int sig)
{
	run_flag = 1;
}

void AIUITester::bind(TEST_CALLBACK callback)
{
	testCallback = callback;
}

// ================== 核心修改区：纯净串口监听模式 ==================
void AIUITester::test()
{
	createAgent();

	printf("\n============================================\n");
	printf("🚀 2026 纯净唤醒模式启动！正在监听串口唤醒信号...\n");
	printf(">>>>> 请喊出唤醒词：小飞小飞\n");
	printf("============================================\n");

	signal(2, exit_sighandler);

	int i = set_awake_word(awake_words);

	while (1)
	{
		static unsigned char buff[1024];
		memset(buff, '\0', 1024);
		int recLen = 0;

		while ((recLen = _serial.read(buff, _serial.available())) > 0)
		{
			uart_rec((const unsigned char *)buff, recLen);
		}

		usleep(20000); // 休眠20ms，释放系统性能，防止占用过高

		if (run_flag)
		{
			printf("收到退出信号，停止监听。\n");
			break;
		}
	}
	_serial.close();
	AIUITester::stop();
	AIUITester::destory();
}
// ================================================================