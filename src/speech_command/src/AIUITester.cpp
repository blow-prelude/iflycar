#include <AIUITester.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#define MAX_BUFFER 2097152

namespace
{
	void logSerialRx(const unsigned char *data, std::size_t length)
	{
		static std::size_t streamOffset = 0;

		for (std::size_t lineStart = 0; lineStart < length; lineStart += 16)
		{
			const std::size_t lineLength = std::min<std::size_t>(16, length - lineStart);
			std::ostringstream line;
			line << std::hex << std::setfill('0') << std::setw(8)
				 << streamOffset + lineStart << ":";

			for (std::size_t i = 0; i < 16; ++i)
			{
				if (i < lineLength)
					line << " " << std::setw(2) << static_cast<unsigned int>(data[lineStart + i]);
				else
					line << "   ";
			}

			line << "  ";
			for (std::size_t i = 0; i < lineLength; ++i)
			{
				const unsigned char byte = data[lineStart + i];
				line << ((byte >= 0x20 && byte <= 0x7e) ? static_cast<char>(byte) : '.');
			}

			ROS_INFO_STREAM("UART received: " << line.str());
		}

		streamOffset += length;
	}
} // namespace

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
	if (businessMsg.data == NULL)
		return -1;

	/* 阵列唤醒事件。 */
	if (businessMsg.modId == 0x02 && businessMsg.msgId == 0x01)
	{
		unsigned char beam_key[] = "beam";
		unsigned char angle_key[] = "angle";
		const int mic_id = get_awake_mic_id(businessMsg.data, beam_key);
		const int wake_angle = get_awake_mic_angle(businessMsg.data, angle_key);

		if (mic_id >= 0 && mic_id <= 5 && wake_angle >= 0 && wake_angle <= 360)
		{
			::if_awake = 1;
			AudioRecorder::if_awake = true;
			angle = wake_angle;
			sign_angle = true;

			const int target_led = get_led_based_angle(wake_angle);
			set_major_mic_led_on(mic_id, target_led);

			ros::NodeHandle nh;
			int awake = 0;
			nh.param("awake", awake, 0);
			if (awake == 0)
				nh.setParam("awake", 1);

			printf(">>>>>麦克风阵列唤醒成功：麦克风=%d，角度=%d，灯=%d\n",
				   mic_id, wake_angle, target_led);
		}
		else
		{
			printf(">>>>>收到无效的阵列唤醒数据：麦克风=%d，角度=%d\n",
				   mic_id, wake_angle);
		}
	}
	/* 阵列系统状态应答：0 表示正常工作，1 表示正在升级。 */
	else if (businessMsg.modId == 0x03 && businessMsg.msgId == 0x01)
	{
		unsigned char status_key[] = "status";
		const int status = whether_set_succeed(businessMsg.data, status_key);
		if (status == 0)
		{
			is_boot = 5;
			AudioRecorder::if_success_boot = true;
			cout << ">>>>>麦克风阵列状态：正常工作" << endl;
		}
		else if (status == 1)
		{
			cout << ">>>>>麦克风阵列状态：正在升级，暂时不能唤醒" << endl;
		}
		else
		{
			cout << ">>>>>无法解析麦克风阵列系统状态，status=" << status << endl;
		}
	}

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
		delete audioRecorder;
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
					Json::Reader reader;
					Json::Value root;
					if (reader.parse(info, root))
					{
						angle = root["ivw"]["angle"].asFloat();

						ros::NodeHandle nh;
						nh.setParam("/awake", 1);
						ROS_INFO("唤醒成功：角度=%d，已设置 /awake=1，正在退出语音节点", angle);
						ros::requestShutdown();
					}
				}
			}
		}
	}
}

void uart_rec(const unsigned char *msg, unsigned int msglen)
{
	static int recv_index = 0;
	static unsigned char recv_buf[RECV_BUF_LEN];
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

void AIUITester::bind(TEST_CALLBACK callback)
{
	testCallback = callback;
}

void AIUITester::test()
{
	printf("\n============================================\n");
	printf("纯串口唤醒模式启动，正在监听唤醒信号...\n");
	printf(">>>>> 请喊出唤醒词：小飞小飞\n");
	printf("============================================\n");

	while (ros::ok())
	{
		static unsigned char buff[1024];
		while (_serial.available() > 0)
		{
			const size_t bytes_to_read = std::min(_serial.available(), sizeof(buff));
			const size_t recLen = _serial.read(buff, bytes_to_read);
			if (recLen == 0)
				break;
			logSerialRx(buff, recLen);
			uart_rec((const unsigned char *)buff, recLen);
		}

		usleep(20000);
	}

	printf("收到 ROS 退出请求，停止串口监听。\n");
	if (_serial.isOpen())
		_serial.close();
}
