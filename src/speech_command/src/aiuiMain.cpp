#include <AIUITester.h>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>

#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

namespace
{
const int kTestTimeoutSeconds = 10;
const int kAwakeTimeoutSeconds = 10;
bool get_request_test = false;
vector<string> third_line;

/* Keep the strings alive while the legacy SDK uses their char pointers. */
struct RuntimePaths
{
	explicit RuntimePaths(const string &base)
		: package(base),
		  cfg(base + CFG_FILE_PATH),
		  source(base + SOURCE_FILE_PATH),
		  grammar(base + GRAMMAR_FILE_PATH),
		  testAudio(base + TEST_AUDIO_PATH),
		  log(base + LOG_DIR),
		  config(base + CONFIG_FILE_PATH),
		  pcm(base + PCM_FILE_PATH),
		  originalPcm(base + ORIPCM_FILE_PATH),
		  wakeupResponse(base + WAKEUP_RESPONSE_WAV),
		  noInternetResponse(base + NO_INTERNET_RESPONSE_WAV)
	{
	}

	void apply()
	{
		package_path = const_cast<char *>(package.c_str());
		package_path1 = package_path;
		CFG_FILE_PATH = const_cast<char *>(cfg.c_str());
		SOURCE_FILE_PATH = const_cast<char *>(source.c_str());
		GRAMMAR_FILE_PATH = const_cast<char *>(grammar.c_str());
		TEST_AUDIO_PATH = const_cast<char *>(testAudio.c_str());
		LOG_DIR = const_cast<char *>(log.c_str());
		CONFIG_FILE_PATH = const_cast<char *>(config.c_str());
		PCM_FILE_PATH = const_cast<char *>(pcm.c_str());
		ORIPCM_FILE_PATH = const_cast<char *>(originalPcm.c_str());
		WAKEUP_RESPONSE_WAV = const_cast<char *>(wakeupResponse.c_str());
		NO_INTERNET_RESPONSE_WAV = const_cast<char *>(noInternetResponse.c_str());
	}

	string package;
	string cfg;
	string source;
	string grammar;
	string testAudio;
	string log;
	string config;
	string pcm;
	string originalPcm;
	string wakeupResponse;
	string noInternetResponse;
};

bool openSerialPort(const string &port, int baudRate)
{
	try
	{
		_serial.setPort(port);
		_serial.setBaudrate(baudRate);
		_serial.setFlowcontrol(serial::flowcontrol_none);
		_serial.setParity(serial::parity_none);
		_serial.setStopbits(serial::stopbits_one);
		_serial.setBytesize(serial::eightbits);
		serial::Timeout timeout = serial::Timeout::simpleTimeout(1000);
		_serial.setTimeout(timeout);
		_serial.open();
	}
	catch (const std::exception &e)
	{
		ROS_ERROR("无法打开串口 %s：%s", port.c_str(), e.what());
		return false;
	}

	if (!_serial.isOpen())
	{
		ROS_ERROR("无法打开串口 %s", port.c_str());
		return false;
	}

	ros::Duration(0.1).sleep();
	_serial.flush();
	ROS_INFO("串口初始化成功：%s，波特率 %d", port.c_str(), baudRate);
	return true;
}

void awakeWatchdog()
{
	ros::NodeHandle node;
	const ros::WallTime deadline =
		ros::WallTime::now() + ros::WallDuration(kAwakeTimeoutSeconds);
	ros::WallRate waitRate(10);

	while (ros::ok() && ros::WallTime::now() < deadline)
		waitRate.sleep();

	if (!ros::ok())
		return;

	int awake = 0;
	node.param("/awake", awake, 0);
	if (awake == 1)
		return;

	ROS_WARN("启动 %d 秒后仍未收到唤醒信号，自动将 /awake 设置为 1 并结束节点",
			 kAwakeTimeoutSeconds);
	node.setParam("/awake", 1);
	ros::shutdown();
}
} // namespace

int LoadUserConfig(const string &configPath)
{
	ifstream file(configPath.c_str());
	if (!file.is_open())
	{
		ROS_ERROR("无法打开离线语义配置：%s", configPath.c_str());
		return -1;
	}

	string line;
	int lineNumber = 0;
	while (getline(file, line))
	{
		++lineNumber;
		vector<string> fields;
		boost::split(fields, line, boost::is_any_of(":"));
		if (fields.size() != 3)
		{
			ROS_WARN("忽略离线语义配置第 %d 行：格式应为 问题:协议:音频路径", lineNumber);
			continue;
		}

		third_line.push_back(fields[2]);
		vector<string> questions;
		boost::split(questions, fields[0], boost::is_any_of("|"));
		for (vector<string>::const_iterator it = questions.begin(); it != questions.end(); ++it)
		{
			QA_list_[*it] = fields[1];
			QA_list_doc[*it] = fields[2];
		}
	}
	return 0;
}

static string normalizeRecogniseResult(string result)
{
	const string punctuation[] = {"。", "."};
	for (size_t i = 0; i < sizeof(punctuation) / sizeof(punctuation[0]); ++i)
	{
		const size_t position = result.find(punctuation[i]);
		if (position != string::npos)
			result.erase(position, punctuation[i].size());
	}
	return result;
}

string FindKeyWords(string recogniseResult)
{
	const map<string, string>::const_iterator it =
		QA_list_.find(normalizeRecogniseResult(recogniseResult));
	return it == QA_list_.end() ? "" : it->second;
}

string FindDocument(string recogniseResult)
{
	const map<string, string>::const_iterator it =
		QA_list_doc.find(normalizeRecogniseResult(recogniseResult));
	return it == QA_list_doc.end() ? "" : it->second;
}

void data_send()
{
	ros::NodeHandle node;
	ros::Publisher questionPublisher = node.advertise<std_msgs::String>("/question", 10);
	ros::Publisher answerPublisher = node.advertise<std_msgs::String>("/answer", 10);
	ros::Publisher anglePublisher = node.advertise<std_msgs::Int32>("/angle", 10);
	ros::Rate loopRate(100);

	while (ros::ok())
	{
		if (sign_conversation_cloud == 1 || sign_conversation_local == 1)
		{
			const string questionText(question);
			const string answerText(answer);

			std_msgs::String questionMessage;
			questionMessage.data = questionText;
			questionPublisher.publish(questionMessage);

			std_msgs::String answerMessage;
			answerMessage.data = answerText;
			answerPublisher.publish(answerMessage);

			cout << "问题:\t" << questionText << "\n答案:\t" << answerText << endl;
			if (sign_conversation_local == 1)
			{
				const string command = FindKeyWords(questionText);
				const string document = FindDocument(questionText);
				cout << "下发协议:\t" << command << endl;
				if (!document.empty())
				{
					const string playCommand = "play " + string(package_path) + document;
					system(playCommand.c_str());
				}
			}

			sign_conversation_cloud = 0;
			sign_conversation_local = 0;
		}

		if (sign_angle)
		{
			std_msgs::Int32 angleMessage;
			angleMessage.data = angle;
			anglePublisher.publish(angleMessage);
			sign_angle = false;
		}

		ros::spinOnce();
		loopRate.sleep();
	}
}

bool get_test_server(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &response)
{
	ROS_INFO("收到语音测试请求");
	const ros::WallTime start = ros::WallTime::now();
	get_request_test = false;
	ros::WallRate waitRate(100);
	while (ros::ok() && !get_request_test)
	{
		if ((ros::WallTime::now() - start).toSec() > kTestTimeoutSeconds)
		{
			response.success = false;
			response.message = "timeout_error";
			return true;
		}
		waitRate.sleep();
	}
	response.success = get_request_test;
	response.message = get_request_test ? "success" : "shutdown";
	return true;
}

bool get_test_video(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &response)
{
	system("aplay /home/iflytek/ucar_ws/src/speech_command/src/tts_sample.wav &");
	response.success = true;
	response.message = "success";
	return true;
}

void test_callback()
{
	get_request_test = true;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "speech_command_node");
	ros::NodeHandle privateNode("~");

	string serialPort = DEV_ID;
	int baudRate = BAUD_RATE;
	privateNode.param<string>("serial_port", serialPort, serialPort);
	privateNode.param("baud_rate", baudRate, baudRate);

	if (baudRate <= 0)
	{
		ROS_ERROR("启动参数无效：baud_rate=%d", baudRate);
		return 1;
	}

	if (!openSerialPort(serialPort, baudRate))
		return 1;

	const string packagePath = ros::package::getPath("speech_command");
	if (packagePath.empty())
	{
		ROS_ERROR("找不到 ROS 功能包 speech_command");
		_serial.close();
		return 1;
	}

	RuntimePaths paths(packagePath);
	paths.apply();
	LoadUserConfig(packagePath + USER_CONFIG_PATH);

	ROS_INFO("唤醒方式：纯串口（不初始化 USB HID 和 ALSA 录音）");

	AIUITester tester;
	tester.bind(test_callback);

	thread publisherThread(data_send);
	thread awakeWatchdogThread(awakeWatchdog);
	int exitCode = 0;
	try
	{
		tester.test();
	}
	catch (const std::exception &e)
	{
		ROS_ERROR("语音处理异常退出：%s", e.what());
		exitCode = 1;
	}

	ros::shutdown();
	if (awakeWatchdogThread.joinable())
		awakeWatchdogThread.join();
	if (publisherThread.joinable())
		publisherThread.join();
	if (_serial.isOpen())
		_serial.close();
	tester.destory();
	return exitCode;
}
