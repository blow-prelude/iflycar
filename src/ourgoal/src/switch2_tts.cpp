#include "switch2.h"

#include <cstdlib>
#include <string>

namespace
{
std::string shellQuote(const std::string &value)
{
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('\'');

    for (const char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted.push_back(ch);
        }
    }

    quoted.push_back('\'');
    return quoted;
}
} // namespace

bool OURSWITCH::speakText(const std::string &text)
{
    if (text.empty())
    {
        ROS_WARN("Skip empty speech text");
        return false;
    }

    const std::string runtime_dir = nh_.param<std::string>(
        "piper_runtime_dir",
        "/home/ucar/ucar_ws/3rdparty/piper");
    const std::string model_path = nh_.param<std::string>(
        "piper_model_path",
        "/home/ucar/ucar_ws/3rdparty/piper/models/zh_CN-huayan-medium.onnx");
    const std::string config_path = nh_.param<std::string>(
        "piper_config_path",
        "/home/ucar/ucar_ws/3rdparty/piper/models/zh_CN-huayan-medium.onnx.json");
    const std::string wav_path = nh_.param<std::string>(
        "piper_wav_path",
        "/tmp/ucar_piper_tts.wav");

    const std::string piper_command =
        "cd " + shellQuote(runtime_dir) +
        " && printf '%s\\n' " + shellQuote(text) +
        " | ./piper --model " + shellQuote(model_path) +
        " --config " + shellQuote(config_path) +
        " --output_file " + shellQuote(wav_path) +
        " && test -s " + shellQuote(wav_path) +
        " && aplay --quiet " + shellQuote(wav_path);

    const int piper_status = std::system(piper_command.c_str());
    if (piper_status == 0)
    {
        ROS_INFO("Piper speech completed: %s", text.c_str());
        return true;
    }

    ROS_WARN(
        "Piper speech failed with status %d; falling back to espeak",
        piper_status);

    const std::string espeak_command =
        "espeak -v zh+f2 -s 130 " + shellQuote(text);
    const int espeak_status = std::system(espeak_command.c_str());
    if (espeak_status == 0)
    {
        ROS_WARN("Speech completed with espeak fallback: %s", text.c_str());
        return true;
    }

    ROS_ERROR(
        "Both Piper and espeak failed (piper=%d, espeak=%d): %s",
        piper_status,
        espeak_status,
        text.c_str());
    return false;
}
