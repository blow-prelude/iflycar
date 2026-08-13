#include <AudioRecorder.h>
#include <unistd.h>

using namespace std;
bool AudioRecorder::if_success_boot = false;
bool AudioRecorder::if_awake = false;
hid_device *AudioRecorder::handle = NULL;

AudioRecorder::AudioRecorder(const string &audioPath)
    : mAudioPath(audioPath)
{
    if_success_boot = false;
    if_awake = false;
    is_boot = 0;
    is_reboot = 0;

    handle = hid_open();
    if (handle == NULL)
    {
        if (mic_open_status == -2)
            cout << ">>>>>无法打开麦克风阵列 HID：设备已被占用" << endl;
        else
            cout << ">>>>>无法打开麦克风阵列 HID：未找到设备" << endl;
        return;
    }
    cout << ">>>>>成功打开麦克风阵列 HID" << endl;

    protocol_proc_init(send_to_usb_device, recv_from_usb_device,
                       AudioRecorder::business_proc_callback, err_proc);
    get_system_status();

    /* HID 应答由协议线程异步回调，最多等待 10 秒。 */
    for (int count = 0; count < 100 && !is_boot && !is_reboot; ++count)
    {
        usleep(100000);
    }

    if (is_boot)
    {
        if_success_boot = true;
        cout << ">>>>>麦克风阵列启动成功" << endl;
    }
    else if (is_reboot)
    {
        cout << ">>>>>麦克风阵列正在重启，初始化失败" << endl;
    }
    else
    {
        cout << ">>>>>等待麦克风阵列状态应答超时" << endl;
    }
}

AudioRecorder::~AudioRecorder()
{
    cout << ">>>>>停止所有阵列录音" << endl;
    if (handle != NULL)
    {
        finish_to_record_denoised_sound();
        hid_close();
        handle = NULL;
    }
    if_success_boot = false;
    if_awake = false;
}

bool AudioRecorder::startRecord()
{

    cout << ">>>>>开始录降噪音频\n"
         << endl;
    return true;
}

void AudioRecorder::stopRecord()
{

    cout << ">>>>>停止录降噪音频\n"
         << endl;
}

bool AudioRecorder::startRecordOriginal()
{

    cout << ">>>>>开始录音原始音频\n"
         << endl;
    return true;
}
void AudioRecorder::stopRecordOriginal()
{

    cout << ">>>>>停止录原始音频\n"
         << endl;
}

bool AudioRecorder::setRecordAngle(int angle)
{
    if (!if_success_boot)
    {
        return false;
    }
    // set_mic_angle(angle);
    return true;
}
