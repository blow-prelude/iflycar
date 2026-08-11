#include "camera_capture.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <linux/videodev2.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>

namespace
{
    struct ControlDescription
    {
        __u32 id;
        const char *name;
    };

    bool v4l2_ioctl(int fd, unsigned long request, void *argument)
    {
        int result;
        do
        {
            result = ioctl(fd, request, argument);
        } while (result == -1 && errno == EINTR);
        return result != -1;
    }

    std::string fourcc_to_string(__u32 fourcc)
    {
        std::string value(4, ' ');
        value[0] = static_cast<char>(fourcc & 0xff);
        value[1] = static_cast<char>((fourcc >> 8) & 0xff);
        value[2] = static_cast<char>((fourcc >> 16) & 0xff);
        value[3] = static_cast<char>((fourcc >> 24) & 0xff);
        return value;
    }

    double fraction_to_fps(const v4l2_fract &interval)
    {
        if (interval.numerator == 0)
        {
            return 0.0;
        }
        return static_cast<double>(interval.denominator) / interval.numerator;
    }

    std::string fps_to_string(const v4l2_fract &interval)
    {
        std::ostringstream text;
        text << std::fixed << std::setprecision(2) << fraction_to_fps(interval);
        return text.str();
    }

    bool same_fraction(const v4l2_fract &left, const v4l2_fract &right)
    {
        return static_cast<unsigned long long>(left.numerator) * right.denominator ==
               static_cast<unsigned long long>(right.numerator) * left.denominator;
    }

    std::string enumerate_frame_intervals(int fd, __u32 pixel_format, __u32 width,
                                          __u32 height, bool &has_adjustable_fps)
    {
        std::ostringstream result;
        v4l2_frmivalenum interval;
        std::memset(&interval, 0, sizeof(interval));
        interval.pixel_format = pixel_format;
        interval.width = width;
        interval.height = height;

        bool found = false;
        v4l2_fract first_interval = {};
        for (interval.index = 0; v4l2_ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &interval);
             ++interval.index)
        {
            if (found)
            {
                result << ", ";
            }

            if (interval.type == V4L2_FRMIVAL_TYPE_DISCRETE)
            {
                result << fps_to_string(interval.discrete);
                if (!found)
                {
                    first_interval = interval.discrete;
                }
                else if (!same_fraction(first_interval, interval.discrete))
                {
                    has_adjustable_fps = true;
                }
            }
            else
            {
                const v4l2_frmival_stepwise &range = interval.stepwise;
                result << fps_to_string(range.max) << "-" << fps_to_string(range.min);
                if (interval.type == V4L2_FRMIVAL_TYPE_STEPWISE)
                {
                    result << " (step interval " << range.step.numerator << "/"
                           << range.step.denominator << " s)";
                }
                else
                {
                    result << " (continuous)";
                }
                if (!same_fraction(range.min, range.max))
                {
                    has_adjustable_fps = true;
                }
            }
            found = true;
        }

        return found ? result.str() : "driver did not enumerate";
    }

    const char *control_type_name(__u32 type)
    {
        switch (type)
        {
        case V4L2_CTRL_TYPE_INTEGER:
            return "integer";
        case V4L2_CTRL_TYPE_BOOLEAN:
            return "boolean";
        case V4L2_CTRL_TYPE_MENU:
            return "menu";
        case V4L2_CTRL_TYPE_BUTTON:
            return "button";
        case V4L2_CTRL_TYPE_INTEGER64:
            return "integer64";
        case V4L2_CTRL_TYPE_CTRL_CLASS:
            return "class";
        case V4L2_CTRL_TYPE_STRING:
            return "string";
        case V4L2_CTRL_TYPE_BITMASK:
            return "bitmask";
        case V4L2_CTRL_TYPE_INTEGER_MENU:
            return "integer menu";
        default:
            return "other";
        }
    }

    void print_control_details(int fd, const v4l2_queryctrl &query,
                               const char *display_name, std::ostream &out)
    {
        const bool disabled = (query.flags & V4L2_CTRL_FLAG_DISABLED) != 0;
        const bool read_only = (query.flags & V4L2_CTRL_FLAG_READ_ONLY) != 0;
        const bool inactive = (query.flags & V4L2_CTRL_FLAG_INACTIVE) != 0;
        const bool grabbed = (query.flags & V4L2_CTRL_FLAG_GRABBED) != 0;

        out << "  - " << display_name << ": ";
        if (disabled)
        {
            out << "unsupported (disabled by driver)\n";
            return;
        }

        out << (read_only ? "supported, read-only" : "adjustable");
        if (inactive)
        {
            out << ", currently inactive";
        }
        if (grabbed)
        {
            out << ", currently busy";
        }
        out << " [" << control_type_name(query.type) << "]";

        if (query.type == V4L2_CTRL_TYPE_INTEGER ||
            query.type == V4L2_CTRL_TYPE_BOOLEAN ||
            query.type == V4L2_CTRL_TYPE_MENU ||
            query.type == V4L2_CTRL_TYPE_INTEGER_MENU ||
            query.type == V4L2_CTRL_TYPE_BITMASK)
        {
            out << ", range=" << query.minimum << ".." << query.maximum
                << ", step=" << query.step << ", default=" << query.default_value;

            v4l2_control control;
            std::memset(&control, 0, sizeof(control));
            control.id = query.id;
            if (v4l2_ioctl(fd, VIDIOC_G_CTRL, &control))
            {
                out << ", current=" << control.value;
                if (query.type == V4L2_CTRL_TYPE_MENU ||
                    query.type == V4L2_CTRL_TYPE_INTEGER_MENU)
                {
                    v4l2_querymenu menu;
                    std::memset(&menu, 0, sizeof(menu));
                    menu.id = query.id;
                    menu.index = control.value;
                    if (v4l2_ioctl(fd, VIDIOC_QUERYMENU, &menu))
                    {
                        if (query.type == V4L2_CTRL_TYPE_MENU)
                        {
                            out << " (" << reinterpret_cast<const char *>(menu.name) << ")";
                        }
                        else
                        {
                            out << " (" << menu.value << ")";
                        }
                    }
                }
            }
        }
        out << '\n';
    }

    bool query_and_print_control(int fd, const ControlDescription &description,
                                 std::ostream &out)
    {
        v4l2_queryctrl query;
        std::memset(&query, 0, sizeof(query));
        query.id = description.id;
        if (!v4l2_ioctl(fd, VIDIOC_QUERYCTRL, &query) ||
            (query.flags & V4L2_CTRL_FLAG_DISABLED) != 0)
        {
            out << "  - " << description.name << ": unsupported\n";
            return false;
        }

        print_control_details(fd, query, description.name, out);
        return true;
    }

    bool open_camera(cv::VideoCapture &capture, int camera_index, int width, int height,
                     bool convert_rgb)
    {
        if (!capture.open(camera_index, cv::CAP_V4L2))
        {
            return false;
        }

        capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
        capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        capture.set(cv::CAP_PROP_FPS, 30);
        capture.set(cv::CAP_PROP_BUFFERSIZE, 3);
        capture.set(cv::CAP_PROP_CONVERT_RGB, convert_rgb ? 1 : 0);
        return true;
    }
} // namespace

cv::Mat CameraCapture::perspective_matrix = (cv::Mat_<double>(3, 3) << -0.493484, -1.133680, 215.265780,
                                             -0.052136, -2.137570, 309.269947,
                                             -0.000221, -0.007728, 1.000000);
CameraCapture::CameraCapture()
{

    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, true))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::CameraCapture(int index, int width, int height) : camera_index(index), frame_width(width), frame_height(height)
{
    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, true))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::CameraCapture(int index, int width, int height, cv::Mat mtx, cv::Mat dist) : camera_index(index), frame_width(width), frame_height(height), mtx(mtx), dist(dist)
{
    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, true))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::CameraCapture(int index, int width, int height, cv::Mat mtx, cv::Mat dist, bool convert_rgb) : camera_index(index), frame_width(width), frame_height(height), mtx(mtx), dist(dist)
{
    cap = new cv::VideoCapture();
    if (!open_camera(*cap, camera_index, frame_width, frame_height, convert_rgb))
    {
        std::cerr << "Error: Could not open camera with index " << camera_index << std::endl;
        throw std::runtime_error("Could not open camera");
    }

    // frame = cv::Mat(frame_height, frame_width, CV_8UC3);
    std::cout << "Camera opened successfully with index " << camera_index << " and resolution " << frame_width << "x" << frame_height << std::endl;
}

CameraCapture::~CameraCapture()
{
    closeCamera();
    std::cout << "CameraCapture object destroyed, camera released." << std::endl;
}

cv::Mat CameraCapture::captureFrame()
{
    if (!cap->isOpened())
    {
        throw std::runtime_error("Camera is not opened");
    }

    cv::Mat frame;
    *cap >> frame;
    return frame;
}

cv::Mat CameraCapture::correctFrame(const cv::Mat &frame)
{
    try
    {
        if (this->mtx.empty() || this->dist.empty())
        {
            throw std::invalid_argument("Camera matrix or distortion coefficients are empty");
        }

        int h = frame.rows;
        int w = frame.cols;
        // 计算新的内参矩阵
        cv::Rect roi;
        cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(this->mtx, this->dist, cv::Size(w, h), 0, cv::Size(w, h), &roi);

        // 生成去畸变表
        cv::Mat mapx, mapy;
        cv::initUndistortRectifyMap(this->mtx, this->dist, cv::Mat(), newCameraMatrix, cv::Size(w, h), CV_32FC1, mapx, mapy);

        // 应用去畸变映射
        cv::Mat dst;
        cv::remap(frame, dst, mapx, mapy, cv::INTER_LINEAR);

        return dst;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in correctFrame: " << e.what() << std::endl;
        return frame; // 返回原始帧
    }
}

cv::Mat CameraCapture::perspectiveFrame(const cv::Mat &frame)
{
    try
    {
        if (perspective_matrix.empty())
        {
            throw std::invalid_argument("Perspective matrix is empty");
        }

        cv::Mat dst;
        cv::warpPerspective(frame, dst, perspective_matrix, cv::Size(160, 144), cv::INTER_LINEAR);

        return dst;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in perspectiveFrame: " << e.what() << std::endl;
        return frame;
    }
}

void CameraCapture::closeCamera()
{
    if (cap->isOpened())
    {
        cap->release();
    }
}

void CameraCapture::printCameraPropertySupport(std::ostream &out) const
{
    const std::string device = "/dev/video" + std::to_string(camera_index);
    int fd = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd == -1)
    {
        // Some systems grant query access but not write access. Control flags still
        // describe whether the driver supports writing each property.
        fd = open(device.c_str(), O_RDONLY | O_NONBLOCK);
    }
    if (fd == -1)
    {
        out << "Cannot query camera capabilities for " << device << ": "
            << std::strerror(errno) << '\n';
        return;
    }

    out << "\n=== Camera property support: " << device << " ===\n";

    v4l2_capability capability;
    std::memset(&capability, 0, sizeof(capability));
    if (!v4l2_ioctl(fd, VIDIOC_QUERYCAP, &capability))
    {
        out << "VIDIOC_QUERYCAP failed: " << std::strerror(errno) << '\n';
        close(fd);
        return;
    }

    out << "Device: " << reinterpret_cast<const char *>(capability.card)
        << ", driver: " << reinterpret_cast<const char *>(capability.driver) << '\n';

    __u32 capabilities = capability.capabilities;
    if ((capabilities & V4L2_CAP_DEVICE_CAPS) != 0)
    {
        capabilities = capability.device_caps;
    }

    v4l2_buf_type buffer_type;
    if ((capabilities & V4L2_CAP_VIDEO_CAPTURE) != 0)
    {
        buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    }
    else if ((capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0)
    {
        buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    }
    else
    {
        out << "This device does not report a video-capture capability.\n";
        close(fd);
        return;
    }

    // V4L2 formats are scoped to each open file handle. Read the active format
    // from OpenCV's handle instead of reporting the probe handle's defaults.
    if (cap != nullptr && cap->isOpened())
    {
        const __u32 current_fourcc = static_cast<__u32>(cap->get(cv::CAP_PROP_FOURCC));
        std::ostringstream current_fps;
        current_fps << std::fixed << std::setprecision(2) << cap->get(cv::CAP_PROP_FPS);
        out << "Current format: "
            << static_cast<int>(cap->get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
            << static_cast<int>(cap->get(cv::CAP_PROP_FRAME_HEIGHT)) << " "
            << fourcc_to_string(current_fourcc) << '\n';
        out << "Current frame rate: " << current_fps.str() << " FPS\n";
    }

    bool can_set_time_per_frame = false;
    v4l2_streamparm stream_parameters;
    std::memset(&stream_parameters, 0, sizeof(stream_parameters));
    stream_parameters.type = buffer_type;
    if (v4l2_ioctl(fd, VIDIOC_G_PARM, &stream_parameters))
    {
        const v4l2_captureparm &capture_parameters = stream_parameters.parm.capture;
        can_set_time_per_frame =
            (capture_parameters.capability & V4L2_CAP_TIMEPERFRAME) != 0;
    }

    std::set<std::pair<__u32, __u32> > discrete_resolutions;
    bool has_resolution_range = false;
    bool has_adjustable_fps = false;
    bool found_format = false;
    std::ostringstream modes;

    v4l2_fmtdesc format;
    std::memset(&format, 0, sizeof(format));
    format.type = buffer_type;
    for (format.index = 0; v4l2_ioctl(fd, VIDIOC_ENUM_FMT, &format); ++format.index)
    {
        found_format = true;
        modes << "  " << fourcc_to_string(format.pixelformat) << " ("
              << reinterpret_cast<const char *>(format.description) << ")\n";

        v4l2_frmsizeenum frame_size;
        std::memset(&frame_size, 0, sizeof(frame_size));
        frame_size.pixel_format = format.pixelformat;
        bool found_frame_size = false;
        for (frame_size.index = 0;
             v4l2_ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frame_size);
             ++frame_size.index)
        {
            found_frame_size = true;
            if (frame_size.type == V4L2_FRMSIZE_TYPE_DISCRETE)
            {
                const __u32 width = frame_size.discrete.width;
                const __u32 height = frame_size.discrete.height;
                discrete_resolutions.insert(std::make_pair(width, height));
                modes << "    " << width << "x" << height << " @ "
                      << enumerate_frame_intervals(fd, format.pixelformat, width, height,
                                                   has_adjustable_fps)
                      << " FPS\n";
            }
            else
            {
                const v4l2_frmsize_stepwise &range = frame_size.stepwise;
                modes << "    " << range.min_width << "x" << range.min_height << " - "
                      << range.max_width << "x" << range.max_height;
                if (frame_size.type == V4L2_FRMSIZE_TYPE_STEPWISE)
                {
                    modes << " (step " << range.step_width << "x" << range.step_height << ")";
                }
                else
                {
                    modes << " (continuous)";
                }
                modes << '\n';
                if (range.min_width != range.max_width || range.min_height != range.max_height)
                {
                    has_resolution_range = true;
                }
            }
        }
        if (!found_frame_size)
        {
            modes << "    driver did not enumerate resolutions\n";
        }
    }

    const bool has_adjustable_resolution =
        has_resolution_range || discrete_resolutions.size() > 1;
    out << "\nVideo format controls:\n"
        << "  - Resolution: " << (found_format ? "supported" : "not enumerated")
        << ", multiple choices: " << (has_adjustable_resolution ? "yes" : "no") << '\n'
        << "  - Frame rate: "
        << ((can_set_time_per_frame || has_adjustable_fps) ? "adjustable" : "not adjustable")
        << " (V4L2_CAP_TIMEPERFRAME=" << (can_set_time_per_frame ? "yes" : "no") << ")\n";
    if (found_format)
    {
        out << "Supported modes (frame rates):\n" << modes.str();
    }

    const ControlDescription common_controls[] = {
        {V4L2_CID_BRIGHTNESS, "Brightness / 亮度"},
        {V4L2_CID_CONTRAST, "Contrast / 对比度"},
        {V4L2_CID_SATURATION, "Saturation / 饱和度"},
        {V4L2_CID_HUE, "Hue / 色调"},
        {V4L2_CID_GAMMA, "Gamma / 伽马"},
        {V4L2_CID_GAIN, "Gain / 增益"},
        {V4L2_CID_SHARPNESS, "Sharpness / 锐度"},
        {V4L2_CID_BACKLIGHT_COMPENSATION, "Backlight compensation / 逆光补偿"},
        {V4L2_CID_AUTO_WHITE_BALANCE, "Auto white balance / 自动白平衡"},
        {V4L2_CID_WHITE_BALANCE_TEMPERATURE, "White-balance temperature / 白平衡色温"},
        {V4L2_CID_EXPOSURE_AUTO, "Auto exposure mode / 自动曝光模式"},
        {V4L2_CID_EXPOSURE_ABSOLUTE, "Exposure time / 曝光时间 (unit: 100 us)"},
        {V4L2_CID_EXPOSURE_AUTO_PRIORITY, "Auto exposure priority / 自动曝光优先"},
        {V4L2_CID_FOCUS_AUTO, "Auto focus / 自动对焦"},
        {V4L2_CID_FOCUS_ABSOLUTE, "Focus / 焦距"},
        {V4L2_CID_ZOOM_ABSOLUTE, "Zoom / 变焦"},
        {V4L2_CID_IRIS_ABSOLUTE, "Iris (aperture) / 光圈"},
        {V4L2_CID_IRIS_RELATIVE, "Relative iris / 相对光圈"},
    };

    out << "\nCommon image and lens controls:\n";
    std::set<__u32> common_control_ids;
    for (std::size_t i = 0; i < sizeof(common_controls) / sizeof(common_controls[0]); ++i)
    {
        common_control_ids.insert(common_controls[i].id);
        query_and_print_control(fd, common_controls[i], out);
    }

    out << "\nOther controls reported by the driver:\n";
    bool found_other_control = false;
    v4l2_queryctrl query;
    std::memset(&query, 0, sizeof(query));
    query.id = V4L2_CTRL_FLAG_NEXT_CTRL;
    while (v4l2_ioctl(fd, VIDIOC_QUERYCTRL, &query))
    {
        if (common_control_ids.count(query.id) == 0 &&
            (query.flags & V4L2_CTRL_FLAG_DISABLED) == 0 &&
            query.type != V4L2_CTRL_TYPE_CTRL_CLASS)
        {
            print_control_details(fd, query,
                                  reinterpret_cast<const char *>(query.name), out);
            found_other_control = true;
        }
        query.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
    }
    if (!found_other_control)
    {
        out << "  (none)\n";
    }
    out << "=== End camera property support ===\n";
    close(fd);
}
