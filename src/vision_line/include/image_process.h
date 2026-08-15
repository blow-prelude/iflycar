#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

struct ImageProcessConfig
{
    // 预处理
    int process_max_h = 240;
    int process_max_w = 320;

    // 边线搜索
    int x_continual = 10;
    int y_continual = 5;
    int search_offset = 30; // 起始搜索偏移量
    int init_stable_count = 5;
    int miss_threshold = 3;
    float up_ratio = 0.50;
    float down_ratio = 0.90;
    // get_side_line_task_1 专用纵向扫描区间（与 task_2 的 up_ratio/down_ratio 相互独立）
    float task1_down_ratio = 0.88f;      // 起始扫描行比例（靠下，对应 scan_y_start）
    float task1_up_ratio = 0.65f;        // 结束扫描行比例（靠上，对应 scan_y_end）
    int search_range_wide = 50;          // 动态搜索窗口最大宽度
    int search_range_narrow = 30;        // 动态搜索窗口最小宽度
    float search_range_threshold = 0.60; // 搜索窗口宽度调整阈值
    int min_left_right_distance = 30;    // 左右边线最小间距

    // 拐点检测
    int corner_angle_high = 135;
    int corner_angle_low = 45;
    float corner_y_ratio = 0.70;

    // 停止线检测
    float stop_roi_y0 = 0.65;
    float stop_roi_y1 = 0.97;
    float stop_roi_x0 = 0.30;
    float stop_roi_x1 = 0.70;
    int stop_kernel_w = 15;
    int stop_min_width = 80;

    // 转弯判断
    float turning_enter_y_thresh = 0.65;
    int turning_end_x_diff = 20; // 转弯结束时两边线末端 x 坐标差异阈值
    int turning_end_y_diff = 30;

    // 多项式拟合（分段线性）
    float fit_angle_thresh = 30.0;
    int fit_max_offset = 10; // 近端最大横向偏移量
    float fit_y_div_far_w = 0.45;
    float fit_y_div_near_w = 0.55;

    // 插值 & 填充
    int interp_dx_thresh = 6;
    int interp_dy_thresh = 3;
    float fill_down_ratio = 0.90;

    // 追踪点的索引
    int straight_target_p_index = -25;
    int left_target_p_index = -15;
    // int right_target_p_index = -20;
    int tracking2_target_p_index = -15;

    // TURNING 状态下用单边线生成 mid_line 的横向偏移（像素）
    int turning_mid_offset = 33;
};

// 线性拟合结果
struct LineFit
{
    float k;
    float b;
};

// 判断丢线状态机
enum MissLineState
{
    NO_MISS = 0,
    MISS = 1,
    RECOVERED = 2
};

// mid_line 计算模式
enum MidLineMode
{
    MID_AVG = 0,     // (supple_left + supple_right) / 2
    LEFT_OFFSET = 1, // supple_left + offset
    RIGHT_OFFSET = 2 // supple_right - offset
};

// 边线搜索范围
enum SearchSide
{
    BOTH = 0,
    LEFT_ONLY = 1,
    RIGHT_ONLY = 2,
};

// 绘制目标点所属的跟踪策略，与显示文本解耦。
enum class TrackingTarget
{
    STRAIGHT,
    TURNING,
    TRACKING2,
};

struct StopLineDetection
{
    StopLineDetection() : found(false) {}
    StopLineDetection(cv::Point detected_center, cv::Rect detected_bounds)
        : found(true), center(detected_center), bounds(detected_bounds) {}

    explicit operator bool() const { return found; }

    bool found;
    cv::Point center;
    cv::Rect bounds;
};

class ImageProcess
{
public:
    // 使用 RGA 进行 YUYV 色域转换。
    enum YuyvColorSpace
    {
        YUYV_BT601_LIMIT = 1,
        YUYV_BT601_FULL = 2,
        YUYV_BT709_LIMIT = 3,
    };

    ImageProcess(const ImageProcessConfig &config);
    // ~ImageProcess();

    // 将 cv::Mat 中的 YUYV422 转成 CV_8UC3，通道顺序为 R、G、B。
    // 支持 CV_8UC2(height, width) 和 CV_8UC1(height, width * 2)。
    cv::Mat yuyv_to_rgb(const cv::Mat &yuyv,
                        YuyvColorSpace color_space = YUYV_BT601_LIMIT) const;

    // 将裸 YUYV422 缓冲区转成 CV_8UC3。stride_bytes 为每行字节数，0 表示
    // 使用紧凑布局 width * 2；输入缓冲区在转换完成前必须保持有效。
    cv::Mat yuyv_to_rgb(const void *yuyv_data, int width, int height,
                        std::size_t stride_bytes = 0,
                        YuyvColorSpace color_space = YUYV_BT601_LIMIT) const;

    // 预处理图像；未提供掩膜时，将纯黑区域视为透视变换产生的无效区域。
    cv::Mat1b preprocess(const cv::Mat &img);
    // 使用显式有效区域掩膜进行预处理。掩膜必须是单通道且尺寸与图像一致。
    cv::Mat1b preprocess(const cv::Mat &img, const cv::Mat &valid_mask);
    void resize_frame(cv::Mat &img);
    cv::Mat return_frame() const;

    void clear_lines();
    void fit_polynomial();
    void fit_polynomial2();
    StopLineDetection get_stop_line(const cv::Mat &binary, cv::Mat &canvas, bool is_draw);
    bool judge_enter_turning(const StopLineDetection &stop_line,
                             int img_h, int img_w, float &y_norm);
    bool judge_turing_end(int img_w, int img_h, MissLineState &miss_line);
    void get_side_line_task_1(const cv::Mat &img, cv::Mat &canvas, bool is_draw);
    void get_side_line_task_2(const cv::Mat &img, cv::Mat &canvas, bool is_draw, bool find_corner, SearchSide side = BOTH);
    void calculate_mid_line(cv::Size image_size, float left_weight = 0.5f);

    void draw_line(cv::Mat &canvas, float fps, const std::string &state,
                   TrackingTarget target);

    // 公开访问线检测结果（Python 版本直接访问这些成员）
    void set_mid_line_mode(MidLineMode mode);
    MidLineMode get_mid_line_mode() const { return mid_line_mode_; }
    const std::vector<cv::Point> &get_fit_mid_line() const { return fit_mid_line_; }
    const cv::Point &get_left_corners() const { return left_corners_; }
    const cv::Point &get_right_corners() const { return right_corners_; }

private:
    struct ValidatedScanFrame
    {
        ValidatedScanFrame(const cv::Mat &binary_image, cv::Mat *drawing_canvas,
                           cv::Size max_image_size);

        const cv::Mat &binary;
        cv::Mat *canvas;
    };

    cv::Mat1b preprocess_impl(const cv::Mat &img, const cv::Mat *valid_mask);
    void scan_side_lines(const ValidatedScanFrame &frame,
                         float scan_down_ratio, float scan_up_ratio,
                         bool find_corner, SearchSide side);

    float get_angle_k(float k1, float k2) const;
    float get_angle_p(cv::Point p1, cv::Point p2, cv::Point p3) const;
    std::array<float, 3> polyfit_quadratic(const std::vector<float> &x,
                                           const std::vector<float> &y) const;
    void update_prev_frame_lines();
    void linear_interpolation(const std::vector<cv::Point> &line,
                              std::vector<cv::Point> &interp_line) const;
    bool add_point_with_stable_start(std::vector<cv::Point> &line,
                                     cv::Point point,
                                     std::vector<cv::Point> &stable_buf,
                                     bool &stable) const;
    void fill_boundary(const std::vector<cv::Point> &left_line,
                       const std::vector<cv::Point> &right_line,
                       cv::Size image_size,
                       std::vector<cv::Point> &supple_left_line,
                       std::vector<cv::Point> &supple_right_line,
                       bool allow_prev_fallback = false) const;

    MidLineMode mid_line_mode_ = MID_AVG; // TURNING 时单边线 mid_line 模式
    const ImageProcessConfig config_;     // 构造期校验后不再变化
    cv::Mat frame_;
    std::vector<cv::Point> left_line_;
    std::vector<cv::Point> right_line_;
    std::vector<cv::Point> supple_left_line_;
    std::vector<cv::Point> supple_right_line_;
    std::vector<cv::Point> mid_line_;
    std::vector<cv::Point> fit_mid_line_;
    std::vector<cv::Point> prev_supple_left_line_;
    std::vector<cv::Point> prev_supple_right_line_;
    // 拐点
    cv::Point left_corners_;
    cv::Point right_corners_;
};
