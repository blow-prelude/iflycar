#pragma once
#include <opencv2/opencv.hpp>
// #include <Eigen/Dense>
#include <iostream>

struct ImageProcessConfig
{
    // 预处理
    int process_max_h = 240;
    int process_max_w = 320;

    // 边线搜索
    int x_continual = 10;
    int y_continual = 5;
    int search_offset = 30; // 起始搜索偏移量
    int init_stable_count = 8;
    int miss_threshold = 3;
    float up_ratio = 0.50;
    float down_ratio = 0.90;
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
    float stop_roi_y1 = 0.85;
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
    int straight_target_p_index = -10;
    int left_target_p_index = -15;
    // int right_target_p_index = -20;
    int tracking2_target_p_index = -15;

    // TURNING 状态下用单边线生成 mid_line 的横向偏移（像素）
    int turning_mid_offset = 40;
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

struct LineFit
{
    float k;
    float b;
};

class ImageProcess
{
public:
    ImageProcess(const ImageProcessConfig &config);
    // ~ImageProcess();

    cv::Mat preprocess(cv::Mat &img);
    void resize_frame(cv::Mat &img);
    void set_frame(const cv::Mat &frame);
    cv::Mat return_frame();

    void clear_lines();
    float get_angle_k(float k1, float k2);
    float get_angle_p(cv::Point p1, cv::Point p2, cv::Point p3);
    LineFit fit_line_1d(int N, int sx, int sy, int sxx, int syy, int sxy);
    float det3x3(float a00, float a01, float a02,
                 float a10, float a11, float a12,
                 float a20, float a21, float a22);
    std::array<float, 3> polyfit_quadratic(const std::vector<float> &x, const std::vector<float> &y);
    void update_prev_frame_lines(); // 更新上一帧的边线数据
    // void linear_interpolation(Eigen::MatrixX2d &line);
    void linear_interpolation(std::vector<cv::Point> &line, std::vector<cv::Point> &interp_line);
    bool add_point_with_stable_start(std::vector<cv::Point> &line, cv::Point point, std::vector<cv::Point> &stable_buf, bool &stable, int x_thresh, int y_thresh);
    int get_search_start_point(std::vector<cv::Point> &pre_line, int cur_y, int img_w, bool is_left);
    // void fill_boundary(Eigen::MatrixX2d &left_line, Eigen::MatrixX2d &right_line, std::vector<int> img_shape, Eigen::MatrixX2d &supple_left_line, Eigen::MatrixX2d &supple_right_line);
    void fill_boundary(std::vector<cv::Point> &left_line, std::vector<cv::Point> &right_line, std::vector<int> img_shape, std::vector<cv::Point> &supple_left_line, std::vector<cv::Point> &supple_right_line, bool allow_prev_fallack = false);
    void fit_polynomial();
    void fit_polynomial2();
    std::vector<int> get_stop_line(cv::Mat &binary, cv::Mat &canvas, bool is_draw);
    bool judge_enter_turning(std::vector<int> &stop_mid, int img_h, int img_w, float &y_norm);
    bool judge_turing_end(int img_w, int img_h, MissLineState &miss_line);
    void get_side_line_task_1(cv::Mat &img, cv::Mat &canvas, bool is_draw);
    void get_side_line_task_2(cv::Mat &img, cv::Mat &canvas, bool is_draw, bool find_corner, SearchSide side = BOTH);
    void calculate_mid_line(cv::Mat &img);

    void draw_line(cv::Mat &canvas, float fps, std::string state);

    // 公开访问线检测结果（Python 版本直接访问这些成员）
    void set_mid_line_mode(MidLineMode mode) { mid_line_mode_ = mode; }
    MidLineMode get_mid_line_mode() const { return mid_line_mode_; }
    std::vector<cv::Point> &get_fit_mid_line() { return fit_mid_line_; }
    cv::Point &get_left_corners() { return left_corners_; }
    cv::Point &get_right_corners() { return right_corners_; }

private:
    MidLineMode mid_line_mode_ = MID_AVG; // TURNING 时单边线 mid_line 模式
    ImageProcessConfig config_;
    cv::Mat frame_;
    // Eigen::MatrixX2d left_line_;
    // Eigen::MatrixX2d right_line_;
    // Eigen::MatrixX2d supple_left_line_;
    // Eigen::MatrixX2d supple_right_line_;
    // Eigen::MatrixX2d mid_line_;
    // Eigen::MatrixX2d fit_mid_line_;
    std::vector<cv::Point> left_line_;
    std::vector<cv::Point> right_line_;
    std::vector<cv::Point> supple_left_line_;
    std::vector<cv::Point> supple_right_line_;
    std::vector<cv::Point> mid_line_;
    std::vector<cv::Point> fit_mid_line_;
    std::vector<cv::Point> prev_left_line_;
    std::vector<cv::Point> prev_right_line_;
    std::vector<cv::Point> prev_supple_left_line_;
    std::vector<cv::Point> prev_supple_right_line_;
    // 拐点
    cv::Point left_corners_;
    cv::Point right_corners_;
};