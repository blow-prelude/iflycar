#include "image_process.h"

ImageProcess::ImageProcess(const ImageProcessConfig &config)
    : config_(config)
{
    std::cout << "ImageProcess initialized with config" << std::endl;
}

cv::Mat ImageProcess::preprocess(cv::Mat &img)
{
    if (img.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }
    if (img.rows >= this->config_.process_max_h || img.cols >= this->config_.process_max_w)
    {
        int h = img.rows;
        int w = img.cols;
        float scale = std::min(float(this->config_.process_max_h) / h, float(this->config_.process_max_w) / w);

        cv::resize(img, frame_, cv::Size(int(w * scale), int(h * scale)));
    };

    cv::Mat gray;
    if (frame_.channels() != 3)
    {
        throw std::runtime_error("Input image must have 3 channels.");
    }
    cv::cvtColor(frame_, gray, cv::COLOR_BGR2GRAY);

    // 大尺寸高斯模糊获得背景光照分布
    cv::Mat background;
    auto kernel_size = cv::Size(gray.cols / 5 | 1, gray.rows / 5 | 1); // 根据图像尺寸动态调整核大小，确保足够大
    cv::GaussianBlur(gray, background, kernel_size, 0);
    // 原图减去背景，得到滤除光照后的特征
    cv::Mat foreground;
    cv::subtract(gray, background, foreground);

    // 二值化
    cv::Mat binary;
    cv::threshold(foreground, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 闭运算
    cv::Mat closed;
    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 3);

    return closed;
}

/*
 * 拟合一维直线
 * N: 点的数量
 * sx: x坐标的和
 * sy: y坐标的和
 * sxx: x坐标平方和
 * syy: y坐标平方和
 * sxy: x和y坐标的乘积和
 *
 * 返回值：LineFit结构体，包含斜率k和截距b
 *
 * 逻辑：
 * 1. 计算分母denom = N * syy - sy * sy
 * 2. 如果denom的绝对值大于一个小阈值（1e-9），则计算斜率k和截距b，并返回
 */
LineFit ImageProcess::fit_line_1d(int N, int sx, int sy, int sxx, int syy, int sxy)
{
    float denom = N * syy - sy * sy;
    if (std::abs(denom) > 1e-9)
    {
        float k = (N * sxy - sx * sy) / denom;
        float b = (sx - k * sy) / N;
        return {float(k), float(b)};
    }
    else
    {
        float b = double(sx) / N; // 分母极小，认为是水平线，b退化为均值
        return {0.0f, float(b)};
    }
}

float ImageProcess::det3x3(float a00, float a01, float a02,
                           float a10, float a11, float a12,
                           float a20, float a21, float a22)
{

    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
}

std::array<float, 3> ImageProcess::polyfit_quadratic(const std::vector<float> &x, const std::vector<float> &y)
{
    int N = x.size();
    float sx = 0, sy = 0, sx2 = 0, sxy = 0, sxy2 = 0, sy2 = 0, sy3 = 0, sy4 = 0;
    for (int i = 0; i < N; ++i)
    {
        float yi = y[i];
        float xi = x[i];
        float yi2 = yi * yi;

        sy += yi;
        sy2 += yi2;
        sy3 += yi2 * yi;
        sy4 += yi2 * yi2;
        sx += xi;
        sxy += xi * yi;
        sxy2 += xi * yi2;
    }

    // 构造方程组 M*P=V 的系数
    float m00 = sy4, m01 = sy3, m02 = sy2;
    float m10 = sy3, m11 = sy2, m12 = sy;
    float m20 = sy2, m21 = sy, m22 = static_cast<float>(N);

    float v0 = sxy2, v1 = sxy, v2 = sx;

    // 使用cramer法则求解3x3线性方程组
    float D = det3x3(m00, m01, m02,
                     m10, m11, m12,
                     m20, m21, m22);

    if (std::abs(D) < 1e-9)
    {
        throw std::runtime_error("the matrix is singular or nearly singular");
    }

    float D0 = det3x3(v0, m01, m02,
                      v1, m11, m12,
                      v2, m21, m22);
    float D1 = det3x3(m00, v0, m02,
                      m10, v1, m12,
                      m20, v2, m22);
    float D2 = det3x3(m00, m01, v0,
                      m10, m11, v1,
                      m20, m21, v2);

    return {D0 / D, D1 / D, D2 / D}; // 返回二次项系数、一阶项系数和常数项系数
}

/*
* 计算两条直线的夹角，返回值为角度值
* k1, k2: 两条直线的斜率
* 逻辑：
1. 计算两条直线的夹角的正切值tan_theta = |(k2 - k1) / (1 + k1 * k2)|
2. 使用反正切函数计算夹角theta = atan(tan_theta)
3. 将夹角theta从弧度转换为角度，并返回
*/
float ImageProcess::get_angle_k(float k1, float k2)
{
    double cos_theta = (1 + k1 * k2) / (std::sqrt(1 + k1 * k1) * std::sqrt(1 + k2 * k2));
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta)); // 防止越界
    double theta = std::acos(cos_theta);
    return static_cast<float>(theta * 180.0 / CV_PI); // 将角从弧度转换为角度
}

/*
* 计算3个点之间的夹角，返回值为角度值
* p1, p2, p3: 三个点的坐标，p2为夹角的顶点
* 逻辑：
1. 计算向量v1 = p1 - p2和v2 = p3 - p2
2. 计算向量v1和v2的点积dot和模长norm_v1、norm_v2
3. 计算夹角的余弦值cos_theta = dot / (norm_v1 * norm_v2)，并使用反余弦函数计算夹角theta = acos(cos_theta)
4. 将夹角theta从弧度转换为角度，并返回
*/
float ImageProcess::get_angle_p(cv::Point p1, cv::Point p2, cv::Point p3)
{
    auto v1 = p1 - p2;
    auto v2 = p3 - p2;

    double norm_v1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    double norm_v2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);

    if (norm_v1 < 1e-9 || norm_v2 < 1e-9)
    {
        return 0.0f; // 如果任一向量的模长为零，认为夹角为0度
    }

    double dot = v1.x * v2.x + v1.y * v2.y;
    double cos_theta = dot / (norm_v1 * norm_v2);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta)); // 防止越界
    double theta = std::acos(cos_theta);
    return static_cast<float>(theta * 180.0 / CV_PI); // 将角从弧度转换为角度
}

/*
* 根据上一帧的边线位置获取当前帧的搜索起点
* pre_line: 上一帧的边线点集合，每行一个点，第一列为x坐标，第二列为y坐标
* cur_y: 当前帧的y坐标
* img_w: 图像宽度
* is_left: 是否为左边线
* 返回值：搜索起点的x坐标
* 逻辑：

*/
int ImageProcess::get_search_start_point(std::vector<cv::Point> &pre_line, int cur_y, int img_w, bool is_left)
{
    if (pre_line.empty())
    {
        return img_w / 2;
    }
}

/*
 * 将点加入边线并维持稳定性
 * std::vector<cv::Point> &line: 当前边线点集合
 * cv::Point point: 当前候选点
 * std::vector<cv::Point> &stable_buf: 稳定点缓冲区
 * bool &stable: 边线稳定标志
 * int x_thresh, y_thresh: 稳定点的连续性阈值
 *
 * 返回值：是否将当前点加入边线
 * 逻辑：
 * 1. 如果边线已经稳定（stable为true），则直接比较当前点与边线最后一个点的坐标差异。如果在阈值范围内，则加入边线
 * 2. 如果边线不稳定（stable为false），则将当前点加入稳定点缓冲区，并比较当前点与缓冲区最后一个点的坐标差异。如果在阈值范围内，则继续积累稳定点；如果达到预设的稳定点数量，则将缓冲区的点加入边线，并将stable置为true；如果不在阈值范围内，则清空缓冲区，重新开始积累。
 */
bool ImageProcess::add_point_with_stable_start(std::vector<cv::Point> &line, cv::Point point, std::vector<cv::Point> &stable_buf, bool &stable, int x_thresh, int y_thresh)
{

    if (stable)
    {
        if (std::abs(line.back().x - point.x) < x_thresh && std::abs(line.back().y - point.y) < y_thresh)
        {
            line.push_back(point);
            return true;
        }
    }
    else
    {
        if (stable_buf.empty())
        {
            stable_buf.push_back(point);
        }
        else if (std::abs(stable_buf.back().x - point.x) < x_thresh && std::abs(stable_buf.back().y - point.y) < y_thresh)
        {
            stable_buf.push_back(point);
            if (stable_buf.size() >= this->config_.init_stable_count)
            {
                line.reserve(line.size() + stable_buf.size());
                line.insert(line.end(), stable_buf.begin(), stable_buf.end());
                stable_buf.clear();
                stable = true;
                return true;
            }
        }
        else
        {
            stable_buf.clear();
            stable_buf.push_back(point);
        }
    }
    return false;
}

std::vector<int> ImageProcess::get_stop_line(cv::Mat &binary, cv::Mat &canvas, bool is_draw)
{
    if (binary.empty())
    {
        throw std::runtime_error("Input binary image is empty.");
    }
    int img_h = binary.rows;
    int img_w = binary.cols;

    // 定义停止线ROI
    int roi_y0 = int(img_h * this->config_.stop_roi_y0);
    int roi_y1 = int(img_h * this->config_.stop_roi_y1);
    int roi_x0 = int(img_w * this->config_.stop_roi_x0);
    int roi_x1 = int(img_w * this->config_.stop_roi_x1);
    cv::Rect stop_roi(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0);
    // 提取ROI区域
    cv::Mat roi = binary(stop_roi);

    // 横向形态学削弱斜线
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(this->config_.stop_kernel_w, 1));
    cv::Mat morph;
    cv::morphologyEx(roi, morph, cv::MORPH_OPEN, kernel);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        return {};
    }

    // 找到最像横线的轮廓
    std::vector<int> stop_line; // {x_center, y}
    int line_w = 0, line_h = 0, line_x = 0, line_y = 0;
    double max_score = 0;
    bool found = false;
    for (const auto &contour : contours)
    {
        cv::Rect bbox = cv::boundingRect(contour);
        double aspect_ratio = double(bbox.width) / bbox.height;
        double score = aspect_ratio * bbox.width; // 评估分数，倾向于宽且长的轮廓
        if (score > max_score && bbox.width >= this->config_.stop_min_width)
        {
            max_score = score;
            line_w = bbox.width;
            line_h = bbox.height;
            line_x = bbox.x;
            line_y = bbox.y;
            found = true;
        }
    }
    if (!found)
    {
        return {};
    }
    stop_line = {line_x + line_w / 2 + roi_x0, line_y + line_h / 2 + roi_y0};

    if (is_draw && !stop_line.empty())
    {
        cv::circle(canvas, cv::Point(stop_line[0], stop_line[1]), 5, cv::Scalar(0, 0, 255), -1);
        cv::rectangle(canvas, cv::Rect(line_x + roi_x0, line_y + roi_y0, line_w, line_h), cv::Scalar(255, 0, 0), 2);
    }
    return stop_line;
}

bool ImageProcess::judge_enter_turning(std::vector<int> &stop_mid, int img_h, int img_w, float &y_norm)
{
    if (stop_mid.empty())
    {
        return false;
    }
    y_norm = stop_mid[1] / float(img_h);
    return y_norm > this->config_.turning_enter_y_thresh;
}

/*
 * 判断转弯是否结束
 * img_w, img_h: 图像宽度和高度
 * miss_line: 是否丢过线
 * 逻辑：
1. 如果之前丢过线（miss_line为true），且当前左右边线的末端点x坐标差异大于预设的转弯结束阈值，则认为转弯结束，返回true
2. 否则认为转弯未结束，返回false
3. 有时候可能左右都没有丢线，但是末端点x坐标差异也很小，即两边同时把一条线当作自己的线，这时认为是丢线
 */
bool ImageProcess::judge_turing_end(int img_w, int img_h, MissLineState &miss_line)
{

    if (miss_line == RECOVERED)
    {
        return true; // 已经恢复过线，直接认为转弯结束
    }

    if (this->left_line_.empty() || this->right_line_.empty())
    {
        miss_line = MISS;
        return false;
    }

    int left_x = this->left_line_.back().x;
    int right_x = this->right_line_.back().x;
    int x_diff = std::abs(left_x - right_x);
    if (miss_line == MISS && x_diff > this->config_.turning_end_x_diff)
    {
        miss_line = RECOVERED; // 转弯结束，重置丢线状态
        return true;
    }
    else if (miss_line == NO_MISS && x_diff < this->config_.turning_end_x_diff)
    {
        miss_line = MISS; // 可能同时丢线
    }
    return false;
}

/*
Eigen::MatrixX2d &line: 当前边线点集合，每行一个点，第一列为x坐标，第二列为y坐标
逻辑：
1. 遍历边线点集合，比较相邻点之间的坐标差异
2. 如果相邻点之间的坐标差异超过预设的阈值，则在两点之间进行线性插值，生成新的点，并将其加入边线集合
3. 最后将插值后的点集合转换回Eigen::MatrixX2d格式
*/
// void ImageProcess::linear_interpolation(Eigen::MatrixX2d &line)
// {
//     std::vector<Eigen::Vector2d> interp_points;
//     // 预扩容
//     interp_points.reserve(line.rows() * 2); // 预估插值后点的数量，避免频繁扩容

//     for (int i = 0; i < line.rows() - 1; i++)
//     {
//         Eigen::Vector2d p1 = line.row(i);
//         Eigen::Vector2d p2 = line.row(i + 1);
//         interp_points.push_back(p1);

//         double dx = std::abs(p1.x() - p2.x());
//         double dy = std::abs(p1.y() - p2.y());
//         if (dx > this->config_.interp_dx_thresh || dy > this->config_.interp_dy_thresh)
//         {
//             int interp_count = std::max(dx / this->config_.interp_dx_thresh, dy / this->config_.interp_dy_thresh);
//             for (int j = 1; j < interp_count; j++)
//             {
//                 double alpha = double(j) / interp_count;
//                 Eigen::Vector2d interp_point = (1 - alpha) * p1 + alpha * p2;
//                 interp_points.push_back(interp_point);
//                 // line.conservativeResize(line.rows() + 1, Eigen::NoChange);
//                 // line.row(line.rows() - 1) = interp_point;
//             }
//         }
//     }
//     interp_points.push_back(line.row(line.rows() - 1));

//     line.resize(interp_points.size(), 2);
//     // Convert the vector of interpolated points to an Eigen::MatrixX2d
//     for (size_t i = 0; i < interp_points.size(); ++i)
//     {
//         line.row(i) = interp_points[i];
//     }
// }

/*
 更新边线信息
*/
void ImageProcess::update_prev_frame_lines()
{
    if (!this->left_line_.empty())
    {
        this->prev_left_line_ = this->left_line_;
    }
    if (!this->right_line_.empty())
    {
        this->prev_right_line_ = this->right_line_;
    }

    if (!this->supple_left_line_.empty())
    {
        this->prev_supple_left_line_ = this->supple_left_line_;
    }
    if (!this->supple_right_line_.empty())
    {
        this->prev_supple_right_line_ = this->supple_right_line_;
    }
}
/*
std::vector<cv::Point> &line: 当前边线点集合，每行一个点，第一列为x坐标，第二列为y坐标
逻辑：
1. 遍历边线点集合，比较相邻点之间的坐标差异
2. 如果相邻点之间的坐标差异超过预设的阈值，则在两点之间进行线性插值，生成新的点，并将其加入边线集合
*/
void ImageProcess::linear_interpolation(std::vector<cv::Point> &line, std::vector<cv::Point> &interp_points)
{
    interp_points.clear();
    if (line.empty())
    {
        std::cout << "[linear_interpolation] Input line is empty!" << std::endl;
        return;
    }

    // 预扩容
    interp_points.reserve(line.size() * 2); // 预估插值后点的数量，避免频繁扩容

    // 计算 x 范围
    int min_x = line[0].x, max_x = line[0].x;
    for (const auto &pt : line)
    {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
    }

    // std::cout << "[linear_interpolation] Input line size: " << line.size()
    //           << ", y range: [" << line.front().y << ", " << line.back().y << "]"
    //           << ", x range: [" << min_x << ", " << max_x << "]" << std::endl;

    // 从后向前遍历，使输出的点按 y 从小到大排列
    for (size_t i = line.size() - 1; i > 0; i--)
    {
        cv::Point p1 = line[i];
        cv::Point p2 = line[i - 1];
        interp_points.push_back(p1);

        double dx = std::abs(p1.x - p2.x);
        double dy = std::abs(p1.y - p2.y);
        if (dx > this->config_.interp_dx_thresh || dy > this->config_.interp_dy_thresh)
        {
            int interp_count = std::max(int(dx / this->config_.interp_dx_thresh), int(dy / this->config_.interp_dy_thresh));
            for (int j = 1; j < interp_count; j++)
            {
                double alpha = double(j) / interp_count;
                cv::Point interp_point = (1 - alpha) * p1 + alpha * p2;
                interp_points.push_back(interp_point);
            }
        }
    }
    interp_points.push_back(line[0]);

    // 计算输出的 x 范围
    min_x = interp_points[0].x, max_x = interp_points[0].x;
    for (const auto &pt : interp_points)
    {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
    }

    // std::cout << "[linear_interpolation] Output size: " << interp_points.size()
    //           << ", y range: [" << interp_points.front().y << ", " << interp_points.back().y << "]"
    //           << ", x range: [" << min_x << ", " << max_x << "]" << std::endl;
}

/*
std::vector<cv::Point> &left_line, &right_line: 当前左右边线点集合，每行一个点，第一列为x坐标，第二列为y坐标
std::vector<int> img_shape: 图像的高度和宽度，格式为{height, width}
std::vector<cv::Point> &supple_left_line, &supple_right_line: 用于存储填充后的边线点集合
逻辑：
1. 根据图像高度计算填充的底部y坐标限制，通常为图像高度的某个比例位置
2. 判断左右边线的存在情况：
   a. 如果两边都有线，则对两边线进行插值，并将两边线的底部y坐标取较小值作为填充起点，向下填充到底部y坐标限制
   b. 如果左线丢失但右线存在，则以右线为基准进行插值，并从右线的底部y坐标开始向下填充，同时将左线的填充点x坐标设置为0
   c. 如果右线丢失但左线存在，则以左线为基准进行插值，并从左线的底部y坐标开始向下填充，同时将右线的填充点x坐标设置为图像宽度减1
*/
void ImageProcess::fill_boundary(std::vector<cv::Point> &left_line, std::vector<cv::Point> &right_line, std::vector<int> img_shape, std::vector<cv::Point> &supple_left_line, std::vector<cv::Point> &supple_right_line, bool allow_prev_fallack)
{
    int img_h = img_shape[0];
    int img_w = img_shape[1];
    int bottom_y_limit = int(img_h * this->config_.fill_down_ratio);

    // std::cout << "[fill_boundary] left_line.size()=" << left_line.size()
    //           << ", right_line.size()=" << right_line.size() << std::endl;

    // 两边都有线：插值 + 把较短的边补齐到较长的边
    if (!left_line.empty() && !right_line.empty())
    {
        // std::cout << "[fill_boundary] Both lines exist, interpolating..." << std::endl;
        // 插值
        linear_interpolation(left_line, supple_left_line);
        linear_interpolation(right_line, supple_right_line);

        // 先判断哪边 y 值更大，把另一边填充到对应的 y（插值点按 y 从小到大，push_back 即可）
        int left_bottom_y = int(supple_left_line.back().y);
        int right_bottom_y = int(supple_right_line.back().y);
        if (left_bottom_y > right_bottom_y)
        {
            for (int y = right_bottom_y + 2; y <= left_bottom_y; y += 2)
            {
                supple_right_line.push_back(cv::Point(img_w - 1, y));
            }
        }
        else if (right_bottom_y > left_bottom_y)
        {
            for (int y = left_bottom_y + 2; y <= right_bottom_y; y += 2)
            {
                supple_left_line.push_back(cv::Point(0, y));
            }
        }
    }

    // 左线丢失，右线存在
    else if (left_line.empty() && !right_line.empty())
    {
        // std::cout << "[fill_boundary] Left line missing, using right line..." << std::endl;
        // 以右线为基准插值，丢线边拷贝相同 y 序列、x 置 0
        linear_interpolation(right_line, supple_right_line);

        // 左边线使用右线相同的 y 坐标序列，但 x 全部为 0
        supple_left_line = supple_right_line;
        const size_t n = supple_left_line.size();
        cv::Point *ptr = supple_left_line.data();

        for (size_t i = 0; i < n; i++)
        {
            ptr[i].x = 0;
        }
    }

    // 右线丢失，左线存在
    else if (!left_line.empty() && right_line.empty())
    {
        // std::cout << "[fill_boundary] Right line missing, using left line..." << std::endl;
        // 以左线为基准插值，丢线边拷贝相同 y 序列、x 置 img_w - 1
        linear_interpolation(left_line, supple_left_line);

        // 右边线使用左线相同的 y 坐标序列，但 x 全部为 img_w - 1
        supple_right_line = supple_left_line;
        const size_t n = supple_right_line.size();
        cv::Point *ptr = supple_right_line.data();

        for (size_t i = 0; i < n; i++)
        {
            ptr[i].x = img_w - 1;
        }
    }

    // 左右都丢线
    else
    {
        // std::cout << "[fill_boundary] Both lines missing, using previous fallback..." << std::endl;
        if (allow_prev_fallack && !this->prev_supple_left_line_.empty() && !this->prev_supple_right_line_.empty())
        {
            // std::cout << "[fill_boundary] Using previous frame's supple lines" << std::endl;
            supple_left_line = this->prev_supple_left_line_;
            supple_right_line = this->prev_supple_right_line_;
        }
        // else
        // {
        //     // 如果没有可用的前一帧数据，则生成默认的边界线
        //     for (int y = 0; y < bottom_y_limit; y += 2)
        //     {
        //         supple_left_line.push_back(cv::Point(0, y));
        //         supple_right_line.push_back(cv::Point(img_w - 1, y));
        //     }
        // }
    }
}

void ImageProcess::fit_polynomial()
{
    if (this->mid_line_.size() < 5)
    {
        std::cerr << "Not enough points to fit polynomial." << std::endl;

        return;
    }

    int y_dive = this->mid_line_.back().y * this->config_.fit_y_div_far_w + this->mid_line_.front().y * this->config_.fit_y_div_near_w;

    float near_sx = 0.0f, near_sy = 0.0f, near_sxx = 0.0f, near_syy = 0.0f, near_sxy = 0.0f;
    float far_sx = 0.0f, far_sy = 0.0f, far_sxx = 0.0f, far_syy = 0.0f, far_sxy = 0.0f;
    float near_y_min = std::numeric_limits<float>::max();
    float near_y_max = std::numeric_limits<float>::lowest();
    float far_y_min = std::numeric_limits<float>::max();
    float far_y_max = std::numeric_limits<float>::lowest();
    int near_count = 0, far_count = 0;

    // 根据y坐标将点分为近段和远段两部分，分别计算统计量
    for (const auto &p : this->mid_line_)
    {
        float x = p.x;
        float y = p.y;
        if (y < y_dive)
        {
            near_sx += x;
            near_sy += y;
            near_sxx += x * x;
            near_syy += y * y;
            near_sxy += x * y;
            near_y_min = std::min(near_y_min, y);
            near_y_max = std::max(near_y_max, y);

            near_count++;
        }
        else
        {
            far_sx += x;
            far_sy += y;
            far_sxx += x * x;
            far_syy += y * y;
            far_sxy += x * y;
            far_y_min = std::min(far_y_min, y);
            far_y_max = std::max(far_y_max, y);
            far_count++;
        }
    }

    // 如果有一段点过少，退化为整段拟合
    if (near_count < 2 || far_count < 2)
    {
        float sx2 = far_sx + near_sx;
        float sy2 = far_sy + near_sy;
        float sxx2 = far_sxx + near_sxx;
        float syy2 = far_syy + near_syy;
        float sxy2 = far_sxy + near_sxy;
        if (near_count + far_count < 2)
        {
            std::cerr << "Not enough points to fit polynomial." << std::endl;
            this->fit_mid_line_.clear();
            return;
        }

        LineFit fit = fit_line_1d(near_count + far_count, sx2, sy2, sxx2, syy2, sxy2);
        this->fit_mid_line_.clear();
        // 修正：使用整体y范围，确保循环条件正确
        int y_start = static_cast<int>(std::min(far_y_min, near_y_min));
        int y_end = static_cast<int>(std::max(far_y_max, near_y_max));
        // std::cerr << "[fit_poly][DEGENERATE] mid=" << this->mid_line_.size()
        //           << " near_count=" << near_count << " far_count=" << far_count
        //           << " far_y_min=" << far_y_min << " near_y_max=" << near_y_max
        //           << " y_range=[" << y_start << "," << y_end << "]"
        //           << " k=" << fit.k << " b=" << fit.b << std::endl;
        for (int y = y_start; y <= y_end; y++)
        {
            int x = static_cast<int>(fit.k * y + fit.b);
            this->fit_mid_line_.emplace_back(cv::Point(x, y));
        }
        return;
    }

    // 分段线性拟合
    LineFit near_fit = fit_line_1d(near_count, near_sx, near_sy, near_sxx, near_syy, near_sxy);
    LineFit far_fit = fit_line_1d(far_count, far_sx, far_sy, far_sxx, far_syy, far_sxy);

    // 计算2直线夹角得到偏移量
    float angle = get_angle_k(near_fit.k, far_fit.k);
    int offset = 0;
    if (angle > this->config_.fit_angle_thresh)
    {
        float sign_k_far = (far_fit.k > 0.0) ? 1.0 : ((far_fit.k < 0.0) ? -1.0 : 0.0);
        float factor = std::min(angle / 60.0f, 1.0f);
        offset = static_cast<int>(-sign_k_far * factor * this->config_.fit_max_offset);
    }

    int far_y_start = static_cast<int>(far_y_min);
    int far_y_end = static_cast<int>(far_y_max);
    int near_y_start = static_cast<int>(near_y_min);
    int near_y_end = static_cast<int>(near_y_max);

    // std::cerr << "[fit_poly][PIECEWISE] mid=" << this->mid_line_.size()
    //           << " near_count=" << near_count << " far_count=" << far_count
    //           << " far_y=[" << far_y_min << "," << far_y_max << "]"
    //           << " near_y=[" << near_y_min << "," << near_y_max << "]"
    //           << " near(k=" << near_fit.k << ",b=" << near_fit.b << ")"
    //           << " far(k=" << far_fit.k << ",b=" << far_fit.b << ")"
    //           << " offset=" << offset << std::endl;

    // 预分配内存
    this->fit_mid_line_.reserve(far_y_end - far_y_start + near_y_end - near_y_start + 1);

    // 根据拟合结果生成新的点集合（按y值从小到大有序生成）
    // 注意：在图像坐标系中，y值越小=越靠上（远），y值越大=越靠下（近）
    // near段（y < y_dive）= 远段，应该先生成
    // far段（y >= y_dive）= 近段，应该后生成
    for (int y = near_y_start; y <= near_y_end; y++)
    {
        int x = static_cast<int>(near_fit.k * y + near_fit.b) + offset;
        this->fit_mid_line_.emplace_back(cv::Point(x, y));
    }
    for (int y = far_y_start; y <= far_y_end; y++)
    {
        int x = static_cast<int>(far_fit.k * y + far_fit.b) + offset;
        this->fit_mid_line_.emplace_back(cv::Point(x, y));
    }

    // // 打印所有线的首尾
    // std::cout << "[fit_poly] left_line :first:" << (this->left_line_.empty() ? "empty" : std::to_string(this->left_line_.front().x) + "," + std::to_string(this->left_line_.front().y))
    //           << " last:" << (this->left_line_.empty() ? "empty" : std::to_string(this->left_line_.back().x) + "," + std::to_string(this->left_line_.back().y)) << std::endl;
    // std::cout << "[fit_poly] right_line :first:" << (this->right_line_.empty() ? "empty" : std::to_string(this->right_line_.front().x) + "," + std::to_string(this->right_line_.front().y))
    //           << " last:" << (this->right_line_.empty() ? "empty" : std::to_string(this->right_line_.back().x) + "," + std::to_string(this->right_line_.back().y)) << std::endl;
    // std::cout << "[fit_poly] supple_left_line :first:" << (this->supple_left_line_.empty() ? "empty" : std::to_string(this->supple_left_line_.front().x) + "," + std::to_string(this->supple_left_line_.front().y))
    //           << " last:" << (this->supple_left_line_.empty() ? "empty" : std::to_string(this->supple_left_line_.back().x) + "," + std::to_string(this->supple_left_line_.back().y)) << std::endl;
    // std::cout << "[fit_poly] supple_right_line :first:" << (this->supple_right_line_.empty() ? "empty" : std::to_string(this->supple_right_line_.front().x) + "," + std::to_string(this->supple_right_line_.front().y))
    //           << " last:" << (this->supple_right_line_.empty() ? "empty" : std::to_string(this->supple_right_line_.back().x) + "," + std::to_string(this->supple_right_line_.back().y)) << std::endl;
    // std::cout << "[fit_poly] fit_mid_line :first:" << (this->fit_mid_line_.empty() ? "empty" : std::to_string(this->fit_mid_line_.front().x) + "," + std::to_string(this->fit_mid_line_.front().y))
    //           << " last:" << (this->fit_mid_line_.empty() ? "empty" : std::to_string(this->fit_mid_line_.back().x) + "," + std::to_string(this->fit_mid_line_.back().y)) << std::endl;
    // for (int i = 0; i < std::min((int)this->fit_mid_line_.size(), 10); i++)
    // {
    //     std::cout << "[fit_poly] fit_mid_line_[" << i << "]:" << this->fit_mid_line_[i].x << "," << this->fit_mid_line_[i].y << std::endl;
    // }

    // for (int i = this->fit_mid_line_.size(); i > std::max(0, (int)this->fit_mid_line_.size() - 10); i--)
    // {
    //     std::cout << "[fit_poly] fit_mid_line_[" << i - 1 << "]:" << this->fit_mid_line_[i - 1].x << "," << this->fit_mid_line_[i - 1].y << std::endl;
    // }
}

/*
* 用二次函数拟合中线

*/
void ImageProcess::fit_polynomial2()
{
    size_t N = this->mid_line_.size();
    if (N < 3)
    {
        // std::cout << "Not enough points to fit polynomial." << std::endl;
        return;
    }

    std::vector<float> y_pts, x_pts;
    y_pts.reserve(N);
    x_pts.reserve(N);
    for (const auto &p : this->mid_line_)
    {
        x_pts.push_back(p.x);
        y_pts.push_back(p.y);
    }
    try
    {
        std::array<float, 3> coeffs = polyfit_quadratic(x_pts, y_pts);
        float a = coeffs[0];
        float b = coeffs[1];
        float c = coeffs[2];

        int y_start = static_cast<int>(this->mid_line_.front().y);
        int y_end = static_cast<int>(this->mid_line_.back().y);
        this->fit_mid_line_.reserve(y_end - y_start + 1);
        for (int y = y_start; y <= y_end; y += 3)
        {
            int x = static_cast<int>(a * y * y + b * y + c);
            this->fit_mid_line_.emplace_back(cv::Point(x, y));
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Polynomial fitting failed: " << e.what() << std::endl;
    }
}

void ImageProcess::get_side_line_task_1(cv::Mat &img, cv::Mat &canvas, bool is_draw)
{
    int mid_x = int(img.cols / 2);
    int img_h = img.rows;
    int img_w = img.cols;

    // 逐行地推的搜索起点
    int prev_row_left_x = mid_x;
    int prev_row_right_x = mid_x;

    int search_left_start;
    int search_left_end;
    int search_right_start;
    int search_right_end;

    // miss计数
    int miss_left_count = 0;
    int miss_right_count = 0;

    // 稳定点缓冲区及标志
    std::vector<cv::Point> left_stable_buf;
    std::vector<cv::Point> right_stable_buf;
    bool left_stable_flag = false;
    bool right_stable_flag = false;

    try
    {
        clear_lines();

        int lx = 0, rx = 0, y = 0;
        for (y = int(img_h * this->config_.down_ratio); y >= int(img_h * this->config_.up_ratio); y--)
        {
            // 逐行计算相邻像素差异，避免计算整张图
            cv::Mat row_mask = (img.row(y) == 0);
            cv::Mat row_left = row_mask(cv::Range::all(), cv::Range(0, img_w - 1));
            cv::Mat row_right = row_mask(cv::Range::all(), cv::Range(1, img_w));
            cv::Mat row_diff;
            cv::bitwise_xor(row_left, row_right, row_diff);

            // 动态搜索：二段阶梯
            float y_norm = y / img_h;
            int cur_range = y_norm > this->config_.search_range_threshold ? this->config_.search_range_wide : this->config_.search_range_narrow;

            // 搜索左边线
            if (left_stable_flag)
            {
                // 左侧稳定时，围绕上一帧位置向左右搜索
                search_left_start = std::min(prev_row_left_x + cur_range, img_w - 2);
                search_left_end = std::max(prev_row_left_x - cur_range, 0);
            }
            else
            {
                // 左侧不稳定时，从中线偏左位置向左搜索到图像边缘
                search_left_start = mid_x - this->config_.search_offset;
                search_left_end = 0;
            }

            // // 调试输出：显示搜索区间
            // if (y % 10 == 0) // 每10行输出一次，避免过多输出
            // {
            //     std::cout << "[LEFT] y: " << y << ", range: [" << search_left_end << ", " << search_left_start << "], prev_x: " << prev_row_left_x << std::endl;
            // }

            if (is_draw)
            {
                cv::circle(canvas, cv::Point(search_left_start, y), 1, cv::Scalar(0, 255, 255), -1);
                cv::circle(canvas, cv::Point(search_left_end, y), 1, cv::Scalar(0, 255, 255), -1);
            }

            // 获取当前行的候选点
            const uchar *row_ptr = row_diff.ptr<uchar>(0); // 获取当前行的指针

            lx = -1;                                                   // 重置x，避免使用旧值
            for (int i = search_left_start; i >= search_left_end; i--) // 修复：应该用 >= 而不是 <=
            {
                if (row_ptr[i] != 0)
                { // 非0表示存在黑白跳变点
                    lx = i;
                    break;
                }
            }
            bool left_added = false;
            if (0 <= lx && lx < img_w) // 允许图像边缘的点
            {
                left_added = add_point_with_stable_start(this->left_line_, cv::Point(lx, y), left_stable_buf, left_stable_flag, this->config_.x_continual, this->config_.y_continual);
            }

            // // 调试输出
            // if (y % 10 == 0)
            // {
            //     std::cout << "[LEFT] found x=" << lx << ", left_added=" << left_added << ", stable=" << left_stable_flag << std::endl;
            // }

            // 如果当前行的点没有加入左边线，则下一行的搜索起点不更新
            if (left_added)
                prev_row_left_x = lx;

            // miss记数，只有在左边线稳定时才计算
            if (left_stable_flag && !left_added)
                miss_left_count++;
            else
                miss_left_count = 0;
            if (miss_left_count > this->config_.miss_threshold)
            {
                miss_left_count = 0;
                prev_row_left_x = mid_x - this->config_.search_offset;
                left_stable_flag = false; // 重置稳定标志，避免使用错误的搜索范围
                left_stable_buf.clear();  // 清空稳定缓冲区
            }

            // 搜索右边线
            if (right_stable_flag)
            {
                // 右侧稳定时，围绕上一帧位置向左右搜索
                search_right_start = std::max(prev_row_right_x - cur_range, 0);
                search_right_end = std::min(prev_row_right_x + cur_range, img_w - 1);
            }
            else
            {
                // 右侧不稳定时，从中线偏右位置向右搜索到图像边缘
                search_right_start = mid_x + this->config_.search_offset;
                search_right_end = img_w - 1;
            }
            // // 调试输出：显示搜索区间
            // if (y % 10 == 0) // 每10行输出一次，避免过多输出
            // {
            //     std::cout << "[RIGHT] y: " << y << ", range: [" << search_right_start << ", " << search_right_end << "], prev_x: " << prev_row_right_x << std::endl;
            // }

            if (is_draw)
            {
                cv::circle(canvas, cv::Point(search_right_start, y), 1, cv::Scalar(255, 255, 0), -1);
                cv::circle(canvas, cv::Point(search_right_end, y), 1, cv::Scalar(255, 255, 0), -1);
            }

            rx = -1; // 重置x，避免使用旧值
            for (int i = search_right_start; i <= search_right_end; i++)
            {
                if (row_ptr[i] != 0)
                { // 非0表示存在黑白跳变点
                    rx = i;
                    break; // 找到第一个跳变点就停止
                }
            }
            bool right_added = false;
            if (0 <= rx && rx < img_w) // 允许图像边缘的点
            {
                right_added = add_point_with_stable_start(this->right_line_, cv::Point(rx, y), right_stable_buf, right_stable_flag, this->config_.x_continual, this->config_.y_continual);
            }

            // // 调试输出
            // if (y % 10 == 0)
            // {
            //     std::cout << "[RIGHT] found x=" << rx << ", right_added=" << right_added << ", stable=" << right_stable_flag << std::endl;
            // }

            if (right_added)
                prev_row_right_x = rx;
            if (right_stable_flag && !right_added)
                miss_right_count++;
            else
                miss_right_count = 0;
            if (miss_right_count > this->config_.miss_threshold)
            {
                miss_right_count = 0;
                prev_row_right_x = mid_x + this->config_.search_offset;
                right_stable_flag = false; // 重置稳定标志，避免使用错误的搜索范围
                right_stable_buf.clear();  // 清空稳定缓冲区
            }

            // 检测左右边线距离，如果太小则认为是噪点，移除这一对
            if (left_added && right_added)
            {
                int distance = std::abs(lx - rx);
                if (distance < this->config_.min_left_right_distance)
                {
                    // 移除最后加入的点
                    this->left_line_.pop_back();
                    this->right_line_.pop_back();
                    left_stable_flag = false;
                    right_stable_flag = false;
                    prev_row_left_x = mid_x - this->config_.search_offset;
                    prev_row_right_x = mid_x + this->config_.search_offset;
                }
            }
        }

        // 插值+填充边线（同时处理边线情况）
        fill_boundary(this->left_line_, this->right_line_, {img_h, img_w}, this->supple_left_line_, this->supple_right_line_);

        //  使用优化后的边线计算中线
        int n = std::min(this->supple_left_line_.size(), this->supple_right_line_.size());
        this->mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            this->mid_line_[i].x = (this->supple_left_line_[i].x + this->supple_right_line_[i].x) / 2.0;
            this->mid_line_[i].y = (this->supple_left_line_[i].y + this->supple_right_line_[i].y) / 2.0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in get_side_line_task_1: " << e.what() << std::endl;
    }
}

// void ImageProcess::get_side_line_task_1(cv::Mat &img, cv::Mat &canvas, bool is_draw)
// {
//     int mid_x = int(img.cols / 2);
//     int img_h = img.rows;
//     int img_w = img.cols;

//     // 逐行地推的搜索起点
//     int prev_row_left_x = mid_x;
//     int prev_row_right_x = mid_x;

//     int search_left_start;
//     int search_left_end;
//     int search_right_start;
//     int search_right_end;

//     std::vector<int> candidate_xs;

//     // miss计数
//     int miss_left_count = 0;
//     int miss_right_count = 0;

//     // 稳定点缓冲区及标志
//     std::vector<cv::Point> left_stable_buf;
//     std::vector<cv::Point> right_stable_buf;
//     bool left_stable_flag;
//     bool right_stable_flag;

//     try
//     {
//         clear_lines();

//         auto local_left_line = std::vector<cv::Point>();
//         auto local_right_line = std::vector<cv::Point>();

//         // 获取布尔掩码
//         cv::Mat mask = (img == 0);

//         // 无拷贝切片
//         cv::Mat left = mask(cv::Range::all(), cv::Range(0, img_w - 1));
//         cv::Mat right = mask(cv::Range::all(), cv::Range(1, img_w));

//         // 按位异或，得到0与非0的差异
//         cv::Mat diff;
//         cv::bitwise_xor(left, right, diff);

//         int x, y;
//         for (y = img_h * this->config_.down_ratio; y < img_h * this->config_.up_ratio; y--)
//         {
//             cv::Mat row_diff = diff.row(y); // 浅拷贝，指向当前行的相邻像素差异数据

//             // 动态搜索：二段阶梯
//             float y_norm = y / img_h;
//             int cur_range = y_norm > this->config_.search_range_threshold ? this->config_.search_range_wide : this->config_.search_range_narrow;

//             // 搜索左边线
//             if (left_stable_flag)
//             {
//                 search_left_start = std::min(prev_row_left_x + cur_range, img_w - 1);
//                 search_left_end = std::max(prev_row_left_x - cur_range, 0);
//             }
//             else
//             {
//                 search_left_start = mid_x - this->config_.search_offset;
//                 search_left_end = 0;
//             }

//             if (is_draw)
//             {
//                 cv::circle(canvas, cv::Point(search_left_start, y), 1, cv::Scalar(0, 255, 255), -1);
//                 cv::circle(canvas, cv::Point(search_left_end, y), 1, cv::Scalar(0, 255, 255), -1);
//             }

//             // 获取当前行的候选点
//             const uchar *row_ptr = row_diff.ptr<uchar>(0); // 获取当前行的指针
//             for (int i = search_left_start; i <= search_left_end; i--)
//             {
//                 if (row_ptr[i] != 0)
//                 { // 非0表示存在黑白跳变点
//                     x = i;
//                 }
//             }
//             bool left_added = false;
//             if (0 < x && x < img_w - 1)
//             {
//                 left_added = add_point_with_stable_start(local_left_line, cv::Point(x, y), left_stable_buf, left_stable_flag, this->config_.x_continual, this->config_.y_continual);
//             }

//             // 如果当前行的点没有加入左边线，则下一行的搜索起点不更新
//             if (left_added)
//                 prev_row_left_x = x;

//             // miss记数，只有在左边线稳定时才计算
//             if (left_stable_flag && !left_added)
//                 miss_left_count++;
//             else
//                 miss_left_count = 0;
//             if (miss_left_count > this->config_.miss_threshold)
//             {
//                 miss_left_count = 0;
//                 prev_row_left_x = mid_x - this->config_.search_offset;
//             }

//             // 搜索右边线
//             if (right_stable_flag)
//             {
//                 search_right_start = std::max(std::min(prev_row_right_x - cur_range, img_w - 1), 0);
//                 search_right_end = std::min(prev_row_right_x + cur_range, img_w - 1);
//             }
//             else
//             {
//                 search_right_start = mid_x + this->config_.search_offset;
//                 search_right_end = img_w - 1;
//             }
//             if (is_draw)
//             {
//                 cv::circle(canvas, cv::Point(search_right_start, y), 1, cv::Scalar(255, 255, 0), -1);
//                 cv::circle(canvas, cv::Point(search_right_end, y), 1, cv::Scalar(255, 255, 0), -1);
//             }

//             for (int i = search_right_start; i <= search_right_end; i++)
//             {
//                 if (row_ptr[i] != 0)
//                 { // 非0表示存在黑白跳变点
//                     x = i;
//                 }
//             }
//             bool right_added = false;
//             if (0 < x && x < img_w - 1)
//             {
//                 right_added = add_point_with_stable_start(local_right_line, cv::Point(x, y), right_stable_buf, right_stable_flag, this->config_.x_continual, this->config_.y_continual);
//             }
//             if (right_stable_flag && !right_added)
//                 miss_right_count++;
//             else
//                 miss_right_count = 0;
//             if (miss_right_count > this->config_.miss_threshold)
//             {
//                 miss_right_count = 0;
//                 prev_row_right_x = mid_x + this->config_.search_offset;
//             }
//         }

//         // 转成Eigen格式
//         this->left_line_ = Eigen::MatrixX2d(local_left_line.size(), 2);
//         for (size_t i = 0; i < local_left_line.size(); i++)
//         {
//             this->left_line_(i, 0) = local_left_line[i].x;
//             this->left_line_(i, 1) = local_left_line[i].y;
//         }
//         this->right_line_ = Eigen::MatrixX2d(local_right_line.size(), 2);
//         for (size_t i = 0; i < local_right_line.size(); i++)
//         {
//             this->right_line_(i, 0) = local_right_line[i].x;
//             this->right_line_(i, 1) = local_right_line[i].y;
//         }

//         // 插值+填充边线（同时处理边线情况）
//         fill_boundary(this->left_line_, this->right_line_, {img_h, img_w}, this->supple_left_line_, this->supple_right_line_);

//         //  使用优化后的边线计算中线
//         int n = std::min(this->supple_left_line_.rows(), this->supple_right_line_.rows());
//         this->mid_line_ = Eigen::MatrixX2d(n, 2);
//         for (int i = 0; i < n; i++)
//         {
//             this->mid_line_(i, 0) = (this->supple_left_line_(i, 0) + this->supple_right_line_(i, 0)) / 2.0;
//             this->mid_line_(i, 1) = (this->supple_left_line_(i, 1) + this->supple_right_line_(i, 1)) / 2.0;
//         }
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "Error in get_side_line_task_1: " << e.what() << std::endl;
//     }
// }

void ImageProcess::get_side_line_task_2(cv::Mat &img, cv::Mat &canvas, bool is_draw, bool find_corner)
{
    int mid_x = int(img.cols / 2);
    int img_h = img.rows;
    int img_w = img.cols;

    // 逐行地推的搜索起点
    int prev_row_left_x = mid_x;
    int prev_row_right_x = mid_x;

    int search_left_start;
    int search_left_end;
    int search_right_start;
    int search_right_end;

    // miss计数
    int miss_left_count = 0;
    int miss_right_count = 0;

    // 稳定点缓冲区及标志
    std::vector<cv::Point> left_stable_buf;
    std::vector<cv::Point> right_stable_buf;
    bool left_stable_flag = false;
    bool right_stable_flag = false;

    // 拐点检测标志
    bool find_left_corner = false;
    bool find_right_corner = false;
    cv::Point left_nxt_p, left_cur_p, left_pre_p;
    cv::Point right_nxt_p, right_cur_p, right_pre_p;

    try
    {
        clear_lines();

        int lx = 0, rx = 0, y = 0;
        for (y = int(img_h * this->config_.down_ratio); y >= int(img_h * this->config_.up_ratio); y--)
        {
            // 逐行计算相邻像素差异，避免计算整张图
            cv::Mat row_mask = (img.row(y) == 0);
            cv::Mat row_left = row_mask(cv::Range::all(), cv::Range(0, img_w - 1));
            cv::Mat row_right = row_mask(cv::Range::all(), cv::Range(1, img_w));
            cv::Mat row_diff;
            cv::bitwise_xor(row_left, row_right, row_diff);

            // 动态搜索：二段阶梯
            float y_norm = y / img_h;
            int cur_range = y_norm > this->config_.search_range_threshold ? this->config_.search_range_wide : this->config_.search_range_narrow;

            // 搜索左边线
            if (left_stable_flag)
            {
                // 左侧稳定时，围绕上一帧位置向左右搜索
                search_left_start = std::min(prev_row_left_x + cur_range, img_w - 2);
                search_left_end = std::max(prev_row_left_x - cur_range, 0);
            }
            else
            {
                // 左侧不稳定时，从中线偏左位置向左搜索到图像边缘
                search_left_start = mid_x - this->config_.search_offset;
                search_left_end = 0;
            }

            // // 调试输出：显示搜索区间
            // if (y % 10 == 0) // 每10行输出一次，避免过多输出
            // {
            //     std::cout << "[LEFT] y: " << y << ", range: [" << search_left_end << ", " << search_left_start << "], prev_x: " << prev_row_left_x << std::endl;
            // }

            if (is_draw)
            {
                cv::circle(canvas, cv::Point(search_left_start, y), 1, cv::Scalar(0, 255, 255), -1);
                cv::circle(canvas, cv::Point(search_left_end, y), 1, cv::Scalar(0, 255, 255), -1);
            }

            // 获取当前行的候选点
            const uchar *row_ptr = row_diff.ptr<uchar>(0);             // 获取当前行的指针
            lx = -1;                                                   // 重置x，避免使用旧值
            for (int i = search_left_start; i >= search_left_end; i--) // 修复：应该用 >= 而不是 <=
            {
                if (row_ptr[i] != 0)
                { // 非0表示存在黑白跳变点
                    lx = i;
                    break;
                }
            }

            // 先进行稳定点判断
            bool left_added = false;
            if (0 <= lx && lx < img_w) // 允许图像边缘的点
            {
                left_added = add_point_with_stable_start(this->left_line_, cv::Point(lx, y), left_stable_buf, left_stable_flag, this->config_.x_continual, this->config_.y_continual);
            }

            // // 调试输出
            // if (y % 10 == 0)
            // {
            //     std::cout << "[LEFT-TASK2] found x=" << lx << ", left_added=" << left_added << ", stable=" << left_stable_flag << std::endl;
            // }

            // 如果当前行的点没有加入左边线，则下一行的搜索起点不更新
            if (left_added)
            {
                prev_row_left_x = lx;
                // 只有稳定点才参与拐点检测
                if (find_corner && !find_left_corner)
                {
                    left_nxt_p = cv::Point(lx, y);
                    if (left_cur_p.x != 0 && left_cur_p.y != 0 && left_pre_p.x != 0 && left_pre_p.y != 0)
                    {
                        float angle = get_angle_p(left_pre_p, left_cur_p, left_nxt_p);
                        if (angle < this->config_.corner_angle_high && angle > this->config_.corner_angle_low)
                        {
                            find_left_corner = true;
                            this->left_corners_ = left_cur_p;
                            // std::cout << "[DEBUG] Find left corner at: " << left_cur_p << ", angle: " << angle << std::endl;
                        }
                    }
                    // 更新拐点检测的点
                    left_pre_p = left_cur_p;
                    left_cur_p = left_nxt_p;
                }
            }

            // miss计数逻辑应该在left_added条件之外
            if (left_stable_flag && !left_added)
                miss_left_count++;
            else
                miss_left_count = 0;
            if (miss_left_count > this->config_.miss_threshold)
            {
                miss_left_count = 0;
                prev_row_left_x = mid_x - this->config_.search_offset;
                left_stable_flag = false; // 重置稳定标志，避免使用错误的搜索范围
                left_stable_buf.clear();  // 清空稳定缓冲区
            }

            // 搜索右边线
            if (right_stable_flag)
            {
                // 右侧稳定时，围绕上一帧位置向左右搜索
                search_right_start = std::max(prev_row_right_x - cur_range, 0);
                search_right_end = std::min(prev_row_right_x + cur_range, img_w - 2);
            }
            else
            {
                // 右侧不稳定时，从中线偏右位置向右搜索到图像边缘
                search_right_start = mid_x + this->config_.search_offset;
                search_right_end = img_w - 2;
            }

            // 调试输出：显示搜索区间
            // if (y % 10 == 0)
            // {
            //     std::cout << "[RIGHT-TASK2] y: " << y << ", range: [" << search_right_start << ", " << search_right_end << "], prev_x: " << prev_row_right_x << std::endl;
            // }

            if (is_draw)
            {
                cv::circle(canvas, cv::Point(search_right_start, y), 1, cv::Scalar(255, 255, 0), -1);
                cv::circle(canvas, cv::Point(search_right_end, y), 1, cv::Scalar(255, 255, 0), -1);
            }
            // if (y % 4 == 0)
            // {
            //     std::cout << "[DEBUG] y: " << y << ", search_left_start: " << search_left_start << ", search_right_start: " << search_right_start << "search_left_end: " << search_left_end << ", search_right_end: " << search_right_end << std::endl;
            // }

            // 打印右边界跳变情况
            // std::cout << "[DEBUG] right boundary jump:" << row_ptr[img_w - 3] << std::endl;

            rx = -1; // 重置x，避免使用旧值
            for (int i = search_right_start; i <= search_right_end; i++)
            {
                if (row_ptr[i] != 0)
                { // 非0表示存在黑白跳变点
                    rx = i;
                    break; // 右侧搜索：找到第一个跳变点就停止
                }
            }
            bool right_added = false;
            if (0 <= rx && rx < img_w) // 允许图像边缘的点
            {
                right_added = add_point_with_stable_start(this->right_line_, cv::Point(rx, y), right_stable_buf, right_stable_flag, this->config_.x_continual, this->config_.y_continual);
            }

            // // 调试输出
            // if (y % 10 == 0)
            // {
            //     std::cout << "[RIGHT-TASK2] found x=" << rx << ", right_added=" << right_added << ", stable=" << right_stable_flag << std::endl;
            // }

            if (right_added)
            {
                prev_row_right_x = rx;
                // 只有稳定点才参与拐点检测
                if (find_corner && !find_right_corner)
                {
                    right_nxt_p = cv::Point(rx, y);
                    if (right_cur_p.x != 0 && right_cur_p.y != 0 && right_pre_p.x != 0 && right_pre_p.y != 0)
                    {
                        float angle = get_angle_p(right_pre_p, right_cur_p, right_nxt_p);
                        if (angle < this->config_.corner_angle_high && angle > this->config_.corner_angle_low)
                        {
                            find_right_corner = true;
                            this->right_corners_ = right_cur_p;
                            // std::cout << "[DEBUG] Find right corner at: " << right_cur_p << ", angle: " << angle << std::endl;
                        }
                    }
                    // 更新拐点检测的点
                    right_pre_p = right_cur_p;
                    right_cur_p = right_nxt_p;
                }
            }

            // miss计数逻辑应该在right_added条件之外
            if (right_stable_flag && !right_added)
                miss_right_count++;
            else
                miss_right_count = 0;
            if (miss_right_count > this->config_.miss_threshold)
            {
                miss_right_count = 0;
                prev_row_right_x = mid_x + this->config_.search_offset;
                right_stable_flag = false; // 重置稳定标志，避免使用错误的搜索范围
                right_stable_buf.clear();  // 清空稳定缓冲区
            }

            // 检测左右边线距离，如果太小则认为是噪点，移除这一对
            if (left_added && right_added)
            {
                int distance = std::abs(lx - rx);
                if (distance < this->config_.min_left_right_distance)
                {
                    // 移除最后加入的点
                    this->left_line_.pop_back();
                    this->right_line_.pop_back();
                    left_stable_flag = false;
                    right_stable_flag = false;
                    prev_row_left_x = mid_x - this->config_.search_offset;
                    prev_row_right_x = mid_x + this->config_.search_offset;
                    left_stable_buf.clear();
                    right_stable_buf.clear();
                }
            }
        }
    }

    catch (const std::exception &e)
    {
        std::cerr << "Error in get_side_line_task_2: " << e.what() << std::endl;
    }
}

/*
 * 根据当前左右原始边线插值+填充，并计算中线
 * img: 当前二值化图像（仅用于获取宽高）
 * 逻辑：
 * 1. 调用 fill_boundary 对左右边线做插值和底部填充，得到 supple_left_line_/supple_right_line_
 * 2. 按相同索引对左右填充线的点求中点，写入 mid_line_
 * 3. 更新 prev_* 成员，供下一帧 fill_boundary 的 allow_prev_fallack 使用
 */
void ImageProcess::calculate_mid_line(cv::Mat &img)
{
    int img_h = img.rows;
    int img_w = img.cols;

    // 插值+填充边线（同时处理边线情况）
    fill_boundary(this->left_line_, this->right_line_, {img_h, img_w}, this->supple_left_line_, this->supple_right_line_, true);

    if (this->mid_line_mode_ == LEFT_OFFSET)
    {
        // 用左边线 + 偏移
        int n = this->supple_left_line_.size();
        this->mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            int x = this->supple_left_line_[i].x + this->config_.turning_mid_offset;
            this->mid_line_[i].x = std::max(0, std::min(x, img_w - 1));
            this->mid_line_[i].y = this->supple_left_line_[i].y;
        }
    }
    else if (this->mid_line_mode_ == RIGHT_OFFSET)
    {
        // 用右边线 - 偏移
        int n = this->supple_right_line_.size();
        this->mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            int x = this->supple_right_line_[i].x - this->config_.turning_mid_offset;
            this->mid_line_[i].x = std::max(0, std::min(x, img_w - 1));
            this->mid_line_[i].y = this->supple_right_line_[i].y;
        }
    }
    else // MID_AVG
    {
        // 使用优化后的边线计算中线
        int n = std::min(this->supple_left_line_.size(), this->supple_right_line_.size());
        this->mid_line_.resize(n);
        for (int i = 0; i < n; i++)
        {
            this->mid_line_[i].x = (this->supple_left_line_[i].x + this->supple_right_line_[i].x) / 2.0;
            this->mid_line_[i].y = (this->supple_left_line_[i].y + this->supple_right_line_[i].y) / 2.0;
        }
    }

    this->update_prev_frame_lines();
}

void ImageProcess::draw_line(cv::Mat &canvas, float fps, std::string state)
{
    if (canvas.empty())
    {
        throw std::runtime_error("Canvas is empty.");
    }

    if (canvas.channels() == 1)
    {
        cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
    }

    if (fps >= 0.0f)
    {
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(canvas, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    if (!state.empty())
    {
        cv::putText(canvas, state, cv::Point(canvas.cols - 100, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    // 绘制竖直中线
    cv::line(
        canvas,
        cv::Point(canvas.cols / 2 - 1, 0),
        cv::Point(canvas.cols / 2 - 1, canvas.rows - 1),
        cv::Scalar(255, 0, 127),
        1);

    // std::cout << "[DEBUG]left_line size: " << this->left_line_.size() << ", right_line size: " << this->right_line_.size() << ", mid_line size: " << this->mid_line_.size() << ", fit_mid_line size: " << this->fit_mid_line_.size() << std::endl;
    for (const auto &p : this->supple_left_line_)
    {
        cv::circle(canvas, p, 2, cv::Scalar(0, 0, 255), -1);
    }
    for (const auto &p : this->supple_right_line_)
    {
        cv::circle(canvas, p, 2, cv::Scalar(0, 0, 255), -1);
    }
    for (const auto &p : this->mid_line_)
    {
        cv::circle(canvas, p, 2, cv::Scalar(255, 0, 0), -1);
    }
    for (const auto &p : this->fit_mid_line_)
    {
        cv::circle(canvas, p, 2, cv::Scalar(255, 255, 255), -1);
    }

    // 绘制追踪目标点
    int target_idx = this->config_.left_target_p_index;
    if (state == "TRACKING2")
    {
        target_idx = this->config_.tracking2_target_p_index;
    }
    else if (state == "STRAIGHT_TRACKING" || state == "CROSS" || state == "TURNING")
    {
        target_idx = this->config_.straight_target_p_index;
    }

    // std::cout << "Drawing target point at index: " << target_idx << std::endl;

    // 处理负索引
    if (target_idx < 0)
    {
        target_idx = this->fit_mid_line_.size() + target_idx; // 转换为正索引
    }

    if (this->fit_mid_line_.size() > target_idx)
    {
        cv::circle(canvas, this->fit_mid_line_[target_idx], 4, cv::Scalar(255, 0, 255), -1);
    }
}

void ImageProcess::clear_lines()
{
    this->left_line_.clear();
    this->right_line_.clear();
    this->supple_left_line_.clear();
    this->supple_right_line_.clear();
    this->mid_line_.clear();
    this->fit_mid_line_.clear();
    this->left_corners_ = cv::Point(0, 0);
    this->right_corners_ = cv::Point(0, 0);
}

cv::Mat ImageProcess::return_frame()
{
    if (this->frame_.empty())
    {
        throw std::runtime_error("Frame is empty.");
    }
    return this->frame_;
}

void ImageProcess::set_frame(const cv::Mat &frame)
{
    if (frame.empty())
    {
        throw std::runtime_error("Input frame is empty.");
    }
    this->frame_ = frame.clone();
}