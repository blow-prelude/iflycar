#include "image_process.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "im2d.hpp"

namespace
{

    /** 将应用层的 YUYV 色彩空间映射为 RGA 色彩转换模式。 */
    int get_rga_color_space(ImageProcess::YuyvColorSpace color_space)
    {
        switch (color_space)
        {
        case ImageProcess::YUYV_BT601_LIMIT:
            return IM_YUV_TO_RGB_BT601_LIMIT;
        case ImageProcess::YUYV_BT601_FULL:
            return IM_YUV_TO_RGB_BT601_FULL;
        case ImageProcess::YUYV_BT709_LIMIT:
            return IM_YUV_TO_RGB_BT709_LIMIT;
        default:
            throw std::invalid_argument("Unsupported YUYV color space.");
        }
    }

    /** 已验证的 YUYV 缓冲区视图，下游 RGA 调用不再重复检查尺寸。 */
    struct ValidatedYuyvView
    {
        const void *data;
        int width;
        int height;
        int stride_pixels;
        int source_size;
        int destination_size;
    };

    /** 执行带溢出检查的 size_t 乘法。 */
    bool checked_multiply(std::size_t lhs, std::size_t rhs, std::size_t &result)
    {
        if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        {
            return false;
        }
        result = lhs * rhs;
        return true;
    }

    int checked_rga_buffer_size(std::size_t size, const char *name)
    {
        if (size == 0 || size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument(std::string(name) + " buffer is too large for RGA.");
        return static_cast<int>(size);
    }

    ValidatedYuyvView validate_yuyv_view(const void *data, int width, int height,
                                         std::size_t stride_bytes)
    {
        if (data == nullptr)
            throw std::invalid_argument("YUYV data is null.");
        if (width <= 0 || height <= 0 || (width & 1) != 0)
            throw std::invalid_argument("YUYV width must be a positive even number and height must be positive.");

        std::size_t packed_row_bytes = 0;
        if (!checked_multiply(static_cast<std::size_t>(width), 2, packed_row_bytes))
            throw std::invalid_argument("YUYV row size overflows.");
        if (stride_bytes == 0)
            stride_bytes = packed_row_bytes;
        if (stride_bytes < packed_row_bytes || (stride_bytes % 2) != 0)
            throw std::invalid_argument("YUYV stride must be at least width * 2 and aligned to two bytes.");
        if (stride_bytes / 2 > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("YUYV stride is too large for RGA.");

        std::size_t preceding_rows = 0;
        if (!checked_multiply(stride_bytes, static_cast<std::size_t>(height - 1), preceding_rows) ||
            preceding_rows > std::numeric_limits<std::size_t>::max() - packed_row_bytes)
        {
            throw std::invalid_argument("YUYV buffer size overflows.");
        }
        const std::size_t source_size = preceding_rows + packed_row_bytes;

        std::size_t pixel_count = 0;
        std::size_t destination_size = 0;
        if (!checked_multiply(static_cast<std::size_t>(width), static_cast<std::size_t>(height), pixel_count) ||
            !checked_multiply(pixel_count, 3, destination_size))
        {
            throw std::invalid_argument("RGB buffer size overflows.");
        }

        return {data,
                width,
                height,
                static_cast<int>(stride_bytes / 2),
                checked_rga_buffer_size(source_size, "YUYV source"),
                checked_rga_buffer_size(destination_size, "RGB destination")};
    }

    struct RgaHandleGuard
    {
        explicit RgaHandleGuard(rga_buffer_handle_t value) : handle(value) {}
        RgaHandleGuard(const RgaHandleGuard &) = delete;
        RgaHandleGuard &operator=(const RgaHandleGuard &) = delete;

        ~RgaHandleGuard()
        {
            if (handle != 0)
                releasebuffer_handle(handle);
        }

        rga_buffer_handle_t handle;
    };

    cv::Mat convert_yuyv_view(const ValidatedYuyvView &input,
                              ImageProcess::YuyvColorSpace color_space)
    {
        const int src_format = RK_FORMAT_YUYV_422;
        const int dst_format = RK_FORMAT_RGB_888;
        const int color_mode = get_rga_color_space(color_space);

        cv::Mat rgb(input.height, input.width, CV_8UC3);
        assert(rgb.isContinuous() && rgb.data != nullptr);

        RgaHandleGuard src_handle{importbuffer_virtualaddr(
            const_cast<void *>(input.data), input.source_size)};
        RgaHandleGuard dst_handle{importbuffer_virtualaddr(
            rgb.data, input.destination_size)};

        if (src_handle.handle == 0 || dst_handle.handle == 0)
            throw std::runtime_error("RGA importbuffer_virtualaddr() failed.");

        rga_buffer_t src_img = wrapbuffer_handle(src_handle.handle, input.width, input.height,
                                                 src_format, input.stride_pixels, input.height);
        rga_buffer_t dst_img = wrapbuffer_handle(dst_handle.handle, input.width, input.height,
                                                 dst_format, input.width, input.height);

        int ret = imcheck(src_img, dst_img, {}, {}, 0);
        if (ret != IM_STATUS_NOERROR)
            throw std::runtime_error(std::string("RGA imcheck() failed: ") + imStrError((IM_STATUS)ret));

        ret = imcvtcolor(src_img, dst_img, src_format, dst_format, color_mode);
        if (ret != IM_STATUS_SUCCESS)
            throw std::runtime_error(std::string("RGA imcvtcolor() failed: ") + imStrError((IM_STATUS)ret));

        return rgb;
    }

    /** 将整数限制在指定的闭区间内。 */
    int clamp_int(int value, int lower, int upper)
    {
        assert(lower <= upper);
        return std::max(lower, std::min(value, upper));
    }

    /** 将浮点数四舍五入为 int，超出范围时饱和到 int 边界。 */
    int rounded_int_saturated(double value)
    {
        if (!std::isfinite(value) || value <= static_cast<double>(std::numeric_limits<int>::min()))
        {
            return std::numeric_limits<int>::min();
        }
        if (value >= static_cast<double>(std::numeric_limits<int>::max()))
        {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(std::lround(value));
    }

    /** 根据比例计算坐标，避免浮点转整数时的边界未定义行为。 */
    int scaled_coordinate(int size, float ratio)
    {
        assert(size > 0);
        assert(std::isfinite(ratio) && ratio >= 0.0f && ratio <= 1.0f);
        const double value = static_cast<double>(size) * ratio;
        if (value <= 0.0)
        {
            return 0;
        }
        if (value >= static_cast<double>(size))
        {
            return size;
        }
        return static_cast<int>(value);
    }

    /** 将搜索区间裁剪到有效下标，并按搜索方向整理端点顺序。 */
    void normalize_search_range(int &start, int &end, int max_index, bool descending)
    {
        assert(max_index >= 0);
        start = clamp_int(start, 0, max_index);
        end = clamp_int(end, 0, max_index);
        if ((descending && start < end) || (!descending && start > end))
        {
            std::swap(start, end);
        }
    }

    /** 判断图像是否适合作为 OpenCV 绘制画布。 */
    bool is_drawable_canvas(const cv::Mat &canvas)
    {
        return !canvas.empty() && canvas.dims == 2 &&
               (canvas.channels() == 1 || canvas.channels() == 3 || canvas.channels() == 4);
    }

    /** 判断输入是否为边线扫描所需的二维单通道 8 位图像。 */
    bool is_valid_binary_scan_image(const cv::Mat &image, cv::Size max_image_size)
    {
        return !image.empty() && image.dims == 2 && image.type() == CV_8UC1 &&
               image.rows > 0 && image.cols >= 2 &&
               image.rows <= max_image_size.height && image.cols <= max_image_size.width;
    }

    /** 根据累计统计量拟合 x = k*y + b，并处理退化数据。 */
    LineFit fit_line_from_sums(std::size_t count, double sx, double sy,
                               double syy, double sxy)
    {
        assert(count > 0);
        assert(std::isfinite(sx) && std::isfinite(sy) &&
               std::isfinite(syy) && std::isfinite(sxy));

        const double n = static_cast<double>(count);
        const double denominator = n * syy - sy * sy;
        const double denominator_scale = std::max(1.0, std::max(std::abs(n * syy), std::abs(sy * sy)));

        double k = 0.0;
        if (std::abs(denominator) > 1e-12 * denominator_scale)
        {
            k = (n * sxy - sx * sy) / denominator;
        }
        const double b = (sx - k * sy) / n;
        if (!std::isfinite(k) || !std::isfinite(b) ||
            std::abs(k) > std::numeric_limits<float>::max() ||
            std::abs(b) > std::numeric_limits<float>::max())
        {
            throw std::runtime_error("Line fitting produced a non-finite result.");
        }
        return {static_cast<float>(k), static_cast<float>(b)};
    }

    /** 校验图像处理配置，尽早拒绝可能导致除零或越界的参数。 */
    void validate_config(const ImageProcessConfig &config)
    {
        if (config.process_max_h <= 0 || config.process_max_w < 2 ||
            config.x_continual <= 0 || config.y_continual <= 0 ||
            config.init_stable_count <= 0 || config.miss_threshold < 0 ||
            config.search_offset < 0 || config.search_range_wide <= 0 ||
            config.search_range_narrow <= 0 || config.min_left_right_distance < 0 ||
            config.stop_kernel_w <= 0 || config.stop_min_width <= 0 ||
            config.turning_end_x_diff < 0 || config.turning_end_y_diff < 0 ||
            config.fit_max_offset < 0 || config.interp_dx_thresh <= 0 ||
            config.interp_dy_thresh <= 0 || config.turning_mid_offset < 0)
        {
            throw std::invalid_argument("ImageProcessConfig contains an invalid integer value.");
        }
        if (config.init_stable_count > config.process_max_h ||
            config.search_offset > config.process_max_w ||
            config.search_range_wide > config.process_max_w ||
            config.search_range_narrow > config.search_range_wide ||
            config.min_left_right_distance > config.process_max_w ||
            config.stop_min_width > config.process_max_w)
        {
            throw std::invalid_argument("ImageProcessConfig integer values are inconsistent with the frame limits.");
        }

        const auto in_unit_interval = [](float value)
        {
            return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
        };
        if (!in_unit_interval(config.up_ratio) || !in_unit_interval(config.down_ratio) ||
            config.up_ratio > config.down_ratio ||
            !in_unit_interval(config.task1_down_ratio) || !in_unit_interval(config.task1_up_ratio) ||
            config.task1_up_ratio > config.task1_down_ratio ||
            !in_unit_interval(config.search_range_threshold) ||
            !in_unit_interval(config.corner_y_ratio) ||
            !in_unit_interval(config.stop_roi_y0) || !in_unit_interval(config.stop_roi_y1) ||
            !in_unit_interval(config.stop_roi_x0) || !in_unit_interval(config.stop_roi_x1) ||
            config.stop_roi_y0 >= config.stop_roi_y1 || config.stop_roi_x0 >= config.stop_roi_x1 ||
            !in_unit_interval(config.fill_down_ratio) ||
            !in_unit_interval(config.turning_enter_y_thresh))
        {
            throw std::invalid_argument("ImageProcessConfig contains an invalid ratio.");
        }

        const float fit_weight_sum = config.fit_y_div_far_w + config.fit_y_div_near_w;
        if (config.corner_angle_low < 0 || config.corner_angle_high > 180 ||
            config.corner_angle_low >= config.corner_angle_high ||
            !std::isfinite(config.fit_angle_thresh) || config.fit_angle_thresh < 0.0f ||
            !std::isfinite(config.fit_y_div_far_w) || !std::isfinite(config.fit_y_div_near_w) ||
            config.fit_y_div_far_w < 0.0f || config.fit_y_div_near_w < 0.0f ||
            !std::isfinite(fit_weight_sum) || std::abs(fit_weight_sum - 1.0f) > 1e-5f)
        {
            throw std::invalid_argument("ImageProcessConfig contains an invalid floating-point value.");
        }
    }

    ImageProcessConfig validated_config(const ImageProcessConfig &config)
    {
        validate_config(config);
        return config;
    }

} // namespace

/** 使用给定配置创建图像处理器。 */
ImageProcess::ImageProcess(const ImageProcessConfig &config)
    : config_(validated_config(config))
{
    std::cout << "ImageProcess initialized with config" << std::endl;
}

ImageProcess::ValidatedScanFrame::ValidatedScanFrame(const cv::Mat &binary_image,
                                                     cv::Mat *drawing_canvas,
                                                     cv::Size max_image_size)
    : binary(binary_image), canvas(drawing_canvas)
{
    assert(max_image_size.width >= 2 && max_image_size.height > 0);
    if (!is_valid_binary_scan_image(binary, max_image_size))
    {
        throw std::invalid_argument(
            "Binary scan image must be a non-empty 2D CV_8UC1 image with width >= 2 "
            "and fit within the configured processing size.");
    }

    if (canvas != nullptr &&
        (!is_drawable_canvas(*canvas) || canvas->size() != binary.size()))
    {
        throw std::invalid_argument("Drawing canvas must be drawable and match the binary image size.");
    }
}

void ImageProcess::set_mid_line_mode(MidLineMode mode)
{
    if (mode != MID_AVG && mode != LEFT_OFFSET && mode != RIGHT_OFFSET)
        throw std::invalid_argument("Unsupported mid-line mode.");
    mid_line_mode_ = mode;
}

/** 将 OpenCV 中的 YUYV422 图像转换为 RGB 图像。 */
cv::Mat ImageProcess::yuyv_to_rgb(const cv::Mat &yuyv, YuyvColorSpace color_space) const
{
    if (yuyv.empty())
    {
        throw std::invalid_argument("YUYV image is empty.");
    }
    if (yuyv.depth() != CV_8U || (yuyv.channels() != 1 && yuyv.channels() != 2))
    {
        throw std::invalid_argument("YUYV image must be CV_8UC1 or CV_8UC2.");
    }
    if (yuyv.channels() == 1 && (yuyv.cols & 1) != 0)
    {
        throw std::invalid_argument("Single-channel YUYV image width must contain an even number of bytes.");
    }

    // OpenCV/V4L2 既可能将 YUYV 表示成 [H, W] 的 CV_8UC2，也可能表示成
    // [H, 2W] 的 CV_8UC1。公共入口只解析表示形式，共用验证器负责缓冲区契约。
    const int width = yuyv.channels() == 2 ? yuyv.cols : yuyv.cols / 2;
    return convert_yuyv_view(
        validate_yuyv_view(yuyv.data, width, yuyv.rows, yuyv.step[0]),
        color_space);
}

/** 将裸 YUYV422 缓冲区通过 RGA 转换为 RGB 图像。 */
cv::Mat ImageProcess::yuyv_to_rgb(const void *yuyv_data, int width, int height,
                                  std::size_t stride_bytes,
                                  YuyvColorSpace color_space) const
{
    return convert_yuyv_view(
        validate_yuyv_view(yuyv_data, width, height, stride_bytes),
        color_space);
}

/** 使用默认的黑色无效区域推断规则执行预处理。 */
cv::Mat1b ImageProcess::preprocess(const cv::Mat &img)
{
    return preprocess_impl(img, nullptr);
}

/** 使用显式有效区域掩膜执行预处理。 */
cv::Mat1b ImageProcess::preprocess(const cv::Mat &img, const cv::Mat &valid_mask)
{
    return preprocess_impl(img, &valid_mask);
}

/** 对输入彩色图像执行预处理。 */
cv::Mat1b ImageProcess::preprocess_impl(const cv::Mat &img, const cv::Mat *valid_mask_input)
{
    if (img.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    if (img.dims != 2 || img.channels() != 3 || img.depth() != CV_8U)
    {
        throw std::runtime_error("Input image must be a 2D CV_8UC3 image.");
    }

    const bool explicit_mask = valid_mask_input != nullptr;
    cv::Mat valid_mask;
    if (explicit_mask)
    {
        if (valid_mask_input->empty() || valid_mask_input->dims != 2 ||
            valid_mask_input->channels() != 1 ||
            valid_mask_input->rows != img.rows || valid_mask_input->cols != img.cols)
        {
            throw std::runtime_error("valid_mask shape must match the input frame height and width");
        }

        // 与 Python 中 (valid_mask != 0).astype(np.uint8) 等价。
        cv::compare(*valid_mask_input, cv::Scalar::all(0), valid_mask, cv::CMP_NE);
        if (cv::countNonZero(valid_mask) == 0)
        {
            throw std::runtime_error("valid_mask must contain at least one valid pixel");
        }
    }

    // 仅在超过配置上限时按比例缩小图像。
    // 使用局部 Mat，避免意外修改调用方持有的输入图像。
    cv::Mat frame = img;
    if (frame.rows > config_.process_max_h || frame.cols > config_.process_max_w)
    {
        const double scale = std::min(static_cast<double>(config_.process_max_h) / frame.rows,
                                      static_cast<double>(config_.process_max_w) / frame.cols);
        const int new_w = std::max(1, static_cast<int>(frame.cols * scale));
        const int new_h = std::max(1, static_cast<int>(frame.rows * scale));

        cv::resize(frame, frame, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
        if (explicit_mask)
        {
            cv::resize(valid_mask, valid_mask, cv::Size(new_w, new_h),
                       0.0, 0.0, cv::INTER_NEAREST);
        }
    }

    // 记录实际用于处理和绘制的帧，与 Python 版本在缩放后的行为一致。
    frame_ = frame.clone();

    if (!explicit_mask)
    {
        // 透视变换的 BORDER_CONSTANT 黑色区域按约定视为无效；真实场景中的
        // 纯黑像素若需保留，可通过两参数重载传入显式掩膜。
        cv::Mat black_mask;
        cv::inRange(frame, cv::Scalar::all(0), cv::Scalar::all(0), black_mask);
        cv::bitwise_not(black_mask, valid_mask);
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    const cv::Size kernel_size((gray.cols / 5) | 1, (gray.rows / 5) | 1);
    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F);

    // 使用掩膜归一化高斯卷积估计背景，避免无效黑区拉低有效区域边界的背景
    // 值并产生白色伪边。边界估计向内收缩 1 像素，但最终结果仍使用原掩膜。
    cv::Mat background_mask;
    cv::erode(valid_mask, background_mask,
              cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
              cv::Point(-1, -1), 1);

    cv::Mat mask_float;
    background_mask.convertTo(mask_float, CV_32F, 1.0 / 255.0);
    cv::Mat weighted_background = gray_float.mul(mask_float);
    cv::GaussianBlur(weighted_background, weighted_background, kernel_size, 0);

    cv::Mat blurred_mask;
    cv::GaussianBlur(mask_float, blurred_mask, kernel_size, 0);

    // 没有有效背景样本的位置回退为原灰度，使差值为零。
    cv::Mat background = gray_float.clone();
    cv::Mat has_background;
    cv::compare(blurred_mask, std::numeric_limits<float>::epsilon(),
                has_background, cv::CMP_GT);
    cv::Mat normalized_background;
    cv::divide(weighted_background, blurred_mask, normalized_background);
    normalized_background.copyTo(background, has_background);

    // 原图减去背景，截断到 [0, 255]，得到滤除光照后的特征。
    cv::Mat diff = gray_float - background;
    cv::max(diff, 0.0, diff);
    cv::min(diff, 255.0, diff);
    diff.convertTo(diff, CV_8U);
    diff.setTo(0, valid_mask == 0);

    // 二值化（使用 Otsu 自适应阈值）。
    cv::Mat binary;
    cv::threshold(diff, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    binary.setTo(0, valid_mask == 0);

    // 闭运算，并在最后再次清除无效区域，避免形态学操作向黑色边界扩散。
    cv::Mat closed;
    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
                     cv::Point(-1, -1), 3);
    closed.setTo(0, valid_mask == 0);

    assert(closed.type() == CV_8UC1);
    return cv::Mat1b(closed);
}

/** 按配置限制图像尺寸，同时保持原始宽高比。 */
void ImageProcess::resize_frame(cv::Mat &img)
{
    if (img.empty())
    {
        return;
    }

    if (img.rows > this->config_.process_max_h || img.cols > this->config_.process_max_w)
    {
        int h = img.rows;
        int w = img.cols;
        const double scale = std::min(static_cast<double>(this->config_.process_max_h) / h,
                                      static_cast<double>(this->config_.process_max_w) / w);
        const int new_w = std::max(1, static_cast<int>(w * scale));
        const int new_h = std::max(1, static_cast<int>(h * scale));

        if (new_w != w || new_h != h)
        {
            cv::resize(img, img, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
        }
    }
}

/** 对输入点执行二次曲线拟合，返回 x = a*y^2 + b*y + c 的系数。 */
std::array<float, 3> ImageProcess::polyfit_quadratic(const std::vector<float> &x, const std::vector<float> &y) const
{
    assert(x.size() == y.size() && x.size() >= 3);

    double y_center = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i)
    {
        assert(std::isfinite(x[i]) && std::isfinite(y[i]));
        y_center += static_cast<double>(y[i]);
    }
    y_center /= static_cast<double>(y.size());

    double y_scale = 0.0;
    for (const float value : y)
    {
        y_scale = std::max(y_scale, std::abs(static_cast<double>(value) - y_center));
    }
    if (y_scale <= std::numeric_limits<double>::epsilon())
    {
        throw std::runtime_error("Quadratic fitting points do not span the y axis.");
    }

    // 在归一化后的 y 坐标上求解，避免直接计算 y^4 导致病态矩阵或溢出。
    double m00 = 0.0, m01 = 0.0, m02 = 0.0;
    double m11 = 0.0, m12 = 0.0, m22 = static_cast<double>(x.size());
    double v0 = 0.0, v1 = 0.0, v2 = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double t = (static_cast<double>(y[i]) - y_center) / y_scale;
        const double t2 = t * t;
        const double xi = static_cast<double>(x[i]);
        m00 += t2 * t2;
        m01 += t2 * t;
        m02 += t2;
        m11 += t2;
        m12 += t;
        v0 += xi * t2;
        v1 += xi * t;
        v2 += xi;
    }

    const cv::Mat normal = (cv::Mat_<double>(3, 3) << m00, m01, m02,
                            m01, m11, m12,
                            m02, m12, m22);
    const cv::Mat rhs = (cv::Mat_<double>(3, 1) << v0, v1, v2);
    cv::SVD svd(normal, cv::SVD::NO_UV);
    assert(!svd.w.empty() && svd.w.total() == 3);
    const double largest_singular = svd.w.at<double>(0, 0);
    const double smallest_singular = svd.w.at<double>(svd.w.rows - 1, 0);
    if (!std::isfinite(largest_singular) || !std::isfinite(smallest_singular) ||
        largest_singular <= 0.0 || smallest_singular <= largest_singular * 1e-12)
    {
        throw std::runtime_error("The quadratic fitting matrix is singular or nearly singular.");
    }

    cv::Mat normalized_coeffs;
    if (!cv::solve(normal, rhs, normalized_coeffs, cv::DECOMP_SVD))
    {
        throw std::runtime_error("Failed to solve the quadratic fitting matrix.");
    }
    assert(normalized_coeffs.rows == 3 && normalized_coeffs.cols == 1);

    const double a_norm = normalized_coeffs.at<double>(0, 0);
    const double b_norm = normalized_coeffs.at<double>(1, 0);
    const double c_norm = normalized_coeffs.at<double>(2, 0);
    const double scale_squared = y_scale * y_scale;
    const double a = a_norm / scale_squared;
    const double b = b_norm / y_scale - 2.0 * a_norm * y_center / scale_squared;
    const double c = a_norm * y_center * y_center / scale_squared -
                     b_norm * y_center / y_scale + c_norm;
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
        std::abs(a) > std::numeric_limits<float>::max() ||
        std::abs(b) > std::numeric_limits<float>::max() ||
        std::abs(c) > std::numeric_limits<float>::max())
    {
        throw std::runtime_error("Quadratic fitting produced a non-finite result.");
    }
    return {static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)};
}

/*
* 计算两条直线的夹角，返回值为角度值
* k1, k2: 两条直线的斜率
* 逻辑：
1. 计算两条直线的夹角的正切值tan_theta = |(k2 - k1) / (1 + k1 * k2)|
2. 使用反正切函数计算夹角theta = atan(tan_theta)
3. 将夹角theta从弧度转换为角度，并返回
*/
/** 计算两条由斜率表示的直线之间的夹角，单位为度。 */
float ImageProcess::get_angle_k(float k1, float k2) const
{
    assert(std::isfinite(k1) && std::isfinite(k2));
    // 直接比较斜率向量会在极大斜率时发生乘法溢出；atan 的结果始终有限。
    const double theta = std::abs(std::atan(static_cast<double>(k2)) -
                                  std::atan(static_cast<double>(k1)));
    return static_cast<float>(theta * 180.0 / CV_PI);
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
/** 计算三个点以 p2 为顶点形成的夹角，单位为度。 */
float ImageProcess::get_angle_p(cv::Point p1, cv::Point p2, cv::Point p3) const
{
    const double v1x = static_cast<double>(p1.x) - p2.x;
    const double v1y = static_cast<double>(p1.y) - p2.y;
    const double v2x = static_cast<double>(p3.x) - p2.x;
    const double v2y = static_cast<double>(p3.y) - p2.y;

    double norm_v1 = std::hypot(v1x, v1y);
    double norm_v2 = std::hypot(v2x, v2y);

    if (norm_v1 < 1e-9 || norm_v2 < 1e-9)
    {
        return 0.0f; // 如果任一向量的模长为零，认为夹角为0度
    }

    double dot = v1x * v2x + v1y * v2y;
    double cos_theta = dot / (norm_v1 * norm_v2);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta)); // 防止越界
    double theta = std::acos(cos_theta);
    return static_cast<float>(theta * 180.0 / CV_PI); // 将角从弧度转换为角度
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
/** 将候选点加入边线，并在初始阶段通过连续点缓冲建立稳定状态。 */
bool ImageProcess::add_point_with_stable_start(std::vector<cv::Point> &line,
                                               cv::Point point,
                                               std::vector<cv::Point> &stable_buf,
                                               bool &stable) const
{
    assert(!stable || !line.empty());
    assert(point.x >= 0 && point.x < config_.process_max_w);
    assert(point.y >= 0 && point.y < config_.process_max_h);

    if (stable)
    {
        if (std::abs(line.back().x - point.x) < config_.x_continual &&
            std::abs(line.back().y - point.y) < config_.y_continual)
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
        else if (std::abs(stable_buf.back().x - point.x) < config_.x_continual &&
                 std::abs(stable_buf.back().y - point.y) < config_.y_continual)
        {
            stable_buf.push_back(point);
            if (stable_buf.size() >= static_cast<std::size_t>(this->config_.init_stable_count))
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

/** 在指定 ROI 中检测横向停止线，并按需将结果绘制到画布。 */
StopLineDetection ImageProcess::get_stop_line(const cv::Mat &binary, cv::Mat &canvas, bool is_draw)
{
    const ValidatedScanFrame frame(binary, is_draw ? &canvas : nullptr,
                                   cv::Size(config_.process_max_w, config_.process_max_h));
    int img_h = binary.rows;
    int img_w = binary.cols;

    // 定义停止线ROI
    int roi_y0 = scaled_coordinate(img_h, this->config_.stop_roi_y0);
    int roi_y1 = scaled_coordinate(img_h, this->config_.stop_roi_y1);
    int roi_x0 = scaled_coordinate(img_w, this->config_.stop_roi_x0);
    int roi_x1 = scaled_coordinate(img_w, this->config_.stop_roi_x1);
    if (roi_x1 <= roi_x0 || roi_y1 <= roi_y0)
    {
        return {};
    }
    cv::Rect stop_roi(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0);
    // 提取ROI区域
    cv::Mat roi = binary(stop_roi);

    // 横向形态学削弱斜线
    const int kernel_w = std::min(this->config_.stop_kernel_w, stop_roi.width);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_w, 1));
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
    int line_w = 0, line_h = 0, line_x = 0, line_y = 0;
    double max_score = 0;
    bool found = false;
    for (const auto &contour : contours)
    {
        assert(!contour.empty());
        const cv::Rect bbox = cv::boundingRect(contour);
        assert(bbox.width > 0 && bbox.height > 0);
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
    // 取停止线 bbox 底边中点；Rect 的 bottom 是排他边界，因此减 1。
    const cv::Rect bounds(line_x + roi_x0, line_y + roi_y0, line_w, line_h);
    const cv::Point center(bounds.x + bounds.width / 2,
                           bounds.y + bounds.height - 1);
    const StopLineDetection result(center, bounds);

    if (frame.canvas != nullptr)
    {
        const cv::Rect canvas_rect(0, 0, canvas.cols, canvas.rows);
        assert(canvas_rect.contains(center));
        assert((bounds & canvas_rect) == bounds);
        cv::circle(canvas, center, 5, cv::Scalar(0, 0, 255), -1);
        cv::rectangle(canvas, bounds, cv::Scalar(255, 0, 0), 2);
    }
    return result;
}

/** 根据停止线中心点的归一化纵坐标判断是否进入转弯阶段。 */
bool ImageProcess::judge_enter_turning(const StopLineDetection &stop_line,
                                       int img_h, int img_w, float &y_norm)
{
    (void)img_w;
    y_norm = 0.0f;
    if (!stop_line || img_h <= 0)
    {
        return false;
    }
    y_norm = stop_line.center.y / float(img_h);
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
/** 根据边线丢失与恢复状态判断转弯是否结束。 */
bool ImageProcess::judge_turing_end(int img_w, int img_h, MissLineState &miss_line)
{
    (void)img_w;
    (void)img_h;

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
 更新边线信息
*/
/** 保存当前有效边线，供下一帧丢线回退使用。 */
void ImageProcess::update_prev_frame_lines()
{
    assert(this->supple_left_line_.empty() == this->supple_right_line_.empty());
    if (!this->supple_left_line_.empty())
    {
        this->prev_supple_left_line_ = this->supple_left_line_;
        this->prev_supple_right_line_ = this->supple_right_line_;
    }
}
/*
std::vector<cv::Point> &line: 当前边线点集合，每行一个点，第一列为x坐标，第二列为y坐标
逻辑：
1. 遍历边线点集合，比较相邻点之间的坐标差异
2. 如果相邻点之间的坐标差异超过预设的阈值，则在两点之间进行线性插值，生成新的点，并将其加入边线集合
*/
/** 对边线相邻点进行插值，使点列在空间上更加连续。 */
void ImageProcess::linear_interpolation(const std::vector<cv::Point> &line,
                                        std::vector<cv::Point> &interp_points) const
{
    interp_points.clear();
    assert(!line.empty());

    interp_points.reserve(line.size());

    const double dx_threshold = config_.interp_dx_thresh;
    const double dy_threshold = config_.interp_dy_thresh;

    // 从后向前遍历，使输出的点按 y 从小到大排列
    for (size_t i = line.size() - 1; i > 0; i--)
    {
        cv::Point p1 = line[i];
        cv::Point p2 = line[i - 1];
        interp_points.push_back(p1);

        const double dx = std::abs(static_cast<double>(p1.x) - p2.x);
        const double dy = std::abs(static_cast<double>(p1.y) - p2.y);
        if (dx > dx_threshold || dy > dy_threshold)
        {
            const double required_count = std::max(dx / dx_threshold, dy / dy_threshold);
            const std::size_t interp_count = static_cast<std::size_t>(required_count);
            for (std::size_t j = 1; j < interp_count; j++)
            {
                const double alpha = static_cast<double>(j) / interp_count;
                const cv::Point interp_point(
                    cvRound((1.0 - alpha) * p1.x + alpha * p2.x),
                    cvRound((1.0 - alpha) * p1.y + alpha * p2.y));
                interp_points.push_back(interp_point);
            }
        }
    }
    interp_points.push_back(line[0]);
}

/*
std::vector<cv::Point> &left_line, &right_line: 当前左右边线点集合，每行一个点，第一列为x坐标，第二列为y坐标
cv::Size image_size: 已验证图像的宽度和高度
std::vector<cv::Point> &supple_left_line, &supple_right_line: 用于存储填充后的边线点集合
逻辑：
1. 根据图像高度计算填充的底部y坐标限制，通常为图像高度的某个比例位置
2. 判断左右边线的存在情况：
   a. 如果两边都有线，则对两边线进行插值，并将两边线的底部y坐标取较小值作为填充起点，向下填充到底部y坐标限制
   b. 如果左线丢失但右线存在，则以右线为基准进行插值，并从右线的底部y坐标开始向下填充，同时将左线的填充点x坐标设置为0
   c. 如果右线丢失但左线存在，则以左线为基准进行插值，并从左线的底部y坐标开始向下填充，同时将右线的填充点x坐标设置为图像宽度减1
*/
/** 插值并补齐左右边线，必要时使用上一帧结果作为回退。 */
void ImageProcess::fill_boundary(const std::vector<cv::Point> &left_line,
                                 const std::vector<cv::Point> &right_line,
                                 cv::Size image_size,
                                 std::vector<cv::Point> &supple_left_line,
                                 std::vector<cv::Point> &supple_right_line,
                                 bool allow_prev_fallback) const
{
    assert(image_size.width > 0 && image_size.height > 0);
    const int img_h = image_size.height;
    const int img_w = image_size.width;

    // 输出参数可能复用自上一帧，必须先清空，避免本帧无边线时沿用旧结果。
    supple_left_line.clear();
    supple_right_line.clear();

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
        const int left_bottom_y = supple_left_line.back().y;
        const int right_bottom_y = supple_right_line.back().y;
        assert(left_bottom_y >= 0 && left_bottom_y < img_h);
        assert(right_bottom_y >= 0 && right_bottom_y < img_h);
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
        if (allow_prev_fallback && !this->prev_supple_left_line_.empty() && !this->prev_supple_right_line_.empty())
        {
            const auto copy_clamped = [img_w, img_h](const std::vector<cv::Point> &source,
                                                     std::vector<cv::Point> &target)
            {
                target.reserve(source.size());
                for (const auto &point : source)
                {
                    target.emplace_back(clamp_int(point.x, 0, img_w - 1),
                                        clamp_int(point.y, 0, img_h - 1));
                }
            };
            copy_clamped(this->prev_supple_left_line_, supple_left_line);
            copy_clamped(this->prev_supple_right_line_, supple_right_line);
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

/** 对中线执行分段一次函数拟合，生成用于跟踪的拟合点列。 */
void ImageProcess::fit_polynomial()
{
    this->fit_mid_line_.clear();
    if (this->mid_line_.size() < 5)
    {
        return;
    }

    try
    {
        const double y_dive = this->mid_line_.back().y * this->config_.fit_y_div_far_w +
                              this->mid_line_.front().y * this->config_.fit_y_div_near_w;

        double near_sx = 0.0, near_sy = 0.0, near_syy = 0.0, near_sxy = 0.0;
        double far_sx = 0.0, far_sy = 0.0, far_syy = 0.0, far_sxy = 0.0;
        double near_y_min = std::numeric_limits<double>::max();
        double near_y_max = std::numeric_limits<double>::lowest();
        double far_y_min = std::numeric_limits<double>::max();
        double far_y_max = std::numeric_limits<double>::lowest();
        std::size_t near_count = 0, far_count = 0;

        // 根据 y 坐标将点分为远段和近段，统计量使用 double，避免大图或长线累加溢出。
        for (const auto &p : this->mid_line_)
        {
            const double x = p.x;
            const double y = p.y;
            if (y < y_dive)
            {
                near_sx += x;
                near_sy += y;
                near_syy += y * y;
                near_sxy += x * y;
                near_y_min = std::min(near_y_min, y);
                near_y_max = std::max(near_y_max, y);
                ++near_count;
            }
            else
            {
                far_sx += x;
                far_sy += y;
                far_syy += y * y;
                far_sxy += x * y;
                far_y_min = std::min(far_y_min, y);
                far_y_max = std::max(far_y_max, y);
                ++far_count;
            }
        }

        // 如果有一段点过少，退化为整段拟合。
        if (near_count < 2 || far_count < 2)
        {
            const std::size_t count = near_count + far_count;
            const LineFit fit = fit_line_from_sums(
                count, near_sx + far_sx, near_sy + far_sy,
                near_syy + far_syy, near_sxy + far_sxy);
            const int y_start = rounded_int_saturated(std::min(near_y_min, far_y_min));
            const int y_end = rounded_int_saturated(std::max(near_y_max, far_y_max));
            assert(y_start <= y_end);
            const std::size_t reserve_size = static_cast<std::size_t>(
                y_end - y_start + 1);
            fit_mid_line_.reserve(reserve_size);
            for (int y = y_start; y <= y_end; ++y)
            {
                const int x = rounded_int_saturated(static_cast<double>(fit.k) * y + fit.b);
                fit_mid_line_.emplace_back(x, static_cast<int>(y));
            }
            return;
        }

        // 分段线性拟合。
        const LineFit near_fit = fit_line_from_sums(
            near_count, near_sx, near_sy, near_syy, near_sxy);
        const LineFit far_fit = fit_line_from_sums(
            far_count, far_sx, far_sy, far_syy, far_sxy);

        const float angle = get_angle_k(near_fit.k, far_fit.k);
        int offset = 0;
        if (angle > this->config_.fit_angle_thresh)
        {
            const float sign_k_far = (far_fit.k > 0.0f) ? 1.0f : ((far_fit.k < 0.0f) ? -1.0f : 0.0f);
            const float factor = std::min(angle / 60.0f, 1.0f);
            offset = rounded_int_saturated(-sign_k_far * factor * this->config_.fit_max_offset);
        }

        const int far_y_start = rounded_int_saturated(far_y_min);
        const int far_y_end = rounded_int_saturated(far_y_max);
        const int near_y_start = rounded_int_saturated(near_y_min);
        const int near_y_end = rounded_int_saturated(near_y_max);
        assert(near_y_start <= near_y_end && far_y_start <= far_y_end);

        const std::size_t near_size = static_cast<std::size_t>(
            near_y_end - near_y_start + 1);
        const std::size_t far_size = static_cast<std::size_t>(
            far_y_end - far_y_start + 1);
        assert(near_size <= std::numeric_limits<std::size_t>::max() - far_size);
        fit_mid_line_.reserve(near_size + far_size);

        // 按 y 从小到大生成：远段在前，近段在后。
        for (int y = near_y_start; y <= near_y_end; ++y)
        {
            const int x = rounded_int_saturated(static_cast<double>(near_fit.k) * y + near_fit.b + offset);
            fit_mid_line_.emplace_back(x, static_cast<int>(y));
        }
        for (int y = far_y_start; y <= far_y_end; ++y)
        {
            const int x = rounded_int_saturated(static_cast<double>(far_fit.k) * y + far_fit.b + offset);
            fit_mid_line_.emplace_back(x, static_cast<int>(y));
        }
    }
    catch (const std::runtime_error &e)
    {
        fit_mid_line_.clear();
        std::cerr << "Line fitting failed: " << e.what() << std::endl;
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
/** 对中线执行二次函数拟合，生成平滑的预测中线。 */
void ImageProcess::fit_polynomial2()
{
    this->fit_mid_line_.clear();
    size_t N = this->mid_line_.size();
    if (N < 3)
    {
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

        const int y_start = std::min(this->mid_line_.front().y, this->mid_line_.back().y);
        const int y_end = std::max(this->mid_line_.front().y, this->mid_line_.back().y);
        const std::size_t span = static_cast<std::size_t>(
            y_end - y_start + 1);
        this->fit_mid_line_.reserve((span + 2) / 3);
        for (int y = y_start; y <= y_end; y += 3)
        {
            const int x = rounded_int_saturated(a * y * y + b * y + c);
            this->fit_mid_line_.emplace_back(cv::Point(x, static_cast<int>(y)));
        }
    }
    catch (const std::runtime_error &e)
    {
        this->fit_mid_line_.clear();
        std::cerr << "Polynomial fitting failed: " << e.what() << std::endl;
    }
}

void ImageProcess::get_side_line_task_1(const cv::Mat &img, cv::Mat &canvas, bool is_draw)
{
    const ValidatedScanFrame frame(img, is_draw ? &canvas : nullptr,
                                   cv::Size(config_.process_max_w, config_.process_max_h));
    scan_side_lines(frame, config_.task1_down_ratio, config_.task1_up_ratio,
                    false, BOTH);
}

void ImageProcess::get_side_line_task_2(const cv::Mat &img, cv::Mat &canvas,
                                        bool is_draw, bool find_corner, SearchSide side)
{
    if (side != BOTH && side != LEFT_ONLY && side != RIGHT_ONLY)
        throw std::invalid_argument("Unsupported side-line search mode.");
    const ValidatedScanFrame frame(img, is_draw ? &canvas : nullptr,
                                   cv::Size(config_.process_max_w, config_.process_max_h));
    scan_side_lines(frame, config_.down_ratio, config_.up_ratio,
                    find_corner, side);
}

/** 对已验证的二值帧逐行检测边线，两种 task 只提供不同静态策略。 */
void ImageProcess::scan_side_lines(const ValidatedScanFrame &frame,
                                   float scan_down_ratio, float scan_up_ratio,
                                   bool find_corner, SearchSide side)
{
    assert(side == BOTH || side == LEFT_ONLY || side == RIGHT_ONLY);
    assert(std::isfinite(scan_down_ratio) && std::isfinite(scan_up_ratio));
    assert(scan_up_ratio >= 0.0f && scan_up_ratio <= scan_down_ratio && scan_down_ratio <= 1.0f);

    const cv::Mat &img = frame.binary;

    const int mid_x = img.cols / 2;
    const int img_h = img.rows;
    const int img_w = img.cols;
    const int max_edge_x = img_w - 2; // row_diff 的最后一个有效下标
    const int scan_y_start = clamp_int(scaled_coordinate(img_h, scan_down_ratio), 0, img_h - 1);
    const int scan_y_end = clamp_int(scaled_coordinate(img_h, scan_up_ratio), 0, img_h - 1);
    const bool draw = frame.canvas != nullptr;

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
    std::array<cv::Point, 2> left_corner_history{};
    std::array<cv::Point, 2> right_corner_history{};
    std::size_t left_corner_count = 0;
    std::size_t right_corner_count = 0;

    clear_lines();

    int lx = 0, rx = 0, y = 0;
    for (y = scan_y_start; y >= scan_y_end; y--)
    {
        // 逐行计算相邻像素差异，避免计算整张图
        cv::Mat row_mask = (img.row(y) == 0);
        cv::Mat row_left = row_mask(cv::Range::all(), cv::Range(0, img_w - 1));
        cv::Mat row_right = row_mask(cv::Range::all(), cv::Range(1, img_w));
        cv::Mat row_diff;
        cv::bitwise_xor(row_left, row_right, row_diff);

        // 动态搜索：二段阶梯
        const float y_norm = static_cast<float>(y) / static_cast<float>(img_h);
        int cur_range = y_norm > this->config_.search_range_threshold ? this->config_.search_range_wide : this->config_.search_range_narrow;

        // 共用行指针与本行搜索结果/添加标志（左右搜索块及 L+R 距离检查都要用）
        const uchar *row_ptr = row_diff.ptr<uchar>(0);
        lx = -1; // 重置 x，避免使用旧值
        rx = -1;
        bool left_added = false;
        bool right_added = false;

        // 搜索左边线
        if (side != RIGHT_ONLY)
        {
            if (left_stable_flag)
            {
                // 左侧稳定时，围绕上一帧位置向左右搜索
                search_left_start = std::min(prev_row_left_x + cur_range, max_edge_x);
                search_left_end = std::max(prev_row_left_x - cur_range, 0);
            }
            else
            {
                // 左侧不稳定时，从中线偏左位置向左搜索到图像边缘
                search_left_start = mid_x - this->config_.search_offset;
                search_left_end = 0;
            }

            normalize_search_range(search_left_start, search_left_end, max_edge_x, true);
            if (draw)
            {
                cv::circle(*frame.canvas, cv::Point(search_left_start, y), 1, cv::Scalar(0, 255, 255), -1);
                cv::circle(*frame.canvas, cv::Point(search_left_end, y), 1, cv::Scalar(0, 255, 255), -1);
            }

            for (int i = search_left_start; i >= search_left_end; i--)
            {
                if (row_ptr[i] != 0)
                { // 非0表示存在黑白跳变点
                    lx = i;
                    break;
                }
            }

            // 先进行稳定点判断
            if (lx >= 0)
            {
                assert(lx <= max_edge_x);
                left_added = add_point_with_stable_start(
                    this->left_line_, cv::Point(lx, y), left_stable_buf, left_stable_flag);
            }

            // 如果当前行的点没有加入左边线，则下一行的搜索起点不更新
            if (left_added)
            {
                prev_row_left_x = lx;
                // 只有稳定点才参与拐点检测
                if (find_corner && !find_left_corner)
                {
                    const cv::Point current(lx, y);
                    if (left_corner_count == left_corner_history.size())
                    {
                        const float angle = get_angle_p(
                            left_corner_history[0], left_corner_history[1], current);
                        if (angle < this->config_.corner_angle_high && angle > this->config_.corner_angle_low)
                        {
                            find_left_corner = true;
                            this->left_corners_ = left_corner_history[1];
                        }
                        left_corner_history[0] = left_corner_history[1];
                        left_corner_history[1] = current;
                    }
                    else
                    {
                        left_corner_history[left_corner_count++] = current;
                    }
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
        }

        // 搜索右边线
        if (side != LEFT_ONLY)
        {
            if (right_stable_flag)
            {
                // 右侧稳定时，围绕上一帧位置向左右搜索
                search_right_start = std::max(prev_row_right_x - cur_range, 0);
                search_right_end = std::min(prev_row_right_x + cur_range, max_edge_x);
            }
            else
            {
                // 右侧不稳定时，从中线偏右位置向右搜索到图像边缘
                search_right_start = mid_x + this->config_.search_offset;
                search_right_end = max_edge_x;
            }

            normalize_search_range(search_right_start, search_right_end, max_edge_x, false);
            if (draw)
            {
                cv::circle(*frame.canvas, cv::Point(search_right_start, y), 1, cv::Scalar(255, 255, 0), -1);
                cv::circle(*frame.canvas, cv::Point(search_right_end, y), 1, cv::Scalar(255, 255, 0), -1);
            }

            for (int i = search_right_start; i <= search_right_end; i++)
            {
                if (row_ptr[i] != 0)
                { // 非0表示存在黑白跳变点
                    rx = i;
                    break; // 右侧搜索：找到第一个跳变点就停止
                }
            }
            if (rx >= 0)
            {
                assert(rx <= max_edge_x);
                right_added = add_point_with_stable_start(
                    this->right_line_, cv::Point(rx, y), right_stable_buf, right_stable_flag);
            }

            if (right_added)
            {
                prev_row_right_x = rx;
                // 只有稳定点才参与拐点检测
                if (find_corner && !find_right_corner)
                {
                    const cv::Point current(rx, y);
                    if (right_corner_count == right_corner_history.size())
                    {
                        const float angle = get_angle_p(
                            right_corner_history[0], right_corner_history[1], current);
                        if (angle < this->config_.corner_angle_high && angle > this->config_.corner_angle_low)
                        {
                            find_right_corner = true;
                            this->right_corners_ = right_corner_history[1];
                        }
                        right_corner_history[0] = right_corner_history[1];
                        right_corner_history[1] = current;
                    }
                    else
                    {
                        right_corner_history[right_corner_count++] = current;
                    }
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
        }

        // 检测左右边线距离，如果太小则认为是噪点，移除这一对
        // (side != BOTH 时被跳过的一侧 *_added 为 false，这里自然短路)
        if (left_added && right_added)
        {
            const int distance = std::abs(static_cast<int>(lx) - rx);
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

/*
 * 根据当前左右原始边线插值+填充，并计算中线
 * image_size: 已验证二值图的几何尺寸
 * 逻辑：
 * 1. 调用 fill_boundary 对左右边线做插值和底部填充，得到 supple_left_line_/supple_right_line_
 * 2. 按相同索引对左右填充线的点求中点，写入 mid_line_
 * 3. 更新 prev_* 成员，供下一帧 fill_boundary 的 allow_prev_fallback 使用
 */
/** 根据当前左右边线补线，并按照当前模式计算中线；MID_AVG 模式下按 left_weight 对左右边线加权。 */
void ImageProcess::calculate_mid_line(cv::Size image_size, float left_weight)
{
    if (image_size.width <= 0 || image_size.height <= 0 ||
        image_size.width > config_.process_max_w || image_size.height > config_.process_max_h)
    {
        throw std::invalid_argument("Mid-line image size is outside the configured processing bounds.");
    }
    if (!std::isfinite(left_weight) || left_weight < 0.0f || left_weight > 1.0f)
        throw std::invalid_argument("Left boundary weight must be finite and in [0, 1].");
    assert(mid_line_mode_ == MID_AVG || mid_line_mode_ == LEFT_OFFSET ||
           mid_line_mode_ == RIGHT_OFFSET);
    const int img_w = image_size.width;

    // 插值+填充边线（同时处理边线情况）
    fill_boundary(this->left_line_, this->right_line_, image_size,
                  this->supple_left_line_, this->supple_right_line_, true);

    if (this->mid_line_mode_ == LEFT_OFFSET)
    {
        // 用左边线 + 偏移
        const std::size_t n = this->supple_left_line_.size();
        this->mid_line_.resize(n);
        for (std::size_t i = 0; i < n; i++)
        {
            const double x = static_cast<double>(this->supple_left_line_[i].x) +
                             this->config_.turning_mid_offset;
            this->mid_line_[i].x = clamp_int(rounded_int_saturated(x), 0, img_w - 1);
            this->mid_line_[i].y = this->supple_left_line_[i].y;
        }
    }
    else if (this->mid_line_mode_ == RIGHT_OFFSET)
    {
        // 用右边线 - 偏移
        const std::size_t n = this->supple_right_line_.size();
        this->mid_line_.resize(n);
        for (std::size_t i = 0; i < n; i++)
        {
            const double x = static_cast<double>(this->supple_right_line_[i].x) -
                             this->config_.turning_mid_offset;
            this->mid_line_[i].x = clamp_int(rounded_int_saturated(x), 0, img_w - 1);
            this->mid_line_[i].y = this->supple_right_line_[i].y;
        }
    }
    else // MID_AVG
    {
        // 使用优化后的边线计算中线：x 按左线权重 left_weight 加权融合，y 仍取均值
        const std::size_t n = std::min(this->supple_left_line_.size(), this->supple_right_line_.size());
        this->mid_line_.resize(n);
        for (std::size_t i = 0; i < n; i++)
        {
            this->mid_line_[i].x = static_cast<int>(
                this->supple_left_line_[i].x * left_weight +
                this->supple_right_line_[i].x * (1.0f - left_weight));
            this->mid_line_[i].y = (this->supple_left_line_[i].y + this->supple_right_line_[i].y) / 2.0;
        }
    }

    this->update_prev_frame_lines();
}

/** 在画布上绘制边线、中线、拟合线、状态信息和目标点。 */
void ImageProcess::draw_line(cv::Mat &canvas, float fps, const std::string &state,
                             TrackingTarget target)
{
    if (!is_drawable_canvas(canvas))
        throw std::invalid_argument("Drawing canvas must be a non-empty 2D image with 1, 3, or 4 channels.");
    if (target != TrackingTarget::STRAIGHT && target != TrackingTarget::TURNING &&
        target != TrackingTarget::TRACKING2)
    {
        throw std::invalid_argument("Unsupported tracking target.");
    }

    if (canvas.channels() == 1)
    {
        cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);
    }

    if (std::isfinite(fps) && fps >= 0.0f)
    {
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(canvas, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    if (!state.empty())
    {
        cv::putText(canvas, state, cv::Point(std::max(0, canvas.cols - 100), 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
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
    int target_idx = 0;
    switch (target)
    {
    case TrackingTarget::STRAIGHT:
        target_idx = this->config_.straight_target_p_index;
        break;
    case TrackingTarget::TURNING:
        target_idx = this->config_.left_target_p_index;
        break;
    case TrackingTarget::TRACKING2:
        target_idx = this->config_.tracking2_target_p_index;
        break;
    }

    // std::cout << "Drawing target point at index: " << target_idx << std::endl;

    // 处理负索引
    const std::int64_t line_size = static_cast<std::int64_t>(this->fit_mid_line_.size());
    std::int64_t target_index = target_idx;
    if (target_index < 0)
    {
        target_index = line_size + target_index; // 转换为正索引
    }

    if (target_index >= 0 && target_index < line_size)
    {
        cv::circle(canvas, this->fit_mid_line_[static_cast<std::size_t>(target_index)],
                   4, cv::Scalar(255, 0, 255), -1);
    }
}

/** 清除当前帧的边线、中线、拟合结果和拐点状态。 */
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

/** 返回当前保存的帧图像。 */
cv::Mat ImageProcess::return_frame() const
{
    if (this->frame_.empty())
        throw std::runtime_error("No processed frame is available.");
    return this->frame_;
}
