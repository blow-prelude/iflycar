#include "ground_perspective.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vision_line
{
cv::Mat makeFixedGroundMask(const cv::Size &frame_size, int start_y)
{
    if (frame_size.width <= 0 || frame_size.height <= 0)
    {
        throw std::invalid_argument("frame size must be positive");
    }
    const int top = std::max(0, std::min(start_y, frame_size.height));
    cv::Mat mask = cv::Mat::zeros(frame_size, CV_8UC1);
    if (top < frame_size.height)
    {
        mask(cv::Rect(0, top, frame_size.width,
                      frame_size.height - top)).setTo(255);
    }
    return mask;
}

std::vector<cv::Point2f> makeMetricDestination(
    double width_m, double length_m, double pixels_per_m)
{
    if (width_m <= 0.0 || length_m <= 0.0 || pixels_per_m <= 0.0)
    {
        throw std::invalid_argument("metric dimensions must be positive");
    }
    const float width_px = static_cast<float>(width_m * pixels_per_m);
    const float length_px = static_cast<float>(length_m * pixels_per_m);
    return {{0.0F, 0.0F}, {width_px, 0.0F},
            {width_px, length_px}, {0.0F, length_px}};
}

WarpGeometry makeExpandedHomography(
    const std::vector<cv::Point2f> &source_points,
    const std::vector<cv::Point2f> &destination_points,
    const cv::Mat &ground_mask)
{
    if (source_points.size() != 4U || destination_points.size() != 4U)
    {
        throw std::invalid_argument("exactly four point pairs are required");
    }
    if (ground_mask.empty() || ground_mask.type() != CV_8UC1)
    {
        throw std::invalid_argument("ground mask must be a non-empty CV_8UC1 image");
    }

    std::vector<cv::Point> foreground;
    cv::findNonZero(ground_mask, foreground);
    if (foreground.empty())
    {
        throw std::invalid_argument("ground mask contains no retained pixels");
    }
    const cv::Rect bounds = cv::boundingRect(foreground);
    const float left = static_cast<float>(bounds.x);
    const float top = static_cast<float>(bounds.y);
    const float right = static_cast<float>(bounds.x + bounds.width - 1);
    const float bottom = static_cast<float>(bounds.y + bounds.height - 1);
    const std::vector<cv::Point2f> roi{
        {left, top}, {right, top}, {right, bottom}, {left, bottom}};

    const cv::Mat base =
        cv::getPerspectiveTransform(source_points, destination_points);
    std::vector<cv::Point2f> transformed;
    cv::perspectiveTransform(roi, transformed, base);

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto &point : transformed)
    {
        min_x = std::min(min_x, static_cast<double>(point.x));
        min_y = std::min(min_y, static_cast<double>(point.y));
        max_x = std::max(max_x, static_cast<double>(point.x));
        max_y = std::max(max_y, static_cast<double>(point.y));
    }
    min_x = std::floor(min_x);
    min_y = std::floor(min_y);
    max_x = std::ceil(max_x);
    max_y = std::ceil(max_y);

    const cv::Size output_size(
        static_cast<int>(max_x - min_x) + 1,
        static_cast<int>(max_y - min_y) + 1);
    if (output_size.width <= 0 || output_size.height <= 0)
    {
        throw std::invalid_argument("transformed ground ROI has invalid size");
    }

    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = -min_x;
    translation.at<double>(1, 2) = -min_y;
    return {translation * base, output_size};
}

cv::Mat warpGround(
    const cv::Mat &frame,
    const cv::Mat &ground_mask,
    const std::vector<cv::Point2f> &source_points,
    double width_m,
    double length_m,
    double pixels_per_m)
{
    if (frame.empty() || frame.channels() != 3)
    {
        throw std::invalid_argument("frame must be a non-empty BGR image");
    }
    if (ground_mask.size() != frame.size() || ground_mask.type() != CV_8UC1)
    {
        throw std::invalid_argument("ground mask must match the frame");
    }
    const auto destination =
        makeMetricDestination(width_m, length_m, pixels_per_m);
    const auto geometry =
        makeExpandedHomography(source_points, destination, ground_mask);
    cv::Mat masked_frame;
    cv::bitwise_and(frame, frame, masked_frame, ground_mask);
    cv::Mat warped;
    cv::warpPerspective(masked_frame, warped, geometry.homography,
                        geometry.output_size, cv::INTER_LINEAR);
    return warped;
}

cv::Mat flipAndResize(const cv::Mat &corrected_frame,
                      const cv::Size &output_size)
{
    if (corrected_frame.empty() || output_size.width <= 0 ||
        output_size.height <= 0)
    {
        throw std::invalid_argument("frame and output size must be valid");
    }
    cv::Mat flipped;
    cv::flip(corrected_frame, flipped, 1);
    cv::Mat resized;
    cv::resize(flipped, resized, output_size, 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

bool shouldExit(int key_code)
{
    const int key = key_code & 0xFF;
    return key == 27 || key == 'q' || key == 'Q';
}
} // namespace vision_line
