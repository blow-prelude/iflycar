#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace vision_line
{
struct WarpGeometry
{
    cv::Mat homography;
    cv::Size output_size;
};

cv::Mat makeFixedGroundMask(const cv::Size &frame_size, int start_y);
std::vector<cv::Point2f> makeMetricDestination(
    double width_m, double length_m, double pixels_per_m);
WarpGeometry makeExpandedHomography(
    const std::vector<cv::Point2f> &source_points,
    const std::vector<cv::Point2f> &destination_points,
    const cv::Mat &ground_mask);
cv::Mat warpGround(
    const cv::Mat &frame,
    const cv::Mat &ground_mask,
    const std::vector<cv::Point2f> &source_points,
    double width_m,
    double length_m,
    double pixels_per_m);
cv::Mat flipAndResize(const cv::Mat &corrected_frame,
                      const cv::Size &output_size);
bool shouldExit(int key_code);
} // namespace vision_line
