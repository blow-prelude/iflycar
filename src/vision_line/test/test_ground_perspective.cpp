#include "ground_perspective.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <vector>

namespace
{
const std::vector<cv::Point2f> kSourcePoints{
    {114.0F, 146.0F}, {206.0F, 146.0F},
    {271.0F, 184.0F}, {35.0F, 187.0F}};
}

TEST(FixedGroundMask, KeepsRowsStartingAt139)
{
    const cv::Mat mask = vision_line::makeFixedGroundMask({320, 240}, 139);
    EXPECT_EQ(0, cv::countNonZero(mask.row(138)));
    EXPECT_EQ(320, cv::countNonZero(mask.row(139)));
    EXPECT_EQ(320 * (240 - 139), cv::countNonZero(mask));
}

TEST(FixedGroundMask, ClampsStartRowToImage)
{
    EXPECT_EQ(12, cv::countNonZero(
                      vision_line::makeFixedGroundMask({4, 3}, -1)));
    EXPECT_EQ(0, cv::countNonZero(
                     vision_line::makeFixedGroundMask({4, 3}, 9)));
}

TEST(MetricDestination, ConvertsMetresToPixels)
{
    const auto points = vision_line::makeMetricDestination(0.5, 0.75, 200.0);
    ASSERT_EQ(4U, points.size());
    EXPECT_EQ(cv::Point2f(0.0F, 0.0F), points[0]);
    EXPECT_EQ(cv::Point2f(100.0F, 0.0F), points[1]);
    EXPECT_EQ(cv::Point2f(100.0F, 150.0F), points[2]);
    EXPECT_EQ(cv::Point2f(0.0F, 150.0F), points[3]);
}

TEST(ExpandedHomography, PreservesCalibrationGeometryAndContainsGroundRoi)
{
    const cv::Mat mask = vision_line::makeFixedGroundMask({320, 240}, 139);
    const auto destination =
        vision_line::makeMetricDestination(0.5, 0.75, 200.0);
    const auto geometry = vision_line::makeExpandedHomography(
        kSourcePoints, destination, mask);

    std::vector<cv::Point2f> mapped_source;
    cv::perspectiveTransform(kSourcePoints, mapped_source,
                             geometry.homography);
    EXPECT_NEAR(100.0, mapped_source[1].x - mapped_source[0].x, 1.0e-3);
    EXPECT_NEAR(0.0, mapped_source[1].y - mapped_source[0].y, 1.0e-3);
    EXPECT_NEAR(0.0, mapped_source[3].x - mapped_source[0].x, 1.0e-3);
    EXPECT_NEAR(150.0, mapped_source[3].y - mapped_source[0].y, 1.0e-3);

    const std::vector<cv::Point2f> roi{
        {0.0F, 139.0F}, {319.0F, 139.0F},
        {319.0F, 239.0F}, {0.0F, 239.0F}};
    std::vector<cv::Point2f> mapped_roi;
    cv::perspectiveTransform(roi, mapped_roi, geometry.homography);
    ASSERT_GT(geometry.output_size.width, 0);
    ASSERT_GT(geometry.output_size.height, 0);
    for (const auto &point : mapped_roi)
    {
        EXPECT_GE(point.x, -1.0e-3F);
        EXPECT_GE(point.y, -1.0e-3F);
        EXPECT_LE(point.x, geometry.output_size.width - 1.0F + 1.0e-3F);
        EXPECT_LE(point.y, geometry.output_size.height - 1.0F + 1.0e-3F);
    }
}

TEST(WarpGround, KeepsTheCompleteFixedGroundRoi)
{
    const cv::Mat frame(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat mask = vision_line::makeFixedGroundMask({4, 4}, 2);
    const std::vector<cv::Point2f> source{
        {0.0F, 0.0F}, {3.0F, 0.0F}, {3.0F, 3.0F}, {0.0F, 3.0F}};
    const cv::Mat warped =
        vision_line::warpGround(frame, mask, source, 3.0, 3.0, 1.0);
    EXPECT_EQ(cv::Size(4, 2), warped.size());
    EXPECT_EQ(frame.type(), warped.type());
    EXPECT_EQ(cv::Vec3b(10, 20, 30), warped.at<cv::Vec3b>(0, 0));
}

TEST(FramePreparation, FlipsCorrectedFrameBeforeResize)
{
    cv::Mat corrected(1, 2, CV_8UC1);
    corrected.at<unsigned char>(0, 0) = 1;
    corrected.at<unsigned char>(0, 1) = 2;
    const cv::Mat prepared = vision_line::flipAndResize(corrected, {2, 1});
    EXPECT_EQ(2, prepared.at<unsigned char>(0, 0));
    EXPECT_EQ(1, prepared.at<unsigned char>(0, 1));
}

TEST(KeyHandling, AcceptsQUppercaseQAndEscape)
{
    EXPECT_TRUE(vision_line::shouldExit('q'));
    EXPECT_TRUE(vision_line::shouldExit('Q'));
    EXPECT_TRUE(vision_line::shouldExit(27));
    EXPECT_FALSE(vision_line::shouldExit('x'));
}
