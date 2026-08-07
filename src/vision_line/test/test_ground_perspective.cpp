#include "ground_perspective.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <stdexcept>
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

TEST(InputValidation, RejectsNonPositiveFrameSizes)
{
    EXPECT_THROW(vision_line::makeFixedGroundMask({0, 3}, 0),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::makeFixedGroundMask({3, 0}, 0),
                 std::invalid_argument);
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

TEST(InputValidation, RejectsNonPositiveMetricDimensions)
{
    EXPECT_THROW(vision_line::makeMetricDestination(0.0, 0.75, 200.0),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::makeMetricDestination(0.5, -0.75, 200.0),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::makeMetricDestination(0.5, 0.75, 0.0),
                 std::invalid_argument);
}

TEST(InputValidation, RejectsInvalidHomographyInputs)
{
    const std::vector<cv::Point2f> points{
        {0.0F, 0.0F}, {3.0F, 0.0F}, {3.0F, 3.0F}, {0.0F, 3.0F}};
    const cv::Mat valid_mask = cv::Mat::ones(4, 4, CV_8UC1);

    EXPECT_THROW(vision_line::makeExpandedHomography(
                     {points.begin(), points.begin() + 3}, points,
                     valid_mask),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::makeExpandedHomography(
                     points, {points.begin(), points.begin() + 3},
                     valid_mask),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::makeExpandedHomography(points, points, cv::Mat()),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::makeExpandedHomography(
                     points, points, cv::Mat::ones(4, 4, CV_8UC3)),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::makeExpandedHomography(
                     points, points, cv::Mat::zeros(4, 4, CV_8UC1)),
                 std::invalid_argument);
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

TEST(InputValidation, RejectsInvalidWarpFrameAndMaskInputs)
{
    const std::vector<cv::Point2f> source{
        {0.0F, 0.0F}, {3.0F, 0.0F}, {3.0F, 3.0F}, {0.0F, 3.0F}};
    const cv::Mat valid_frame(4, 4, CV_8UC3);
    const cv::Mat valid_mask = cv::Mat::ones(4, 4, CV_8UC1);

    EXPECT_THROW(vision_line::warpGround(cv::Mat(), valid_mask, source,
                                          3.0, 3.0, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::warpGround(cv::Mat::ones(4, 4, CV_8UC1),
                                          valid_mask, source, 3.0, 3.0, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::warpGround(valid_frame,
                                          cv::Mat::ones(3, 4, CV_8UC1),
                                          source, 3.0, 3.0, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::warpGround(valid_frame,
                                          cv::Mat::ones(4, 4, CV_8UC3),
                                          source, 3.0, 3.0, 1.0),
                 std::invalid_argument);
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

TEST(InputValidation, RejectsInvalidFramePreparationInputs)
{
    const cv::Mat frame(1, 1, CV_8UC1);
    EXPECT_THROW(vision_line::flipAndResize(cv::Mat(), {1, 1}),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::flipAndResize(frame, {0, 1}),
                 std::invalid_argument);
    EXPECT_THROW(vision_line::flipAndResize(frame, {1, -1}),
                 std::invalid_argument);
}

TEST(KeyHandling, AcceptsQUppercaseQAndEscape)
{
    EXPECT_TRUE(vision_line::shouldExit('q'));
    EXPECT_TRUE(vision_line::shouldExit('Q'));
    EXPECT_TRUE(vision_line::shouldExit(27));
    EXPECT_FALSE(vision_line::shouldExit('x'));
}

TEST(FixedGroundPerspective, UsesCalibratedOutputGeometry)
{
    const cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(10, 20, 30));

    const cv::Mat warped = vision_line::warpFixedGroundPerspective(frame);

    EXPECT_EQ(cv::Size(484, 299), warped.size());
    EXPECT_EQ(CV_8UC3, warped.type());
}

TEST(FixedGroundPerspective, RemovesPixelsAboveGroundStartBeforeWarp)
{
    cv::Mat frame = cv::Mat::zeros(240, 320, CV_8UC3);
    frame(cv::Rect(0, 0, 320, 139)).setTo(cv::Scalar(10, 20, 30));

    const cv::Mat warped = vision_line::warpFixedGroundPerspective(frame);

    EXPECT_EQ(0, cv::countNonZero(warped.reshape(1)));
}

TEST(FixedGroundPerspective, RejectsEmptyFrame)
{
    EXPECT_THROW(vision_line::warpFixedGroundPerspective(cv::Mat()),
                 std::invalid_argument);
}

TEST(FixedGroundPerspective, RejectsNonBgr8BitFrame)
{
    const cv::Mat gray(240, 320, CV_8UC1);
    EXPECT_THROW(vision_line::warpFixedGroundPerspective(gray),
                 std::invalid_argument);
}

TEST(FixedGroundPerspective, RejectsUncalibratedFrameSize)
{
    const cv::Mat wrong_size(239, 320, CV_8UC3);
    EXPECT_THROW(vision_line::warpFixedGroundPerspective(wrong_size),
                 std::invalid_argument);
}
