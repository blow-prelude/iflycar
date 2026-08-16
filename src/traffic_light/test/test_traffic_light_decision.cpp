#include <string>

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "traffic_light_decision.hpp"

namespace
{

cv::Mat makeArrowImage(const std::string &direction)
{
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(25, 25, 25));
    cv::rectangle(image, cv::Rect(280, 200, 80, 50), cv::Scalar(0, 180, 0), -1);

    cv::Mat arrow = cv::Mat::zeros(48, 64, CV_8UC1);
    cv::rectangle(arrow, cv::Point(8, 20), cv::Point(40, 28), 255, -1);
    const cv::Point head[] = {
        cv::Point(36, 7), cv::Point(59, 24), cv::Point(36, 41)};
    cv::fillConvexPoly(arrow, head, 3, 255);
    if (direction == "left")
    {
        cv::flip(arrow, arrow, 1);
    }
    else if (direction != "right")
    {
        cv::rotate(arrow, arrow, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    cv::Mat white_arrow;
    arrow.copyTo(white_arrow);
    cv::Mat arrow_bgr;
    cv::cvtColor(white_arrow, arrow_bgr, cv::COLOR_GRAY2BGR);
    arrow_bgr.setTo(cv::Scalar(255, 255, 255), white_arrow);
    arrow_bgr.copyTo(image(cv::Rect(288, 201, 64, 48)), white_arrow);
    return image;
}

} // namespace

TEST(TrafficLightDecision, ClassifiesSyntheticLeftAndRightArrows)
{
    const cv::Mat left = makeArrowImage("left");
    const cv::Mat right = makeArrowImage("right");
    const cv::Rect box(280, 200, 80, 50);

    EXPECT_EQ("left", traffic_light::classifyArrowDirection(left, box));
    EXPECT_EQ("right", traffic_light::classifyArrowDirection(right, box));
}

TEST(TrafficLightDecision, ClassifiesCapturedLeftAndRightArrows)
{
    const std::string picture_dir = VISION_LINE_TEST_PICTURE_DIR;
    cv::Mat left = cv::imread(
        picture_dir + "/captured_image_20260816_231732.jpg");
    cv::Mat right = cv::imread(
        picture_dir + "/captured_image_20260816_231728.jpg");
    ASSERT_FALSE(left.empty());
    ASSERT_FALSE(right.empty());

    const cv::Rect captured_boxes[] = {
        cv::Rect(130, 107, 32, 30),
        cv::Rect(132, 108, 24, 24),
        cv::Rect(125, 103, 21, 29)};
    for (int scale = 1; scale <= 2; ++scale)
    {
        for (const cv::Rect &captured_box : captured_boxes)
        {
            const cv::Rect traffic_light_box(
                captured_box.x * scale,
                captured_box.y * scale,
                captured_box.width * scale,
                captured_box.height * scale);
            EXPECT_EQ("left",
                      traffic_light::classifyArrowDirection(
                          left, traffic_light_box));
            EXPECT_EQ("right",
                      traffic_light::classifyArrowDirection(
                          right, traffic_light_box));
        }

        if (scale == 1)
        {
            cv::resize(left, left, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
            cv::resize(right, right, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        }
    }
}

TEST(TrafficLightDecision, ClassifiesExistingCapturedArrowSet)
{
    struct CapturedArrow
    {
        const char *filename;
        const char *expected_label;
    };
    const CapturedArrow captured_arrows[] = {
        {"00010.jpg", "right"},
        {"00017.jpg", "left"},
        {"00107.jpg", "right"},
        {"003_0030.jpg", "left"},
        {"006_0018.jpg", "left"},
        {"capture_1779365603.jpg", "right"},
        {"capture_1779365606.jpg", "right"},
        {"capture_1779365607.jpg", "right"},
        {"capture_1779365615.jpg", "right"},
        {"capture_1786544471347320238_000720.jpg", "right"},
        {"capture_1786544476988825876_000839.jpg", "left"}};

    const std::string picture_dir = TRAFFIC_LIGHT_TEST_PICTURE_DIR;
    for (const CapturedArrow &captured_arrow : captured_arrows)
    {
        const cv::Mat image = cv::imread(
            picture_dir + "/" + captured_arrow.filename);
        ASSERT_FALSE(image.empty()) << captured_arrow.filename;
        EXPECT_EQ(captured_arrow.expected_label,
                  traffic_light::classifyArrowDirection(
                      image, cv::Rect(0, 0, image.cols, image.rows)))
            << captured_arrow.filename;
    }
}

TEST(TrafficLightDecision, RejectsInvalidArrowInputs)
{
    EXPECT_EQ("unknown",
              traffic_light::classifyArrowDirection(cv::Mat(), cv::Rect()));

    const cv::Mat image(480, 640, CV_8UC3, cv::Scalar(25, 25, 25));
    EXPECT_EQ("unknown",
              traffic_light::classifyArrowDirection(image, cv::Rect(0, 0, 0, 10)));
}

TEST(TrafficLightDecision, UsesEightOfTenIndependentVotes)
{
    traffic_light::DirectionVoteWindow window(10, 8);
    for (int i = 0; i < 7; ++i)
    {
        window.add("left");
    }
    EXPECT_TRUE(window.winner().empty());

    window.add("left");
    EXPECT_EQ("left", window.winner());

    window.add("unknown");
    EXPECT_EQ(8, window.count("left"));
}

TEST(TrafficLightDecision, NormalStageBatchesModelAndCvVotes)
{
    traffic_light::DirectionDecisionAccumulator decision(10, 8, 30, 2);
    for (int frame = 0; frame < 3; ++frame)
    {
        EXPECT_TRUE(decision.processFrame("left", "left").empty());
    }
    EXPECT_EQ("left", decision.processFrame("left", "left"));
}

TEST(TrafficLightDecision, StraightModelVotesCanReachTheNormalThreshold)
{
    traffic_light::DirectionDecisionAccumulator decision(10, 8, 30, 2);
    for (int frame = 0; frame < 7; ++frame)
    {
        EXPECT_TRUE(decision.processFrame("straight", "unknown").empty());
    }
    EXPECT_EQ("straight", decision.processFrame("straight", "unknown"));
}

TEST(TrafficLightDecision, ConflictingDirectionsUseCvFallback)
{
    struct ConflictCase
    {
        const char *model_label;
        const char *cv_label;
    };
    const ConflictCase conflicts[] = {
        {"left", "right"},
        {"right", "left"}};

    for (const ConflictCase &conflict : conflicts)
    {
        traffic_light::DirectionDecisionAccumulator decision(8, 5, 5, 2);
        for (int frame = 0; frame < 4; ++frame)
        {
            EXPECT_TRUE(decision.processFrame(
                conflict.model_label, conflict.cv_label).empty());
        }
        EXPECT_EQ(0, decision.voteCountFor(conflict.model_label));
        EXPECT_EQ(0, decision.voteCountFor(conflict.cv_label));

        EXPECT_TRUE(decision.processFrame(
            conflict.model_label, conflict.cv_label).empty());
        EXPECT_TRUE(decision.inFallback());
        EXPECT_EQ(0u, decision.voteCount());

        EXPECT_TRUE(decision.processFrame(
            conflict.model_label, conflict.cv_label).empty());
        EXPECT_EQ(conflict.cv_label,
                  decision.processFrame(
                      conflict.model_label, conflict.cv_label));
    }
}

TEST(TrafficLightDecision, InvalidCvFrameBreaksStrictFallbackStreak)
{
    traffic_light::DirectionDecisionAccumulator decision(10, 8, 2, 2);
    EXPECT_TRUE(decision.processFrame("left", "right").empty());
    EXPECT_TRUE(decision.processFrame("left", "right").empty());
    ASSERT_TRUE(decision.inFallback());

    EXPECT_TRUE(decision.processFrame("left", "left").empty());
    EXPECT_TRUE(decision.processFrame("left", "unknown").empty());
    EXPECT_TRUE(decision.processFrame("left", "left").empty());
    EXPECT_EQ("left", decision.processFrame("left", "left"));
}

TEST(TrafficLightDecision, ResetDropsPreviousVotesAndFallbackState)
{
    traffic_light::DirectionDecisionAccumulator decision(10, 8, 30, 2);
    for (int frame = 0; frame < 3; ++frame)
    {
        EXPECT_TRUE(decision.processFrame("left", "left").empty());
    }
    EXPECT_EQ(6u, decision.voteCount());

    decision.reset();
    EXPECT_EQ(0u, decision.voteCount());
    EXPECT_EQ(0u, decision.processedFrameCount());
    EXPECT_FALSE(decision.inFallback());
    EXPECT_TRUE(decision.processFrame("right", "right").empty());
}
