#include <cstring>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "ppocr_model.hpp"

TEST(PPOCRCore, MapsWorkersToDedicatedCores)
{
    EXPECT_EQ(ppocr_core_mask_for_worker(0), RKNN_NPU_CORE_0);
    EXPECT_EQ(ppocr_core_mask_for_worker(1), RKNN_NPU_CORE_1);
    EXPECT_EQ(ppocr_core_mask_for_worker(2), RKNN_NPU_CORE_2);
    EXPECT_EQ(ppocr_core_mask_for_worker(-1), RKNN_NPU_CORE_UNDEFINED);
    EXPECT_EQ(ppocr_core_mask_for_worker(3), RKNN_NPU_CORE_UNDEFINED);
}

TEST(PPOCRVisualization, DrawsQuadrangleAndKeepsEmptyResultSafe)
{
    cv::Mat image = cv::Mat::zeros(120, 160, CV_8UC3);
    ppocr_text_recog_array_result_t results;
    std::memset(&results, 0, sizeof(results));
    results.count = 1;
    results.text_result[0].box.left_top = {20, 20};
    results.text_result[0].box.right_top = {120, 20};
    results.text_result[0].box.right_bottom = {120, 70};
    results.text_result[0].box.left_bottom = {20, 70};
    std::strcpy(results.text_result[0].text.str, "signal");
    results.text_result[0].text.score = 0.9f;

    draw_ppocr_results(image, results);
    EXPECT_GT(cv::countNonZero(image.reshape(1)), 0);
    EXPECT_EQ(image.size(), cv::Size(160, 120));
    EXPECT_EQ(image.type(), CV_8UC3);

    cv::Mat empty = cv::Mat::zeros(120, 160, CV_8UC3);
    ppocr_text_recog_array_result_t no_results;
    std::memset(&no_results, 0, sizeof(no_results));
    draw_ppocr_results(empty, no_results);
    EXPECT_EQ(cv::countNonZero(empty.reshape(1)), 0);
}
