#include <array>

#include <gtest/gtest.h>

#include "yolov8.h"
#include "postprocess.h"

TEST(YoloV8Postprocess, ReadsCustomClassCountFromScoreChannels)
{
    rknn_tensor_attr score_attr{};
    score_attr.n_dims = 4;
    score_attr.dims[0] = 1;
    score_attr.dims[1] = 4;
    score_attr.dims[2] = 80;
    score_attr.dims[3] = 80;

    EXPECT_EQ(get_yolov8_class_count(&score_attr), 4);
}

TEST(YoloV8Postprocess, KeepsDflSoftmaxFiniteForLargeLogits)
{
    rknn_app_context_t app_ctx{};
    app_ctx.io_num.n_output = 6;
    app_ctx.model_width = 640;
    app_ctx.model_height = 640;
    app_ctx.is_quant = false;

    rknn_tensor_attr output_attrs[6]{};
    for (int branch = 0; branch < 3; ++branch)
    {
        rknn_tensor_attr &box_attr = output_attrs[branch * 2];
        box_attr.n_dims = 4;
        box_attr.dims[0] = 1;
        box_attr.dims[1] = 64;
        box_attr.dims[2] = 1;
        box_attr.dims[3] = 1;

        rknn_tensor_attr &score_attr = output_attrs[branch * 2 + 1];
        score_attr.n_dims = 4;
        score_attr.dims[0] = 1;
        score_attr.dims[1] = 4;
        score_attr.dims[2] = 1;
        score_attr.dims[3] = 1;
    }
    app_ctx.output_attrs = output_attrs;

    std::array<float, 64> box_logits;
    box_logits.fill(1000.0f);
    std::array<float, 4> detected_scores{{0.9f, 0.0f, 0.0f, 0.0f}};
    std::array<float, 4> empty_scores{{0.0f, 0.0f, 0.0f, 0.0f}};

    rknn_output outputs[6]{};
    for (int branch = 0; branch < 3; ++branch)
    {
        outputs[branch * 2].buf = box_logits.data();
        outputs[branch * 2 + 1].buf =
            branch == 0 ? detected_scores.data() : empty_scores.data();
    }

    letterbox_t letter_box{};
    letter_box.scale = 1.0f;
    object_detect_result_list results{};

    ASSERT_EQ(post_process(&app_ctx, outputs, &letter_box,
                           BOX_THRESH, NMS_THRESH, &results),
              0);
    ASSERT_EQ(results.count, 1);
    EXPECT_EQ(results.results[0].box.left, 0);
    EXPECT_EQ(results.results[0].box.top, 0);
    EXPECT_EQ(results.results[0].box.right, 640);
    EXPECT_EQ(results.results[0].box.bottom, 640);
    EXPECT_FLOAT_EQ(results.results[0].prop, 0.9f);
}
