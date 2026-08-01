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
