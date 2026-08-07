#include "yolov8_model.hpp"

#include <stdio.h>

#include <opencv2/imgproc.hpp>

#include "image_utils.h"

namespace
{
const char *kTrafficLightLabels[] = {"stop", "straight", "right", "left"};
const int kTrafficLightLabelCount =
    sizeof(kTrafficLightLabels) / sizeof(kTrafficLightLabels[0]);
}

YoloV8Model::YoloV8Model(const std::string &model_path)
    : model_path_(model_path), worker_id_(-1)
{
    memset(&app_ctx_, 0, sizeof(rknn_app_context_t));
}

int YoloV8Model::init(rknn_context *shared_ctx, bool /*share_weight*/, int worker_id)
{
    worker_id_ = worker_id;
    rknn_core_mask core_mask = yolov8_core_mask_for_worker(worker_id);
    int ret = init_yolov8_model_on_core(model_path_.c_str(), &app_ctx_, shared_ctx, core_mask);
    if (ret != 0)
    {
        printf("worker %d init_yolov8_model_on_core fail ret=%d\n", worker_id_, ret);
        return ret;
    }
    printf("worker %d bound to core mask %d\n", worker_id_, (int)core_mask);
    return 0;
}

rknn_context *YoloV8Model::get_pctx()
{
    if (app_ctx_.rknn_ctx == 0)
    {
        return nullptr;
    }
    return &app_ctx_.rknn_ctx;
}

cv::Mat YoloV8Model::infer(const cv::Mat &bgr_frame)
{
    cv::Mat output = bgr_frame.clone();
    if (output.empty())
    {
        return output;
    }

    cv::Mat rgb;
    cv::cvtColor(output, rgb, cv::COLOR_BGR2RGB);

    image_buffer_t img;
    memset(&img, 0, sizeof(image_buffer_t));
    img.width = rgb.cols;
    img.height = rgb.rows;
    img.format = IMAGE_FORMAT_RGB888;
    img.virt_addr = rgb.data;
    img.size = rgb.cols * rgb.rows * 3;

    object_detect_result_list results;
    int ret = inference_yolov8_model(&app_ctx_, &img, &results);
    if (ret != 0)
    {
        printf("worker %d inference_yolov8_model fail ret=%d\n", worker_id_, ret);
        return output;
    }

    for (int i = 0; i < results.count; i++)
    {
        const object_detect_result &det = results.results[i];
        cv::rectangle(output,
                      cv::Point(det.box.left, det.box.top),
                      cv::Point(det.box.right, det.box.bottom),
                      cv::Scalar(0, 255, 0), 2);

        char text[OBJ_NAME_MAX_SIZE];
        if (det.cls_id >= 0 && det.cls_id < kTrafficLightLabelCount)
        {
            snprintf(text, sizeof(text), "%s@%.3f",
                     kTrafficLightLabels[det.cls_id], det.prop);
        }
        else
        {
            snprintf(text, sizeof(text), "class_%d@%.3f", det.cls_id, det.prop);
        }
        cv::putText(output, text, cv::Point(det.box.left, std::max(0, det.box.top - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    return output;
}

YoloV8Model::~YoloV8Model()
{
    release_yolov8_model(&app_ctx_);
}
