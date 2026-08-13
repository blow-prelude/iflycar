#ifndef YOLOV8_MODEL_HPP
#define YOLOV8_MODEL_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"
#include "yolov8.h"

struct TrafficLightDetection
{
    cv::Rect box;
    float confidence;
    int class_id;
    std::string label;
};

struct YoloV8Result
{
    cv::Mat image;
    std::vector<TrafficLightDetection> detections;
};

class YoloV8Model
{
public:
    explicit YoloV8Model(const std::string &model_path);
    int init(rknn_context *shared_ctx, bool share_weight, int worker_id);
    rknn_context *get_pctx();
    YoloV8Result infer(const cv::Mat &bgr_frame);
    ~YoloV8Model();

private:
    std::string model_path_;
    int worker_id_;
    rknn_app_context_t app_ctx_;
};

#endif
