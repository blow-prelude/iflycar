#ifndef FIND_SIGNAL_PPOCR_MODEL_HPP
#define FIND_SIGNAL_PPOCR_MODEL_HPP

#include <string>

#include <opencv2/opencv.hpp>

#include "ppocr_system.h"

class PPOCRModel
{
public:
    PPOCRModel(const std::string &det_model_path,
               const std::string &rec_model_path);

    int init(rknn_context *unused_shared_ctx,
             bool unused_share_weight,
             int worker_id);
    rknn_context *get_pctx();
    cv::Mat infer(const cv::Mat &bgr_frame);

    ~PPOCRModel();

private:
    std::string det_model_path_;
    std::string rec_model_path_;
    int worker_id_;
    bool det_initialized_;
    bool rec_initialized_;
    ppocr_system_app_context app_ctx_;
};

void draw_ppocr_results(cv::Mat &bgr_image,
                        const ppocr_text_recog_array_result_t &results);

#endif // FIND_SIGNAL_PPOCR_MODEL_HPP
