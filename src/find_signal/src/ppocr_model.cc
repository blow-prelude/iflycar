#include "ppocr_model.hpp"

#include <cstdio>
#include <cstring>

PPOCRModel::PPOCRModel(const std::string &det_model_path,
                       const std::string &rec_model_path)
    : det_model_path_(det_model_path),
      rec_model_path_(rec_model_path),
      worker_id_(-1),
      det_initialized_(false),
      rec_initialized_(false)
{
    std::memset(&app_ctx_, 0, sizeof(app_ctx_));
}

int PPOCRModel::init(rknn_context *unused_shared_ctx,
                     bool unused_share_weight,
                     int worker_id)
{
    (void)unused_shared_ctx;
    (void)unused_share_weight;

    worker_id_ = worker_id;
    const rknn_core_mask core_mask = ppocr_core_mask_for_worker(worker_id_);
    if (core_mask == RKNN_NPU_CORE_UNDEFINED)
    {
        std::fprintf(stderr, "worker %d has no supported NPU core\n", worker_id_);
        return -1;
    }

    int ret = init_ppocr_model_on_core(det_model_path_.c_str(),
                                       &app_ctx_.det_context, core_mask);
    if (ret != 0)
    {
        std::fprintf(stderr, "worker %d det model init failed ret=%d path=%s\n",
                     worker_id_, ret, det_model_path_.c_str());
        return ret;
    }
    det_initialized_ = true;

    ret = init_ppocr_model_on_core(rec_model_path_.c_str(),
                                   &app_ctx_.rec_context, core_mask);
    if (ret != 0)
    {
        std::fprintf(stderr, "worker %d rec model init failed ret=%d path=%s\n",
                     worker_id_, ret, rec_model_path_.c_str());
        release_ppocr_model(&app_ctx_.det_context);
        det_initialized_ = false;
        return ret;
    }
    rec_initialized_ = true;

    std::printf("worker %d initialized PPOCR det/rec on core mask %d\n",
                worker_id_, static_cast<int>(core_mask));
    return 0;
}

rknn_context *PPOCRModel::get_pctx()
{
    // PPOCR owns two independent contexts. The local pool passes this value
    // for compatibility with the traffic_light pool contract, but this model
    // intentionally does not duplicate/share either context.
    return NULL;
}

cv::Mat PPOCRModel::infer(const cv::Mat &bgr_frame)
{
    cv::Mat output = bgr_frame.clone();
    if (output.empty() || !det_initialized_ || !rec_initialized_)
        return output;

    if (output.type() != CV_8UC3)
    {
        std::fprintf(stderr, "worker %d received unsupported frame type=%d\n",
                     worker_id_, output.type());
        return output;
    }

    cv::Mat rgb_frame;
    cv::cvtColor(output, rgb_frame, cv::COLOR_BGR2RGB);
    if (!rgb_frame.isContinuous())
        rgb_frame = rgb_frame.clone();

    image_buffer_t src_image;
    std::memset(&src_image, 0, sizeof(src_image));
    src_image.width = rgb_frame.cols;
    src_image.height = rgb_frame.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.virt_addr = rgb_frame.data;
    src_image.size = static_cast<int>(rgb_frame.total() * rgb_frame.elemSize());

    ppocr_det_postprocess_params params;
    params.threshold = 0.3f;
    params.box_threshold = 0.6f;
    params.use_dilate = false;
    params.db_score_mode = "slow";
    params.db_box_type = "poly";
    params.db_unclip_ratio = 1.5f;

    ppocr_text_recog_array_result_t results;
    std::memset(&results, 0, sizeof(results));
    const int ret = inference_ppocr_system_model(&app_ctx_, &src_image,
                                                 &params, &results);
    if (ret != 0)
    {
        std::fprintf(stderr,
                     "worker %d inference_ppocr_system_model failed ret=%d\n",
                     worker_id_, ret);
        return output;
    }

    for (int i = 0; i < results.count && i < 1000; ++i)
    {
        std::printf("worker %d result[%d] text=%s score=%.3f\n",
                    worker_id_, i, results.text_result[i].text.str,
                    results.text_result[i].text.score);
    }
    draw_ppocr_results(output, results);
    return output;
}

PPOCRModel::~PPOCRModel()
{
    if (rec_initialized_)
    {
        release_ppocr_model(&app_ctx_.rec_context);
        rec_initialized_ = false;
    }
    if (det_initialized_)
    {
        release_ppocr_model(&app_ctx_.det_context);
        det_initialized_ = false;
    }
}
