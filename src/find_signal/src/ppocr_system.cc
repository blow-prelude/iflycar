#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <array>
#include <vector>

#include "opencv2/opencv.hpp"
#include "ppocr_system.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

bool CompareBox(const std::array<int, 8>& result1, const std::array<int, 8>& result2)
{
    if (result1[1] < result2[1])
    {
        return true;
    } else if (result1[1] == result2[1])
    {
        return result1[0] < result2[0];
    } else
    {
        return false;
    }
}

void SortBoxes(std::vector<std::array<int, 8>>* boxes)
{
    std::sort(boxes->begin(), boxes->end(), CompareBox);

    if (boxes->size() == 0)
    {
        return;
    }

    for (size_t i = 0; i + 1 < boxes->size(); i++)
    {
        for (size_t j = i + 1; j > 0; j--)
        {
            if (std::abs((*boxes)[j][1] - (*boxes)[j - 1][1]) < 10 &&
                (*boxes)[j][0] < (*boxes)[j - 1][0])
            {
                std::swap((*boxes)[j], (*boxes)[j - 1]);
            }
            else
            {
                break;
            }
        }
    }
}

cv::Mat GetRotateCropImage(const cv::Mat& srcimage, const std::array<int, 8>& box)
{
    if (srcimage.empty() || srcimage.cols < 2 || srcimage.rows < 2)
    {
        return cv::Mat();
    }

    const int max_x = srcimage.cols - 1;
    const int max_y = srcimage.rows - 1;
    std::array<cv::Point2f, 4> points;
    for (int i = 0; i < 4; ++i)
    {
        points[i].x = static_cast<float>(std::max(0, std::min(max_x, box[2 * i])));
        points[i].y = static_cast<float>(std::max(0, std::min(max_y, box[2 * i + 1])));
    }

    float left = points[0].x;
    float right = points[0].x;
    float top = points[0].y;
    float bottom = points[0].y;
    for (size_t i = 1; i < points.size(); ++i)
    {
        left = std::min(left, points[i].x);
        right = std::max(right, points[i].x);
        top = std::min(top, points[i].y);
        bottom = std::max(bottom, points[i].y);
    }

    const int crop_left = static_cast<int>(left);
    const int crop_top = static_cast<int>(top);
    const int crop_right = static_cast<int>(right);
    const int crop_bottom = static_cast<int>(bottom);
    if (crop_right <= crop_left || crop_bottom <= crop_top)
    {
        return cv::Mat();
    }

    cv::Rect crop_rect(crop_left, crop_top,
                       crop_right - crop_left + 1,
                       crop_bottom - crop_top + 1);
    cv::Mat img_crop = srcimage(crop_rect).clone();
    for (size_t i = 0; i < points.size(); ++i)
    {
        points[i].x -= static_cast<float>(crop_left);
        points[i].y -= static_cast<float>(crop_top);
    }

    const int img_crop_width = std::max(
        1, static_cast<int>(std::round(cv::norm(points[0] - points[1]))));
    const int img_crop_height = std::max(
        1, static_cast<int>(std::round(cv::norm(points[0] - points[3]))));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Mat M = cv::getPerspectiveTransform(points.data(), pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy;
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_ppocr_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    return init_ppocr_model_on_core(model_path, app_ctx, RKNN_NPU_CORE_AUTO);
}

int init_ppocr_model_on_core(const char* model_path, rknn_app_context_t* app_ctx,
                             rknn_core_mask core_mask)
{
    if (model_path == NULL || app_ctx == NULL)
    {
        printf("init_ppocr_model_on_core invalid argument\n");
        return -1;
    }

    memset(app_ctx, 0, sizeof(rknn_app_context_t));

    int model_len = 0;
    char* model = NULL;
    rknn_context ctx = 0;
    int ret = read_data_from_file(model_path, &model);
    if (ret < 0 || model == NULL)
    {
        printf("load_model fail! path=%s\n", model_path);
        return -1;
    }
    model_len = ret;

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d path=%s\n", ret, model_path);
        return -1;
    }

    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_set_core_mask fail! ret=%d core=%d path=%s\n",
               ret, static_cast<int>(core_mask), model_path);
        rknn_destroy(ctx);
        return -1;
    }

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC || io_num.n_input <= 0 || io_num.n_output <= 0)
    {
        printf("rknn_query in/out num fail! ret=%d path=%s\n", ret, model_path);
        rknn_destroy(ctx);
        return -1;
    }
    printf("model input num: %d, output num: %d, core=%d\n",
           io_num.n_input, io_num.n_output, static_cast<int>(core_mask));

    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < io_num.n_input; ++i)
    {
        memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR,
                         &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query input attr fail! ret=%d path=%s\n", ret, model_path);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    for (int i = 0; i < io_num.n_output; ++i)
    {
        memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR,
                         &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query output attr fail! ret=%d path=%s\n", ret, model_path);
            rknn_destroy(ctx);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    app_ctx->input_attrs = static_cast<rknn_tensor_attr*>(
        malloc(io_num.n_input * sizeof(rknn_tensor_attr)));
    app_ctx->output_attrs = static_cast<rknn_tensor_attr*>(
        malloc(io_num.n_output * sizeof(rknn_tensor_attr)));
    if (app_ctx->input_attrs == NULL || app_ctx->output_attrs == NULL)
    {
        printf("allocate tensor attributes fail! path=%s\n", model_path);
        release_ppocr_model(app_ctx);
        rknn_destroy(ctx);
        app_ctx->rknn_ctx = 0;
        return -1;
    }

    memcpy(app_ctx->input_attrs, input_attrs.data(),
           io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs.data(),
           io_num.n_output * sizeof(rknn_tensor_attr));
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
    return 0;
}

int release_ppocr_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_ppocr_det_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_det_result* out_result)
{
    if (app_ctx == NULL || src_img == NULL || params == NULL || out_result == NULL ||
        app_ctx->rknn_ctx == 0 || src_img->virt_addr == NULL ||
        src_img->width <= 0 || src_img->height <= 0)
    {
        return -1;
    }

    int ret = -1;
    image_buffer_t img;
    rknn_input inputs[1];
    rknn_output outputs[1];
    bool outputs_acquired = false;
    float scale_w = 0.0f;
    float scale_h = 0.0f;

    memset(&img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));
    memset(out_result, 0, sizeof(ppocr_det_result));

    // Pre Process
    img.width = app_ctx->model_width;
    img.height = app_ctx->model_height;
    img.format = IMAGE_FORMAT_RGB888;
    img.size = get_image_size(&img);
    img.virt_addr = (unsigned char*)malloc(img.size);
    if (img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", img.size);
        return -1;
    }

    ret = convert_image(src_img, &img, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image fail! ret=%d\n", ret);
        goto out;
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf   = img.virt_addr;

    scale_w = (float)src_img->width / (float)img.width;
    scale_h = (float)src_img->height / (float)img.height;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }
    outputs_acquired = true;

    // Post Process
    ret = dbnet_postprocess((float*)outputs[0].buf, app_ctx->model_width, app_ctx->model_height,
                                                params->threshold, params->box_threshold, params->use_dilate, params->db_score_mode,
                                                params->db_unclip_ratio, params->db_box_type,
                                                scale_w, scale_h, out_result);

out:
    if (outputs_acquired)
    {
        int release_ret = rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
        if (ret == 0 && release_ret < 0)
            ret = release_ret;
    }
    if (img.virt_addr != NULL) {
        free(img.virt_addr);
    }

    return ret;
}

int inference_ppocr_rec_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, ppocr_rec_result* out_result)
{
    if (app_ctx == NULL || src_img == NULL || out_result == NULL ||
        app_ctx->rknn_ctx == 0 || src_img->virt_addr == NULL ||
        src_img->width <= 0 || src_img->height <= 0)
    {
        return -1;
    }

    int ret = -1;
    rknn_input inputs[1];
    rknn_output outputs[1];
    bool outputs_acquired = false;
    int resized_w = 0;
    int out_len_seq = 0;
    const int imgW = app_ctx->model_width;
    const int imgH = app_ctx->model_height;

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));
    memset(out_result, 0, sizeof(ppocr_rec_result));
    inputs[0].buf = NULL;

    if (imgW <= 0 || imgH <= 0 || app_ctx->model_channel <= 0)
        return -1;

    // Pre Process
    float ratio = src_img->width / float(src_img->height);
    if (std::ceil(imgH*ratio) > imgW) {
        resized_w = imgW;
    }
    else {
        resized_w = std::ceil(imgH*ratio);
    }

    cv::Mat img_M = cv::Mat(src_img->height, src_img->width, CV_8UC3,(uint8_t*)src_img->virt_addr);
    cv::resize(img_M, img_M, cv::Size(resized_w, imgH));
    img_M.convertTo(img_M, CV_32FC3);
    img_M = (img_M - 127.5)/127.5;
    if (resized_w < imgW) {
        copyMakeBorder(img_M, img_M, 0, 0, 0, imgW- resized_w, cv::BORDER_CONSTANT, 0);
    }

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel * sizeof(float);
    inputs[0].buf = malloc(inputs[0].size);
    if (inputs[0].buf == NULL)
    {
        printf("malloc recognition input size:%d fail!\n", inputs[0].size);
        goto out;
    }
    memcpy(inputs[0].buf, img_M.data, inputs[0].size);

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        goto out;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Get Output
    out_len_seq = app_ctx->model_width / 8;
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }
    outputs_acquired = true;

    // Post Process
    ret = rec_postprocess((float*)outputs[0].buf, MODEL_OUT_CHANNEL, out_len_seq, out_result);

out:
    if (outputs_acquired)
    {
        int release_ret = rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
        if (ret == 0 && release_ret < 0)
            ret = release_ret;
    }
    if (inputs[0].buf != NULL) {
        free(inputs[0].buf);
    }

    return ret;
}

int inference_ppocr_system_model(ppocr_system_app_context* sys_app_ctx, image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_text_recog_array_result_t* out_result)
{
    if (sys_app_ctx == NULL || src_img == NULL || params == NULL || out_result == NULL)
        return -1;

    int ret = -1;
    memset(out_result, 0, sizeof(ppocr_text_recog_array_result_t));

    // Detect Text
    ppocr_det_result det_results;
    memset(&det_results, 0, sizeof(det_results));
    ret = inference_ppocr_det_model(&sys_app_ctx->det_context, src_img, params, &det_results);
    if (ret != 0) {
        printf("inference_ppocr_det_model fail! ret=%d\n", ret);
        return -1;
    }

    // Recogize Text
    if (det_results.count == 0) {           // detect nothing
        return 0;
    }

    // boxes to boxes_result
    std::vector<std::array<int, 8>> boxes_result;
    for (int i=0; i < det_results.count; i++) {
        std::array<int, 8> new_box;
        new_box[0] = det_results.box[i].left_top.x;
        new_box[1] = det_results.box[i].left_top.y;
        new_box[2] = det_results.box[i].right_top.x;
        new_box[3] = det_results.box[i].right_top.y;
        new_box[4] = det_results.box[i].right_bottom.x;
        new_box[5] = det_results.box[i].right_bottom.y;
        new_box[6] = det_results.box[i].left_bottom.x;
        new_box[7] = det_results.box[i].left_bottom.y;
        boxes_result.emplace_back(new_box);
    }

    // Sort text boxes in order from top to bottom, left to right for speeding up
    SortBoxes(&boxes_result);

    // text recognize
    for (size_t i = 0; i < boxes_result.size(); i++) {
        cv::Mat in_image = cv::Mat(src_img->height, src_img->width, CV_8UC3,(uint8_t*)src_img->virt_addr);
        cv::Mat crop_image = GetRotateCropImage(in_image, boxes_result[i]);
        if (crop_image.empty())
        {
            continue;
        }

        image_buffer_t text_img;
        memset(&text_img, 0, sizeof(image_buffer_t));
        text_img.width = crop_image.cols;
        text_img.height = crop_image.rows;
        text_img.format = IMAGE_FORMAT_RGB888;
        text_img.size = get_image_size(&text_img);
        text_img.virt_addr = (unsigned char*)malloc(text_img.size);
        if (text_img.virt_addr == NULL) {
            printf("malloc buffer size:%d fail!\n", text_img.size);
            return -1;
        }
        memcpy((void *)text_img.virt_addr, crop_image.data, text_img.size);

        ppocr_rec_result text_result;
        text_result.score = 1.0;
        ret = inference_ppocr_rec_model(&sys_app_ctx->rec_context, &text_img, &text_result);
        if (ret != 0) {
            printf("inference_ppocr_rec_model fail! ret=%d\n", ret);
            free(text_img.virt_addr);
            return -1;
        }
        if (text_img.virt_addr != NULL) {
            free(text_img.virt_addr);
        }

        if (text_result.score < TEXT_SCORE) {
            continue;
        }
        if (out_result->count >= 1000)
            break;

        out_result->text_result[out_result->count].box.left_top.x = boxes_result[i][0];
        out_result->text_result[out_result->count].box.left_top.y = boxes_result[i][1];
        out_result->text_result[out_result->count].box.right_top.x = boxes_result[i][2];
        out_result->text_result[out_result->count].box.right_top.y = boxes_result[i][3];
        out_result->text_result[out_result->count].box.right_bottom.x = boxes_result[i][4];
        out_result->text_result[out_result->count].box.right_bottom.y = boxes_result[i][5];
        out_result->text_result[out_result->count].box.left_bottom.x = boxes_result[i][6];
        out_result->text_result[out_result->count].box.left_bottom.y = boxes_result[i][7];
        out_result->text_result[out_result->count].text = text_result;
        out_result->count ++;
    }

    return ret;
}
