#include "ppocr_model.hpp"

#include <algorithm>
#include <cstdio>

namespace
{
cv::Point clamp_point(const rknn_point_t &point, const cv::Size &size)
{
    return cv::Point(std::max(0, std::min(size.width - 1, point.x)),
                     std::max(0, std::min(size.height - 1, point.y)));
}

bool is_ascii_text(const char *text)
{
    if (text == NULL || text[0] == '\0')
        return false;

    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(text);
         *p != '\0'; ++p)
    {
        if (*p >= 0x80)
            return false;
    }
    return true;
}
} // namespace

void draw_ppocr_results(cv::Mat &bgr_image,
                        const ppocr_text_recog_array_result_t &results)
{
    if (bgr_image.empty() || bgr_image.channels() != 3)
        return;

    const int result_count = std::max(0, std::min(results.count, 1000));
    for (int i = 0; i < result_count; ++i)
    {
        const ppocr_text_recog_result_t &result = results.text_result[i];
        std::vector<cv::Point> polygon;
        polygon.push_back(clamp_point(result.box.left_top, bgr_image.size()));
        polygon.push_back(clamp_point(result.box.right_top, bgr_image.size()));
        polygon.push_back(clamp_point(result.box.right_bottom, bgr_image.size()));
        polygon.push_back(clamp_point(result.box.left_bottom, bgr_image.size()));

        cv::polylines(bgr_image, polygon, true, cv::Scalar(0, 255, 0), 2,
                      cv::LINE_AA);

        const int text_x = std::min(polygon[0].x, polygon[3].x);
        const int text_y = std::max(15, std::min(polygon[0].y, polygon[1].y) - 5);
        char label[sizeof(result.text.str) + 32];
        if (is_ascii_text(result.text.str))
        {
            std::snprintf(label, sizeof(label), "%s@%.3f",
                          result.text.str, result.text.score);
        }
        else
        {
            std::snprintf(label, sizeof(label), "score@%.3f",
                          result.text.score);
        }
        cv::putText(bgr_image, label, cv::Point(text_x, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1,
                    cv::LINE_AA);
    }
}
