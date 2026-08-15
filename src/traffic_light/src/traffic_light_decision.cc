#include "traffic_light_decision.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{

const int kReferenceWidth = 640;
const int kReferenceHeight = 480;
const int kCandidateMinWidth = 15;
const int kCandidateMaxWidth = 90;
const int kCandidateMinHeight = 12;
const int kCandidateMaxHeight = 90;
const int kCandidateMinArea = 80;
const int kCandidateMinColorScore = 200;
const double kCandidateMinFillDensity = 0.15;
const double kCandidateMinColorDensity = 0.10;
const int kCandidatePadding = 10;
const int kCloseKernelSize = 3;
const int kColorSupportKernelSize = 12;
const double kDensityEpsilon = 1e-9;

int scaledLength(int value, double scale)
{
    return std::max(1, static_cast<int>(std::round(value * scale)));
}

int scaledOddLength(int value, double scale)
{
    int scaled = std::max(3, static_cast<int>(std::round(value * scale)));
    if (scaled % 2 == 0)
    {
        ++scaled;
    }
    return scaled;
}

struct ArrowCandidate
{
    cv::Mat component_mask;
    int color_score = 0;
    int component_area = 0;
    double fill_density = 0.0;
    double selection_score = 0.0;
};

bool classifyComponent(const cv::Mat &component_mask, std::string &label)
{
    std::vector<cv::Point> points;
    cv::findNonZero(component_mask, points);
    if (points.size() < 2)
    {
        return false;
    }

    cv::Mat point_data(static_cast<int>(points.size()), 2, CV_64F);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        point_data.at<double>(static_cast<int>(i), 0) = points[i].x;
        point_data.at<double>(static_cast<int>(i), 1) = points[i].y;
    }

    const cv::PCA pca(point_data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    if (pca.eigenvectors.rows < 1 || pca.eigenvectors.cols < 2)
    {
        return false;
    }

    const double vx = pca.eigenvectors.at<double>(0, 0);
    const double vy = pca.eigenvectors.at<double>(0, 1);
    if (std::abs(vy) >= std::abs(vx))
    {
        return false;
    }

    const int width = component_mask.cols;
    const int height = component_mask.rows;
    double maximum_density = -1.0;
    int peak = -1;
    bool tied_peak = false;
    for (int band = 0; band < 8; ++band)
    {
        const int x1 = band * width / 8;
        const int x2 = (band + 1) * width / 8;
        if (x2 <= x1)
        {
            continue;
        }

        const cv::Mat band_mask = component_mask(cv::Rect(x1, 0, x2 - x1, height));
        const double density = static_cast<double>(cv::countNonZero(band_mask)) /
                               static_cast<double>(band_mask.total());
        if (density > maximum_density + kDensityEpsilon)
        {
            maximum_density = density;
            peak = band;
            tied_peak = false;
        }
        else if (std::abs(density - maximum_density) <= kDensityEpsilon)
        {
            tied_peak = true;
        }
    }

    if (peak < 0 || tied_peak)
    {
        return false;
    }

    label = peak <= 3 ? "left" : "right";
    return true;
}

} // namespace

namespace traffic_light
{

std::string classifyArrowDirection(const cv::Mat &bgr_frame,
                                  const cv::Rect &detection_box)
{
    if (bgr_frame.empty() || bgr_frame.type() != CV_8UC3 ||
        detection_box.width <= 0 || detection_box.height <= 0)
    {
        return "unknown";
    }

    const double scale = std::min(
        static_cast<double>(bgr_frame.cols) / kReferenceWidth,
        static_cast<double>(bgr_frame.rows) / kReferenceHeight);
    if (scale <= 0.0)
    {
        return "unknown";
    }

    const int padding = scaledLength(kCandidatePadding, scale);
    const cv::Rect expanded_box(
        detection_box.x - padding,
        detection_box.y - padding,
        detection_box.width + 2 * padding,
        detection_box.height + 2 * padding);
    const cv::Rect image_bounds(0, 0, bgr_frame.cols, bgr_frame.rows);
    const cv::Rect roi_rect = expanded_box & image_bounds;
    if (roi_rect.width <= 0 || roi_rect.height <= 0)
    {
        return "unknown";
    }

    const cv::Mat roi = bgr_frame(roi_rect);
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    cv::Mat green;
    cv::inRange(hsv,
                cv::Scalar(35, 80, 100),
                cv::Scalar(100, 255, 255),
                green);

    cv::Mat bright_raw;
    cv::inRange(hsv,
                cv::Scalar(0, 0, 212),
                cv::Scalar(179, 110, 255),
                bright_raw);

    const cv::Mat close_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(kCloseKernelSize, kCloseKernelSize));
    cv::Mat bright_closed;
    cv::morphologyEx(bright_raw,
                     bright_closed,
                     cv::MORPH_CLOSE,
                     close_kernel);

    const int support_kernel_size = scaledOddLength(kColorSupportKernelSize, scale);
    const cv::Mat support_kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(support_kernel_size, support_kernel_size));
    cv::Mat color_support;
    cv::dilate(green, color_support, support_kernel);

    cv::Mat bright;
    cv::bitwise_and(bright_closed, color_support, bright);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        bright, labels, stats, centroids, 8, CV_32S);
    if (component_count <= 1)
    {
        return "unknown";
    }

    const int min_width = kCandidateMinWidth;
    const int max_width = scaledLength(kCandidateMaxWidth, scale);
    const int min_height = kCandidateMinHeight;
    const int max_height = scaledLength(kCandidateMaxHeight, scale);
    const int min_area = kCandidateMinArea;
    const int min_color_score = kCandidateMinColorScore;
    const int max_padding = scaledLength(kCandidatePadding, scale);

    bool have_candidate = false;
    ArrowCandidate best_candidate;
    for (int component_id = 1; component_id < component_count; ++component_id)
    {
        const int x = stats.at<int>(component_id, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(component_id, cv::CC_STAT_TOP);
        const int width = stats.at<int>(component_id, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(component_id, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(component_id, cv::CC_STAT_AREA);
        if (width < min_width || width > max_width ||
            height < min_height || height > max_height || area < min_area)
        {
            continue;
        }

        const double fill_density = static_cast<double>(area) /
                                    static_cast<double>(width * height);
        if (fill_density < kCandidateMinFillDensity)
        {
            continue;
        }

        const int candidate_padding = std::min(
            max_padding,
            std::max(2, std::min(width, height) / 3));
        const int ex1 = std::max(0, x - candidate_padding);
        const int ey1 = std::max(0, y - candidate_padding);
        const int ex2 = std::min(bright.cols, x + width + candidate_padding);
        const int ey2 = std::min(bright.rows, y + height + candidate_padding);
        if (ex2 <= ex1 || ey2 <= ey1)
        {
            continue;
        }

        const int green_score = cv::countNonZero(
            green(cv::Rect(ex1, ey1, ex2 - ex1, ey2 - ey1)));
        if (green_score < min_color_score)
        {
            continue;
        }

        const int expanded_area = (ex2 - ex1) * (ey2 - ey1);
        const double color_density = static_cast<double>(green_score) /
                                     static_cast<double>(expanded_area);
        if (color_density < kCandidateMinColorDensity)
        {
            continue;
        }

        cv::Mat component_mask;
        cv::compare(labels(cv::Rect(x, y, width, height)),
                    component_id,
                    component_mask,
                    cv::CMP_EQ);

        ArrowCandidate candidate;
        candidate.component_mask = component_mask;
        candidate.color_score = green_score;
        candidate.component_area = area;
        candidate.fill_density = fill_density;
        candidate.selection_score = green_score * fill_density;

        if (!have_candidate ||
            candidate.selection_score > best_candidate.selection_score ||
            (candidate.selection_score == best_candidate.selection_score &&
             (candidate.color_score > best_candidate.color_score ||
              (candidate.color_score == best_candidate.color_score &&
               candidate.component_area > best_candidate.component_area))))
        {
            best_candidate = candidate;
            have_candidate = true;
        }
    }

    if (!have_candidate)
    {
        return "unknown";
    }

    std::string label;
    if (!classifyComponent(best_candidate.component_mask, label))
    {
        return "unknown";
    }
    return label;
}

DirectionVoteWindow::DirectionVoteWindow(std::size_t capacity,
                                         std::size_t threshold)
    : capacity_(capacity), threshold_(threshold)
{
}

void DirectionVoteWindow::reset()
{
    votes_.clear();
}

bool DirectionVoteWindow::isVoteLabel(const std::string &label) const
{
    return label == "straight" || label == "left" || label == "right";
}

void DirectionVoteWindow::add(const std::string &label)
{
    if (!isVoteLabel(label) || capacity_ == 0)
    {
        return;
    }
    if (votes_.size() >= capacity_)
    {
        votes_.pop_front();
    }
    votes_.push_back(label);
}

std::string DirectionVoteWindow::winner() const
{
    if (threshold_ == 0)
    {
        return std::string();
    }
    const char *labels[] = {"straight", "left", "right"};
    for (const char *label : labels)
    {
        if (count(label) >= static_cast<int>(threshold_))
        {
            return label;
        }
    }
    return std::string();
}

std::size_t DirectionVoteWindow::size() const
{
    return votes_.size();
}

int DirectionVoteWindow::count(const std::string &label) const
{
    return static_cast<int>(std::count(votes_.begin(), votes_.end(), label));
}

DirectionDecisionAccumulator::DirectionDecisionAccumulator(
    std::size_t vote_window_size,
    std::size_t vote_threshold,
    std::size_t fallback_frame_limit,
    std::size_t fallback_cv_streak)
    : vote_window_(vote_window_size, vote_threshold),
      fallback_frame_limit_(fallback_frame_limit),
      fallback_cv_streak_(fallback_cv_streak)
{
}

void DirectionDecisionAccumulator::reset()
{
    vote_window_.reset();
    started_ = false;
    fallback_ = false;
    processed_frame_count_ = 0;
    cv_candidate_.clear();
    cv_streak_ = 0;
}

bool DirectionDecisionAccumulator::isModelDirection(const std::string &label) const
{
    return label == "straight" || label == "left" || label == "right";
}

bool DirectionDecisionAccumulator::isCvDirection(const std::string &label) const
{
    return label == "left" || label == "right";
}

std::string DirectionDecisionAccumulator::processFrame(
    const std::string &model_label,
    const std::string &cv_label)
{
    if (!started_)
    {
        if (!isModelDirection(model_label))
        {
            return std::string();
        }
        started_ = true;
    }
    ++processed_frame_count_;

    if (fallback_)
    {
        if (isCvDirection(model_label) && isCvDirection(cv_label))
        {
            if (cv_label == cv_candidate_)
            {
                ++cv_streak_;
            }
            else
            {
                cv_candidate_ = cv_label;
                cv_streak_ = 1;
            }

            if (cv_streak_ >= fallback_cv_streak_ && fallback_cv_streak_ > 0)
            {
                return cv_candidate_;
            }
        }
        else
        {
            cv_candidate_.clear();
            cv_streak_ = 0;
        }
        return std::string();
    }

    vote_window_.add(model_label);
    if (isCvDirection(model_label) && isCvDirection(cv_label))
    {
        vote_window_.add(cv_label);
    }

    const std::string winner = vote_window_.winner();
    if (!winner.empty())
    {
        return winner;
    }

    if (processed_frame_count_ >= fallback_frame_limit_)
    {
        fallback_ = true;
        vote_window_.reset();
        cv_candidate_.clear();
        cv_streak_ = 0;
    }
    return std::string();
}

bool DirectionDecisionAccumulator::inFallback() const
{
    return fallback_;
}

std::size_t DirectionDecisionAccumulator::processedFrameCount() const
{
    return processed_frame_count_;
}

std::size_t DirectionDecisionAccumulator::voteCount() const
{
    return vote_window_.size();
}

int DirectionDecisionAccumulator::voteCountFor(const std::string &label) const
{
    return vote_window_.count(label);
}

const std::string &DirectionDecisionAccumulator::cvCandidate() const
{
    return cv_candidate_;
}

std::size_t DirectionDecisionAccumulator::cvStreak() const
{
    return cv_streak_;
}

} // namespace traffic_light
