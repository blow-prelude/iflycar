#ifndef TRAFFIC_LIGHT_DECISION_HPP
#define TRAFFIC_LIGHT_DECISION_HPP

#include <cstddef>
#include <deque>
#include <string>

#include <opencv2/core.hpp>

namespace traffic_light
{

// Returns "left", "right", or "unknown" for the arrow inside detection_box.
std::string classifyArrowDirection(const cv::Mat &bgr_frame,
                                  const cv::Rect &detection_box);

class DirectionVoteWindow
{
public:
    DirectionVoteWindow(std::size_t capacity, std::size_t threshold);

    void reset();
    void add(const std::string &label);
    std::string winner() const;
    std::size_t size() const;
    int count(const std::string &label) const;

private:
    bool isVoteLabel(const std::string &label) const;

    std::size_t capacity_;
    std::size_t threshold_;
    std::deque<std::string> votes_;
};

class DirectionDecisionAccumulator
{
public:
    DirectionDecisionAccumulator(std::size_t vote_window_size,
                                 std::size_t vote_threshold,
                                 std::size_t fallback_frame_limit,
                                 std::size_t fallback_cv_streak);

    void reset();

    // model_label is the best model direction for this processed frame. The
    // CV label is meaningful only when model_label is left/right. The return
    // value is empty until a final direction is ready to publish.
    std::string processFrame(const std::string &model_label,
                             const std::string &cv_label);

    bool inFallback() const;
    std::size_t processedFrameCount() const;
    std::size_t voteCount() const;
    int voteCountFor(const std::string &label) const;
    const std::string &cvCandidate() const;
    std::size_t cvStreak() const;

private:
    bool isModelDirection(const std::string &label) const;
    bool isCvDirection(const std::string &label) const;

    DirectionVoteWindow vote_window_;
    std::size_t fallback_frame_limit_;
    std::size_t fallback_cv_streak_;
    bool started_ = false;
    bool fallback_ = false;
    std::size_t processed_frame_count_ = 0;
    std::string cv_candidate_;
    std::size_t cv_streak_ = 0;
};

} // namespace traffic_light

#endif
