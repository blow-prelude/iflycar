// scan_side_lines 横向白线遮挡过滤的回归测试。
//
// 图像约定与 preprocess 输出一致：0 表示赛道内部（黑），非 0 表示白色特征。
// 断言方式：通过 draw_line 把补线结果（supple_*，红色圆点）绘制到画布，
// 再逐行提取红色像素的极值列，重建每行左右边线位置；中线偏差用
// get_fit_mid_line 与无横线基线对比。
#include "image_process.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <map>
#include <vector>

namespace
{
// 画布尺寸：宽 160、高 120，小于默认 process_max(320x240)，验证允许。
const int kImgW = 160;
const int kImgH = 120;
// 默认配置下：mid_x=80，未稳定搜索起点为 80∓search_offset(30) = 50/110。
// task2 扫描区间 y∈[60,108]，task1 扫描区间 y∈[78,105]。
const int kLeftEdgeX = 39;  // 左边线所在列（白色条纹 [37,39] 的右缘）
const int kRightEdgeX = 119; // 右边线所在列（白色条纹 [120,122] 的左缘外一点）

// 绘制 3 像素宽的竖直边线条纹，扫描命中点为 edge_x。
void draw_edge_stripe(cv::Mat &binary, int y, int edge_x)
{
    for (int x = edge_x - 2; x <= edge_x; ++x)
    {
        binary.at<uchar>(y, x) = 255;
    }
}

// 生成全黑二值图。
cv::Mat make_black_binary()
{
    return cv::Mat::zeros(kImgH, kImgW, CV_8UC1);
}

// 在 [y0,y1] 行、[x0,x1] 列绘制横向白条。
void draw_horizontal_bar(cv::Mat &binary, int y0, int y1, int x0, int x1)
{
    for (int y = y0; y <= y1; ++y)
    {
        for (int x = x0; x <= x1; ++x)
        {
            binary.at<uchar>(y, x) = 255;
        }
    }
}

// 跑一次 task2 边线搜索 + 补线 + 拟合，并绘制到画布，返回画布。
cv::Mat run_task2_and_draw(ImageProcess &process, const cv::Mat &binary,
                           bool find_corner = false, SearchSide side = BOTH)
{
    cv::Mat canvas(kImgH, kImgW, CV_8UC3, cv::Scalar(0, 0, 0));
    process.set_mid_line_mode(MID_AVG);
    process.get_side_line_task_2(binary, canvas, true, find_corner, side);
    process.calculate_mid_line(binary.size());
    process.fit_polynomial();
    process.draw_line(canvas, -1.0f, "", TrackingTarget::STRAIGHT);
    return canvas;
}

// 行 y 上 [col_from,col_to] 区间内最左侧纯红像素列（supple 边线圆点），无则 -1。
int min_red_col(const cv::Mat &canvas, int y, int col_from, int col_to)
{
    for (int x = col_from; x <= col_to; ++x)
    {
        if (canvas.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255))
        {
            return x;
        }
    }
    return -1;
}

// 行 y 上 [col_from,col_to] 区间内最右侧纯红像素列，无则 -1。
int max_red_col(const cv::Mat &canvas, int y, int col_from, int col_to)
{
    for (int x = col_to; x >= col_from; --x)
    {
        if (canvas.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255))
        {
            return x;
        }
    }
    return -1;
}

// 圆点半径为 2：左边线命中 kLeftEdgeX 时最左红列应为 kLeftEdgeX-2。
int left_line_min_col(const cv::Mat &canvas, int y)
{
    return min_red_col(canvas, y, 0, 69);
}

// 右边线命中 kRightEdgeX 时最右红列应为 kRightEdgeX+2。
int right_line_max_col(const cv::Mat &canvas, int y)
{
    return max_red_col(canvas, y, 101, kImgW - 1);
}
} // namespace

// 计划测试 1 + 3：稳定跟踪阶段出现覆盖两侧搜索起点的横条。
// 期望：横条端点不进入边线（左端点 42 距真实边线 39 仅 3 像素，若无遮挡
// 判断会被连续性检查接受）；空洞由插值补齐；横条后从原锚点续接真实边线；
// 拟合中线与无横条基线一致。
TEST(ScanSideLinesHorizontalBar, StableBarKeepsTrueEdgesAndMidLine)
{
    const int bar_y0 = 86, bar_y1 = 95; // 横条行，位于稳定跟踪阶段

    ImageProcessConfig config;
    ImageProcess baseline(config);
    ImageProcess with_bar(config);

    cv::Mat clean = make_black_binary();
    for (int y = 60; y <= 108; ++y)
    {
        draw_edge_stripe(clean, y, kLeftEdgeX);
        draw_edge_stripe(clean, y, kRightEdgeX + 3); // 条纹 [120,122]
    }

    cv::Mat barred = clean.clone();
    draw_horizontal_bar(barred, bar_y0, bar_y1, 42, 100);

    const cv::Mat canvas_base = run_task2_and_draw(baseline, clean);
    const cv::Mat canvas_bar = run_task2_and_draw(with_bar, barred);

    // 每一行（含横条行）若存在左边线点，必须位于真实边线 39 上；
    // 无遮挡判断时横条行左端点 42 会让最左红列变成 40。
    for (int y = 62; y <= 106; ++y)
    {
        const int left = left_line_min_col(canvas_bar, y);
        if (left >= 0)
        {
            EXPECT_NEAR(kLeftEdgeX - 2, left, 1) << "row " << y;
        }
        const int right = right_line_max_col(canvas_bar, y);
        if (right >= 0)
        {
            EXPECT_NEAR(kRightEdgeX + 2, right, 1) << "row " << y;
        }
    }
    // 横条空洞由插值补齐（横条行中存在补线点）。
    EXPECT_LE(0, left_line_min_col(canvas_bar, 89));
    EXPECT_LE(0, left_line_min_col(canvas_bar, 92));
    // 横条之后从原边线续接（横条上沿之外仍有真实边线点）。
    EXPECT_NEAR(kLeftEdgeX - 2, left_line_min_col(canvas_bar, 70), 1);
    EXPECT_NEAR(kRightEdgeX + 2, right_line_max_col(canvas_bar, 70), 1);

    // 拟合中线与基线同跨度且逐行偏差很小（不向横条端点偏移）。
    // 横条行的空洞使两次拟合的 y 序列不同，按 y 建索引对比。
    const std::vector<cv::Point> &fit_base = baseline.get_fit_mid_line();
    const std::vector<cv::Point> &fit_bar = with_bar.get_fit_mid_line();
    ASSERT_FALSE(fit_base.empty());
    ASSERT_FALSE(fit_bar.empty());
    ASSERT_EQ(fit_base.front().y, fit_bar.front().y);
    ASSERT_EQ(fit_base.back().y, fit_bar.back().y);
    std::map<int, int> base_x_by_y;
    for (const cv::Point &p : fit_base)
    {
        base_x_by_y[p.y] = p.x;
    }
    for (const cv::Point &p : fit_bar)
    {
        const std::map<int, int>::const_iterator it = base_x_by_y.find(p.y);
        ASSERT_TRUE(it != base_x_by_y.end()) << "row " << p.y;
        EXPECT_NEAR(it->second, p.x, 2) << "row " << p.y;
    }
}

// 计划测试 2：横条覆盖扫描起始行（初始未稳定阶段）。
// 期望：稳定点缓冲被清空，横条端点（左 44、右 115）不能形成边线；
// 横条结束后从真实边线重新初始化。
TEST(ScanSideLinesHorizontalBar, InitBarDoesNotCreateSideLineFromEndpoints)
{
    ImageProcessConfig config;
    ImageProcess process(config);

    cv::Mat binary = make_black_binary();
    for (int y = 60; y <= 108; ++y)
    {
        draw_edge_stripe(binary, y, kLeftEdgeX);
        draw_edge_stripe(binary, y, kRightEdgeX + 3);
    }
    // 横条覆盖扫描最底部 9 行，且覆盖两侧未稳定搜索起点 50/110。
    draw_horizontal_bar(binary, 100, 108, 45, 115);

    const cv::Mat canvas = run_task2_and_draw(process, binary);

    // 遮挡行不产生任何边线点：左右区域都没有红色像素。
    // （100/101 行可能带有 99 行圆点的弧线像素，从 102 行开始断言。）
    for (int y = 102; y <= 107; ++y)
    {
        EXPECT_EQ(-1, left_line_min_col(canvas, y)) << "row " << y;
        EXPECT_EQ(-1, right_line_max_col(canvas, y)) << "row " << y;
    }
    // 横条之上从真实边线重新稳定。
    const int sample_rows[] = {63, 75, 90, 97};
    for (const int y : sample_rows)
    {
        EXPECT_NEAR(kLeftEdgeX - 2, left_line_min_col(canvas, y), 1) << "row " << y;
        EXPECT_NEAR(kRightEdgeX + 2, right_line_max_col(canvas, y), 1) << "row " << y;
    }
}

// 计划测试 4（遮挡不计 miss 的保护效果）：稳定跟踪阶段出现长横条（10 行，
// 超过 miss_threshold 的两倍），横条左端 45 距真实左边线 45 仅 1 像素，且
// 同时覆盖稳定搜索起点(95)与未稳定搜索起点(50)。
// 期望：遮挡行整行跳过后，左边线止于横条下沿，横线端点不进入边线。
// 若无遮挡判断：横条行候选端点 44 会被连续性检查接受并形成一串误检点
// （横条行及其上方出现 x≈44 的红点），重置后的未稳定搜索也会打进横条
// 在端点上重新稳定。
TEST(ScanSideLinesHorizontalBar, OccludedRowsSkipSearchAndKeepEndpointsOut)
{
    ImageProcessConfig config;
    ImageProcess process(config);

    cv::Mat binary = make_black_binary();
    for (int y = 60; y <= 108; ++y)
    {
        draw_edge_stripe(binary, y, kRightEdgeX + 3);
    }
    for (int y = 96; y <= 108; ++y)
    {
        draw_edge_stripe(binary, y, 45); // 横条前左边线在 45
    }
    for (int y = 60; y <= 85; ++y)
    {
        draw_edge_stripe(binary, y, 54); // 横条后左边线移到 54
    }
    // 10 行横条 > 2 * miss_threshold(3)，x 覆盖 [45,100]：
    // 同时盖住稳定搜索起点 45+50=95 与未稳定搜索起点 50。
    draw_horizontal_bar(binary, 86, 95, 45, 100);

    const cv::Mat canvas = run_task2_and_draw(process, binary);

    // 横条之前左边线保持在 45。
    const int below_rows[] = {98, 103, 107};
    for (const int y : below_rows)
    {
        EXPECT_NEAR(45 - 2, left_line_min_col(canvas, y), 1) << "row " << y;
    }
    // 横条行及其上方一段（87..93，避开 96 行圆点的弧线）不允许出现任何
    // 左边线点；无遮挡判断时这里会被端点 44 的误检点占据。
    for (int y = 87; y <= 93; ++y)
    {
        EXPECT_EQ(-1, left_line_min_col(canvas, y)) << "row " << y;
    }
    // 右边线在横条之后经普通丢线重置从未稳定起点重新锁定，全程正常。
    const int right_rows[] = {62, 70, 84, 100, 107};
    for (const int y : right_rows)
    {
        EXPECT_NEAR(kRightEdgeX + 2, right_line_max_col(canvas, y), 1) << "row " << y;
    }
}

// 计划测试 3（锚点续接）：横条很短（3 行 < y_continual=5），横条后第一个
// 候选与边线末点的行距 Δy=4 仍满足连续性。
// 期望：遮挡行不累计 miss、不重置，横条后直接从原锚点续接真实边线 39，
// 中间空洞由插值补齐；无遮挡判断时横条行会被端点 42 占据。
TEST(ScanSideLinesHorizontalBar, ShortBarResumesFromPreservedAnchor)
{
    ImageProcessConfig config;
    ImageProcess process(config);

    cv::Mat binary = make_black_binary();
    for (int y = 60; y <= 108; ++y)
    {
        draw_edge_stripe(binary, y, kLeftEdgeX);
        draw_edge_stripe(binary, y, kRightEdgeX + 3);
    }
    // 3 行横条：横条后第一候选(93 行)与末点(97 行)行距 4 < y_continual(5)。
    draw_horizontal_bar(binary, 94, 96, 42, 100);

    const cv::Mat canvas = run_task2_and_draw(process, binary);

    for (int y = 62; y <= 106; ++y)
    {
        const int left = left_line_min_col(canvas, y);
        if (left < 0)
        {
            continue;
        }
        if (y >= 94 && y <= 96)
        {
            // 横条行只允许插值空洞两侧的圆点弦像素（区间宽 ≤ 3 列）。
            // 无遮挡判断时横条行会被端点 41 的误检点占据（区间 [39,43]）。
            const int right = max_red_col(canvas, y, 0, 69);
            EXPECT_GE(left, kLeftEdgeX - 3) << "row " << y;
            EXPECT_LE(right, kLeftEdgeX + 2) << "row " << y;
        }
        else
        {
            EXPECT_NEAR(kLeftEdgeX - 2, left, 1) << "row " << y;
        }
    }
    // 横条后紧邻一行即从锚点续接真实边线（未发生重置，93 行有点）。
    EXPECT_NEAR(kLeftEdgeX - 2, left_line_min_col(canvas, 93), 1);
    EXPECT_NEAR(kRightEdgeX + 2, right_line_max_col(canvas, 93), 1);
}

// 计划测试 4（普通丢线仍按 miss_threshold 累计和重置）：稳定跟踪阶段左边线
// 突然跳到远离上一行的位置（40 像素 > x_continual），中间若干行无左边线。
// 期望：连续被拒后 miss 超过阈值触发重置；重置后 x=5 的候选与上一行边线
// 末点（45）的 x 差距仍很大，不进入边线列表而是逐行计入 miss——左边线最终
// 只保留丢线前的 45 段，空洞上方不再出现跳变边线点。
TEST(ScanSideLinesHorizontalBar, NormalMissStillResetsAndRejectsFarEdge)
{
    ImageProcessConfig config;
    ImageProcess process(config);

    cv::Mat binary = make_black_binary();
    for (int y = 60; y <= 108; ++y)
    {
        draw_edge_stripe(binary, y, kRightEdgeX + 3);
    }
    for (int y = 96; y <= 108; ++y)
    {
        draw_edge_stripe(binary, y, 45);
    }
    for (int y = 60; y <= 92; ++y)
    {
        draw_edge_stripe(binary, y, 5); // 与上一行边线跳变的远处左边线
    }
    // 93..95 行故意不留左边线，制造普通丢线行。

    const cv::Mat canvas = run_task2_and_draw(process, binary);

    // 重置后跳变边线（x=5）不加入边线列表：空洞上方（含触发重置的 92 行）
    // 都没有左边线点。94/95 行不断言：96 行圆点（半径 2）的弧线像素会延伸
    // 到这两行。
    const int above_rows[] = {62, 70, 80, 90, 92, 93};
    for (const int y : above_rows)
    {
        EXPECT_EQ(-1, left_line_min_col(canvas, y)) << "row " << y;
    }
    // 丢线前的旧边线保持 45。
    const int below_rows[] = {98, 103, 107};
    for (const int y : below_rows)
    {
        EXPECT_NEAR(45 - 2, left_line_min_col(canvas, y), 1) << "row " << y;
    }
    // 右边线全程正常。
    EXPECT_NEAR(kRightEdgeX + 2, right_line_max_col(canvas, 70), 1);
}

// 计划测试 5：无横线时各模式结果不变。task1 扫描区间为 [78,105]。
TEST(ScanSideLinesHorizontalBar, Task1ScanUnchangedWithVerticalEdges)
{
    ImageProcessConfig config;
    ImageProcess process(config);

    cv::Mat binary = make_black_binary();
    for (int y = 78; y <= 105; ++y)
    {
        draw_edge_stripe(binary, y, kLeftEdgeX);
        draw_edge_stripe(binary, y, kRightEdgeX + 3);
    }

    cv::Mat canvas(kImgH, kImgW, CV_8UC3, cv::Scalar(0, 0, 0));
    process.set_mid_line_mode(MID_AVG);
    process.get_side_line_task_1(binary, canvas, true);
    process.calculate_mid_line(binary.size());
    process.fit_polynomial();
    process.draw_line(canvas, -1.0f, "", TrackingTarget::STRAIGHT);

    const int sample_rows[] = {80, 88, 95, 103};
    for (const int y : sample_rows)
    {
        EXPECT_NEAR(kLeftEdgeX - 2, left_line_min_col(canvas, y), 1) << "row " << y;
        EXPECT_NEAR(kRightEdgeX + 2, right_line_max_col(canvas, y), 1) << "row " << y;
    }
    // 扫描区间之外不应有边线点。
    EXPECT_EQ(-1, left_line_min_col(canvas, 74));
    EXPECT_EQ(-1, left_line_min_col(canvas, 108));
}

// 计划测试 5：单边模式只搜索单侧，丢失一侧按原逻辑填满画布边缘。
TEST(ScanSideLinesHorizontalBar, SingleSideModesUnchangedWithVerticalEdges)
{
    const int sample_rows[] = {62, 75, 90, 106};

    ImageProcessConfig config;
    ImageProcess left_only(config);
    cv::Mat binary = make_black_binary();
    for (int y = 60; y <= 108; ++y)
    {
        draw_edge_stripe(binary, y, kLeftEdgeX);
        draw_edge_stripe(binary, y, kRightEdgeX + 3);
    }
    const cv::Mat canvas_left = run_task2_and_draw(left_only, binary, false, LEFT_ONLY);
    for (const int y : sample_rows)
    {
        EXPECT_NEAR(kLeftEdgeX - 2, left_line_min_col(canvas_left, y), 1) << "row " << y;
        // 右线丢失时按原逻辑填充到画布右缘。
        EXPECT_NEAR(kImgW - 1, right_line_max_col(canvas_left, y), 1) << "row " << y;
    }

    ImageProcess right_only(config);
    const cv::Mat canvas_right = run_task2_and_draw(right_only, binary, false, RIGHT_ONLY);
    for (const int y : sample_rows)
    {
        EXPECT_NEAR(kRightEdgeX + 2, right_line_max_col(canvas_right, y), 1) << "row " << y;
        // 左线丢失时按原逻辑填充到画布左缘。
        EXPECT_EQ(0, left_line_min_col(canvas_right, y)) << "row " << y;
    }
}

// 计划测试 5：斜向边线（急弯近似）不受遮挡判断影响，逐行跟踪命中。
TEST(ScanSideLinesHorizontalBar, SlantedEdgesTrackedPerRowInBothMode)
{
    ImageProcessConfig config;
    ImageProcess process(config);

    cv::Mat binary = make_black_binary();
    for (int y = 60; y <= 108; ++y)
    {
        const int left_x = cvRound(40.0 + (108 - y) * 0.25);
        const int right_x = cvRound(119.0 - (108 - y) * 0.2);
        draw_edge_stripe(binary, y, left_x);
        draw_edge_stripe(binary, y, right_x + 3);
    }

    const cv::Mat canvas = run_task2_and_draw(process, binary);

    const int sample_rows[] = {62, 70, 78, 85, 92, 99, 106};
    for (const int y : sample_rows)
    {
        const int expect_left = cvRound(40.0 + (108 - y) * 0.25);
        const int expect_right = cvRound(119.0 - (108 - y) * 0.2);
        EXPECT_NEAR(expect_left - 2, left_line_min_col(canvas, y), 1) << "row " << y;
        EXPECT_NEAR(expect_right + 2, right_line_max_col(canvas, y), 1) << "row " << y;
    }
}
