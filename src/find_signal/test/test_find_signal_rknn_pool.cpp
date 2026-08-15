#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "rknn_pool.hpp"

class FakePPOCRModel
{
public:
    static std::vector<std::string> det_paths;
    static std::vector<std::string> rec_paths;
    static std::vector<int> initialized_workers;
    static std::mutex infer_mutex;
    static int active_calls[3];
    static bool overlap_detected;

    FakePPOCRModel(const std::string &det_path, const std::string &rec_path)
        : worker_id_(-1)
    {
        det_paths.push_back(det_path);
        rec_paths.push_back(rec_path);
    }

    int init(void *, bool, int worker_id)
    {
        worker_id_ = worker_id;
        initialized_workers.push_back(worker_id);
        return 0;
    }

    void *get_pctx()
    {
        return nullptr;
    }

    int infer(int input)
    {
        {
            std::lock_guard<std::mutex> lock(infer_mutex);
            ++active_calls[worker_id_];
            if (active_calls[worker_id_] > 1)
                overlap_detected = true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(
            input == 0 ? 150 : (input == 1 ? 30 : 10)));

        {
            std::lock_guard<std::mutex> lock(infer_mutex);
            --active_calls[worker_id_];
        }
        return input * 10;
    }

    static void reset_inference_state()
    {
        std::lock_guard<std::mutex> lock(infer_mutex);
        active_calls[0] = active_calls[1] = active_calls[2] = 0;
        overlap_detected = false;
    }

private:
    int worker_id_;
};

std::vector<std::string> FakePPOCRModel::det_paths;
std::vector<std::string> FakePPOCRModel::rec_paths;
std::vector<int> FakePPOCRModel::initialized_workers;
std::mutex FakePPOCRModel::infer_mutex;
int FakePPOCRModel::active_calls[3] = {0, 0, 0};
bool FakePPOCRModel::overlap_detected = false;

TEST(FindSignalRknnPool, PassesBothModelPathsAndPreservesFifo)
{
    FakePPOCRModel::det_paths.clear();
    FakePPOCRModel::rec_paths.clear();
    FakePPOCRModel::initialized_workers.clear();

    rknnPool<FakePPOCRModel, int, int> pool("det.rknn", "rec.rknn", 2);
    ASSERT_EQ(pool.init(), 0);

    EXPECT_EQ(FakePPOCRModel::det_paths,
              (std::vector<std::string>{"det.rknn", "det.rknn"}));
    EXPECT_EQ(FakePPOCRModel::rec_paths,
              (std::vector<std::string>{"rec.rknn", "rec.rknn"}));
    EXPECT_EQ(FakePPOCRModel::initialized_workers,
              (std::vector<int>{0, 1}));

    ASSERT_EQ(pool.put(1), 0);
    ASSERT_EQ(pool.put(2), 0);

    int output = 0;
    ASSERT_EQ(pool.get(output), 0);
    EXPECT_EQ(output, 10);
    ASSERT_EQ(pool.get(output), 0);
    EXPECT_EQ(output, 20);
    EXPECT_EQ(pool.get(output), 1);
}

TEST(FindSignalRknnPool, RejectsNonPositiveThreadCount)
{
    rknnPool<FakePPOCRModel, int, int> pool("det.rknn", "rec.rknn", 0);
    EXPECT_NE(pool.init(), 0);
}

TEST(FindSignalRknnPool, DoesNotRunOneModelContextConcurrently)
{
    FakePPOCRModel::reset_inference_state();
    rknnPool<FakePPOCRModel, int, int> pool("det.rknn", "rec.rknn", 3);
    ASSERT_EQ(pool.init(), 0);

    for (int input = 0; input < 4; ++input)
        ASSERT_EQ(pool.put(input), 0);

    for (int expected = 0; expected < 4; ++expected)
    {
        int output = -1;
        ASSERT_EQ(pool.get(output), 0);
        EXPECT_EQ(output, expected * 10);
    }

    EXPECT_FALSE(FakePPOCRModel::overlap_detected);
}
