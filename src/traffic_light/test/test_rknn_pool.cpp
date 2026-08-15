#include <gtest/gtest.h>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "rknn_api.h"
#include "rknn_pool.hpp"
#include "yolov8.h"

class FakeModel {
public:
    static std::vector<int> initialized_workers;

    explicit FakeModel(const std::string &) {}

    int init(rknn_context *, bool, int worker_id) {
        initialized_workers.push_back(worker_id);
        return 0;
    }

    rknn_context *get_pctx() { return nullptr; }
    int infer(int input) { return input * 10; }
};

std::vector<int> FakeModel::initialized_workers;

class OverlapTrackingModel {
public:
    static std::mutex state_mutex;
    static int active_calls[3];
    static bool overlap_detected;

    explicit OverlapTrackingModel(const std::string &) : worker_id_(-1) {}

    int init(rknn_context *, bool, int worker_id) {
        worker_id_ = worker_id;
        return 0;
    }

    rknn_context *get_pctx() { return nullptr; }

    int infer(int input) {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            ++active_calls[worker_id_];
            if (active_calls[worker_id_] > 1)
                overlap_detected = true;
        }

        // Keep model 0 busy while models 1 and 2 finish. A generic worker can
        // then pick the next round-robin task, which is assigned to model 0.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(input == 0 ? 150 : 10));

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            --active_calls[worker_id_];
        }
        return input;
    }

    static void reset() {
        std::lock_guard<std::mutex> lock(state_mutex);
        active_calls[0] = active_calls[1] = active_calls[2] = 0;
        overlap_detected = false;
    }

private:
    int worker_id_;
};

std::mutex OverlapTrackingModel::state_mutex;
int OverlapTrackingModel::active_calls[3] = {0, 0, 0};
bool OverlapTrackingModel::overlap_detected = false;

TEST(RknnPool, InitializesThreeWorkersAndPreservesFifoResults) {
    FakeModel::initialized_workers.clear();
    rknnPool<FakeModel, int, int> pool("ignored", 3);

    ASSERT_EQ(pool.init(), 0);
    ASSERT_EQ(FakeModel::initialized_workers,
              (std::vector<int>{0, 1, 2}));

    ASSERT_EQ(pool.put(1), 0);
    ASSERT_EQ(pool.put(2), 0);
    ASSERT_EQ(pool.put(3), 0);

    int output = 0;
    ASSERT_EQ(pool.get(output), 0);
    EXPECT_EQ(output, 10);
    ASSERT_EQ(pool.get(output), 0);
    EXPECT_EQ(output, 20);
    ASSERT_EQ(pool.get(output), 0);
    EXPECT_EQ(output, 30);
    EXPECT_EQ(pool.get(output), 1);
}

TEST(RknnPool, DoesNotRunOneModelContextConcurrently) {
    OverlapTrackingModel::reset();
    rknnPool<OverlapTrackingModel, int, int> pool("ignored", 3);
    ASSERT_EQ(pool.init(), 0);

    for (int input = 0; input < 4; ++input)
        ASSERT_EQ(pool.put(input), 0);

    for (int expected = 0; expected < 4; ++expected) {
        int output = -1;
        ASSERT_EQ(pool.get(output), 0);
        EXPECT_EQ(output, expected);
    }

    EXPECT_FALSE(OverlapTrackingModel::overlap_detected);
}

TEST(YoloV8Core, MapsWorkerIdsToDedicatedCores) {
    EXPECT_EQ(yolov8_core_mask_for_worker(0), RKNN_NPU_CORE_0);
    EXPECT_EQ(yolov8_core_mask_for_worker(1), RKNN_NPU_CORE_1);
    EXPECT_EQ(yolov8_core_mask_for_worker(2), RKNN_NPU_CORE_2);
    EXPECT_EQ(yolov8_core_mask_for_worker(3), RKNN_NPU_CORE_UNDEFINED);
}
