#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "rknn_api.h"
#include "rknn_pool.hpp"

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
