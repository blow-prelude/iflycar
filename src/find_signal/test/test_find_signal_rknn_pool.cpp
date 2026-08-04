#include <chrono>
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

    FakePPOCRModel(const std::string &det_path, const std::string &rec_path)
    {
        det_paths.push_back(det_path);
        rec_paths.push_back(rec_path);
    }

    int init(void *, bool, int worker_id)
    {
        initialized_workers.push_back(worker_id);
        return 0;
    }

    void *get_pctx()
    {
        return nullptr;
    }

    int infer(int input)
    {
        if (input == 1)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        return input * 10;
    }
};

std::vector<std::string> FakePPOCRModel::det_paths;
std::vector<std::string> FakePPOCRModel::rec_paths;
std::vector<int> FakePPOCRModel::initialized_workers;

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
