#ifndef RKNN_POOL_HPP
#define RKNN_POOL_HPP

#include "thread_pool.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

template <typename rknnModel, typename inputType, typename outputType>
class rknnPool
{
private:
    int threadNum;
    std::string detModelPath;
    std::string recModelPath;

    long long id;
    std::mutex idMtx, queueMtx;
    std::unique_ptr<dpool::ThreadPool> pool;
    std::queue<std::future<outputType>> futs;
    std::vector<std::shared_ptr<rknnModel>> models;
    std::vector<std::shared_ptr<std::mutex>> modelMtxes;

protected:
    int getModelId();

public:
    rknnPool(const std::string &detModelPath, const std::string &recModelPath, int threadNum);
    int init();
    int put(inputType inputData);
    int get(outputType &outputData);
    ~rknnPool();
};

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool(const std::string &detModelPath,
                                                     const std::string &recModelPath,
                                                     int threadNum)
{
    this->detModelPath = detModelPath;
    this->recModelPath = recModelPath;
    this->threadNum = threadNum;
    this->id = 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init()
{
    if (this->threadNum <= 0)
    {
        std::cerr << "threadNum must be greater than zero" << std::endl;
        return -1;
    }

    try
    {
        this->pool = std::unique_ptr<dpool::ThreadPool>(new dpool::ThreadPool(this->threadNum));
        for (int i = 0; i < this->threadNum; i++)
        {
            models.push_back(std::make_shared<rknnModel>(this->detModelPath,
                                                        this->recModelPath));
            modelMtxes.push_back(std::make_shared<std::mutex>());
        }
    }
    catch (const std::bad_alloc &e)
    {
        std::cout << "Out of memory: " << e.what() << std::endl;
        models.clear();
        modelMtxes.clear();
        pool.reset();
        return -1;
    }
    for (int i = 0, ret = 0; i < threadNum; i++)
    {
        ret = models[i]->init(models[0]->get_pctx(), i != 0, i);
        if (ret != 0)
        {
            models.clear();
            modelMtxes.clear();
            pool.reset();
            return ret;
        }
    }

    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId()
{
    std::lock_guard<std::mutex> lock(idMtx);
    int modelId = id % threadNum;
    id++;
    return modelId;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    if (!pool || models.empty())
        return -1;
    const int modelId = this->getModelId();
    const std::shared_ptr<rknnModel> model = models[modelId];
    const std::shared_ptr<std::mutex> modelMtx = modelMtxes[modelId];
    futs.push(pool->submit([model, modelMtx, inputData]() mutable -> outputType
                           {
                               // Generic pool workers may finish out of order. Serialize each
                               // RKNN context so a free worker cannot re-enter a busy model.
                               std::lock_guard<std::mutex> modelLock(*modelMtx);
                               return model->infer(inputData);
                           }));
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get(outputType &outputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    if (futs.empty() == true)
        return 1;
    outputData = futs.front().get();
    futs.pop();
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool()
{
    while (!futs.empty())
    {
        try
        {
            futs.front().get();
        }
        catch (const std::exception &e)
        {
            std::cerr << "worker failed while draining pool: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "worker failed while draining pool" << std::endl;
        }
        futs.pop();
    }
}

#endif
