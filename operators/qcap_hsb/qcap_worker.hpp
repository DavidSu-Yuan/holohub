#pragma once
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace holoscan::ops {

class QcapWorker {
public:
    QcapWorker() : stop_flag_(false) {
        worker_thread_ = std::thread([this]() { this->run(); });
    }

    ~QcapWorker() {
        stop();
    }

    // Add a new task to be executed by the worker thread
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
        }
        cond_var_.notify_one();
    }

    // Graceful stop (waits for worker to finish)
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_flag_ = true;
        }
        cond_var_.notify_one();

        if (worker_thread_.joinable())
            worker_thread_.join();
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait(lock, [this]() { return stop_flag_ || !tasks_.empty(); });

                if (stop_flag_ && tasks_.empty())
                    break;

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            // Execute the task outside of the lock
            task();
        }
    }

    std::thread worker_thread_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    std::atomic<bool> stop_flag_;
};

} // namespace hololink::ops
