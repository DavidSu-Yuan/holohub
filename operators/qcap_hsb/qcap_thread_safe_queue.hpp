#pragma once
#include <queue>
#include <mutex>
#include <optional>
#include <chrono>
#include <condition_variable>
#include <functional>

#include <iostream>

namespace holoscan::ops {

template<typename T>
class ThreadSafeQueue {
public:
    struct Item {
        T value;
        std::chrono::steady_clock::time_point timestamp;

        Item(const T& v)
            : value(v), timestamp(std::chrono::steady_clock::now()) {}

        Item(T&& v)
            : value(std::move(v)), timestamp(std::chrono::steady_clock::now()) {}
    };

    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // -----------------------------
    // Push
    // -----------------------------
    void push(const T& value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.emplace_back(value);
        }
        cond_var_.notify_one();
    }

    void push(T&& value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.emplace_back(std::move(value));
        }
        cond_var_.notify_one();
    }

    void push_and_keep(const T& value, int max_size) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (queue_.size() >= max_size)
                queue_.pop_front();
            queue_.emplace_back(value);
        }
        cond_var_.notify_one();
    }

    void push_and_keep(T&& value, int max_size) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (queue_.size() >= max_size)
                queue_.pop_front();
            queue_.emplace_back(std::move(value));
        }
        cond_var_.notify_one();
    }

    void push_keep_and_call(const T& value, int max_size,
                            std::function<void(T&, std::chrono::milliseconds)> callback) {
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (queue_.size() >= max_size) {
                Item item = std::move(queue_.front());
                queue_.pop_front();
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - item.timestamp);
                callback(item.value, age);
            }
            queue_.emplace_back(value);
        }
        cond_var_.notify_one();
    }

    void push_keep_and_call(T&& value, int max_size,
                            std::function<void(T&, std::chrono::milliseconds)> callback) {
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (queue_.size() >= max_size) {
                Item item = std::move(queue_.front());
                queue_.pop_front();
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - item.timestamp);
                callback(item.value, age);
            }
            queue_.emplace_back(std::move(value));
        }
        cond_var_.notify_one();
    }

    // -----------------------------
    // Pop FIFO (blocking)
    // -----------------------------
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]() { return quit_flag_ || !queue_.empty(); });
        if (queue_.empty()) return std::nullopt;

        Item item = std::move(queue_.front());
        queue_.pop_front();
        return std::move(item.value);
    }

    std::optional<T> pop_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_var_.wait_for(lock, timeout, [this]() { return quit_flag_ || !queue_.empty(); }))
            return std::nullopt;

        if (queue_.empty()) return std::nullopt;
        Item item = std::move(queue_.front());
        queue_.pop_front();
        return std::move(item.value);
    }

    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;

        Item item = std::move(queue_.front());
        queue_.pop_front();
        return std::move(item.value);
    }

    // -----------------------------
    // pop_last(): returns newest item, drop everything
    // -----------------------------
    std::optional<T> pop_last() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]() { return quit_flag_ || !queue_.empty(); });
        if (queue_.empty()) return std::nullopt;

        Item last = std::move(queue_.back());
        queue_.clear();
        return std::move(last.value);
    }

    // -----------------------------
    // pop_last_and_call_with_age
    // -----------------------------
    std::optional<T> pop_last_and_call_with_age(
        std::chrono::milliseconds max_age,
        std::function<void(T&, std::chrono::milliseconds)> callback)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]() { return quit_flag_ || !queue_.empty(); });
        if (queue_.empty()) return std::nullopt;

        auto now = std::chrono::steady_clock::now();
        Item last = std::move(queue_.back());
        queue_.pop_back();

        // 遍历剩余元素
        auto it = queue_.begin();
        while (it != queue_.end()) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->timestamp);
            if (age >= max_age) {
                Item oldItem = std::move(*it);
                it = queue_.erase(it);
                lock.unlock();
                callback(oldItem.value, age);
                lock.lock();
            } else {
                ++it;
            }
        }

        return std::move(last.value);
    }

    // -----------------------------
    // Utilities
    // -----------------------------
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void quit() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            quit_flag_ = true;
        }
        cond_var_.notify_all();
    }

    void reset_quit() {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_flag_ = false;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
    }

    void clear_and_call(std::function<void(T&, std::chrono::milliseconds)> callback) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        for (auto& item : queue_) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - item.timestamp);
            callback(item.value, age);
        }
        queue_.clear();
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cond_var_;
    std::deque<Item>        queue_;
    bool quit_flag_ = false;
};

} // namespace hololink::ops
