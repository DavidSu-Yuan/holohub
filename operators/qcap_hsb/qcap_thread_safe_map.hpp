#pragma once
#include <map>
#include <mutex>
#include <optional>
#include <functional>

namespace holoscan::ops {

template<typename Key, typename Value>
class ThreadSafeMap {
public:
    // Insert or update
    void insert_or_assign(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        data_[key] = value;
    }

    void insert_or_assign(Key&& key, Value&& value) {
        std::lock_guard<std::mutex> lock(mtx_);
        data_[std::move(key)] = std::move(value);
    }

    // Try to get a copy of the value
    std::optional<Value> get(const Key& key) const {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = data_.find(key);
        if (it == data_.end()) return std::nullopt;
        return it->second;
    }

    // Check if key exists
    bool contains(const Key& key) const {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_.find(key) != data_.end();
    }

    // Erase key
    bool erase(const Key& key) {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_.erase(key) > 0;
    }

    // Clear entire map
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        data_.clear();
    }

    // Number of elements
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_.empty();
    }

    // ✅ New: Clear all and call user function before erase
    // Example: map.clear_and_call([](const Key& k, Value& v){ delete v; });
    void clear_and_call(std::function<void(const Key&, Value&)> func) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& [k, v] : data_) {
            func(k, v);
        }
        data_.clear();
    }

    // Apply function to all entries
    // If func(key, value) returns true → remove entry
    void for_all(std::function<bool(const Key&, Value&)> func) {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto it = data_.begin(); it != data_.end(); ) {
            if (func(it->first, it->second)) {
                it = data_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Visit without modification
    void for_each(std::function<void(const Key&, const Value&)> func) const {
        std::lock_guard<std::mutex> lock(mtx_);
        for (const auto& [k, v] : data_) {
            func(k, v);
        }
    }

private:
    mutable std::mutex mtx_;
    std::map<Key, Value> data_;
};

} // namespace hololink::ops
