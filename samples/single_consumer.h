// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <functional>

/**
 * A special FIFO that is restricted to only have one consumer
 * The consumer must return the previous borrowed item before taking the next
 */
template <typename ValueType>
class SingleConsumerFIFO {
 public:
  struct ListEntry {
    ValueType* value = nullptr;
    ListEntry* next = nullptr;
  };
 private:
  // fixed size
  std::vector<ListEntry> values_;
  ListEntry* free_list_ = nullptr;
  // whenever free_list_ is nullptr, free_list_tail_ should equal to &free_list_;
  ListEntry** free_list_tail_ = &free_list_;
  bool is_consumer_running_ = false;
  std::function<void(ValueType* )> deleter_;

 public:
  explicit SingleConsumerFIFO(size_t len, const std::function<void(ValueType* )>& deleter) : values_(len),
                                                                                             deleter_(deleter){

  }
  template <typename T>
  void Init(const T& t) {
    for (ListEntry& e : values_) {
      t(e);
    }
  }

  /**
   * Return a borrowed item
   * @param e a pointer returned from the Take() function
   * @return ID of the entry, in [0,len)
   */
  size_t Return(ListEntry *e) {
    is_consumer_running_ = false;
    return e - values_.data();
  }

  void Put(size_t element_id) {
    // printf("Append %zd to the free list\n", element_id);
    ListEntry* t = &values_[element_id];
    t->next = nullptr;
    (*free_list_tail_) = t;
    free_list_tail_ = &t;
  }

  ListEntry* Take() {
    if (is_consumer_running_) return nullptr;
    if (free_list_ == nullptr) {
      is_consumer_running_ = false;
      return nullptr;
    }
    auto input_tensor = free_list_;
    is_consumer_running_ = true;
    if ((free_list_ = free_list_->next) == nullptr) free_list_tail_ = &free_list_;
    return input_tensor;
  }

  ~SingleConsumerFIFO() {
    for (const ListEntry& t : values_) {
      deleter_(t.value);
    }
  }
};