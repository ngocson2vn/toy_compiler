#pragma once

#include <string>

namespace status {

template <typename ValueT>
class Result {
 public:
  Result(ValueT&& val, const std::string& err = "") : value_(val), error_(err) {}

  // Enable move semantics
  Result(Result&& other) = default;
  Result& operator=(Result&& other) = default;

  // Disable copy constructor and copy assignment operator
  Result(const Result& other) = delete;
  Result& operator=(const Result& other) = delete;

  bool ok() const {
    return error_.empty();
  }

  const ValueT& value() const {
    return value_;
  }

  const std::string& error_message() const {
    return error_;
  }

 private:
  ValueT value_;
  std::string error_;
};

}