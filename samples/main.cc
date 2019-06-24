// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string.h>
#include <sstream>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>
#include <setjmp.h>
#include <vector>
#include <memory>
#include "providers.h"
#include "local_filesystem.h"
#include <glib.h>

#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include "image_loader.h"
#include "Callback.h"
#include "AsyncRingBuffer.h"
#include "parallel_task_callback.h"
#include <fstream>
#include <condition_variable>
#include "common_macros.h"

static std::vector<std::string> readFileToVec(const std::string& file_path, size_t expected_line_count) {
  std::ifstream ifs(file_path);
  if (!ifs) {
    throw std::runtime_error("open file failed");
  }
  std::string line;
  std::vector<std::string> labels;
  while (std::getline(ifs, line)) {
    if (!line.empty()) labels.push_back(line);
  }
  if (labels.size() != expected_line_count) {
    throw std::runtime_error("line count mismatch");
  }
  return labels;
}

static int ExtractImageNumberFromFileName(const TCharString& image_file) {
  size_t s = image_file.rfind('.');
  if (s == std::string::npos) throw std::runtime_error("illegal filename");
  size_t s2 = image_file.rfind('_');
  if (s2 == std::string::npos) throw std::runtime_error("illegal filename");

  const char* start_ptr = image_file.c_str() + s2 + 1;
  const char* endptr = nullptr;
  long value = strtol(start_ptr, (char**)&endptr, 10);
  if (start_ptr == endptr || value > INT32_MAX || value <= 0) throw std::runtime_error("illegal filename");
  return static_cast<int>(value);
}

static void verify_input_output_count(OrtSession* session) {
  size_t count;
  ORT_ABORT_ON_ERROR(OrtSessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(OrtSessionGetOutputCount(session, &count));
  assert(count == 1);
}

void thread_pool_dispatcher(void* data, void* user_data) { (*(RunnableTask*)data)(); }

int real_main(int argc, ORTCHAR_T* argv[]) {
  if (argc < 5) return -1;
  std::vector<TCharString> image_file_paths;
  TCharString data_dir = argv[1];
  TCharString model_path = argv[2];
  // imagenet_lsvrc_2015_synsets.txt
  TCharString label_file_path = argv[3];
  TCharString validation_file_path = argv[4];
  std::vector<std::string> labels = readFileToVec(label_file_path, 1000);
  // TODO: remove the slash at the end of data_dir string
  LoopDir(data_dir, [&data_dir, &image_file_paths](const ORTCHAR_T* filename, OrtFileType filetype) -> bool {
    if (filetype != OrtFileType::TYPE_REG) return true;
    if (filename[0] == '.') return true;
    const char* p = strrchr(filename, '.');
    if (p == nullptr) return true;
    // as we tested filename[0] is not '.', p should larger than filename
    assert(p > filename);
    if (strcasecmp(p, ".JPEG") != 0 && strcasecmp(p, ".JPG") != 0) return true;
    TCharString v(data_dir);
#ifdef _WIN32
    v.append(1, '\\');
#else
    v.append(1, '/');
#endif
    v.append(filename);
    image_file_paths.emplace_back(v);
    return true;
  });
  std::vector<std::string> validation_data = readFileToVec(validation_file_path, image_file_paths.size());

  std::vector<uint8_t> data;
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  OrtSessionOptions* session_option;
  ORT_ABORT_ON_ERROR(OrtCreateSessionOptions(&session_option));
#ifdef USE_CUDA
  ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
#endif
  OrtSession* session;
  ORT_ABORT_ON_ERROR(OrtCreateSession(env, model_path.c_str(), session_option, &session));

  verify_input_output_count(session);

  OrtTypeInfo* info;
  ORT_ABORT_ON_ERROR(OrtSessionGetInputTypeInfo(session, 0, &info));
  const OrtTensorTypeAndShapeInfo* tensor_info;
  ORT_ABORT_ON_ERROR(OrtCastTypeInfoToTensorInfo(info, &tensor_info));
  size_t dim_count;
  ORT_ABORT_ON_ERROR(OrtGetDimensionsCount(tensor_info, &dim_count));
  assert(dim_count == 4);
  std::vector<int64_t> dims(dim_count);
  ORT_ABORT_ON_ERROR(OrtGetDimensions(tensor_info, dims.data(), dims.size()));
  if (dims[1] != dims[2] || dims[3] != 3) {
    printf("This model is not supported by this program. input tensor need be in NHWC format");
    return -1;
  }
  const int image_size = dims[1];
  const int output_class_count = 1001;
  OrtAllocatorInfo* allocator_info;
  ORT_ABORT_ON_ERROR(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
  std::atomic<size_t> top_1_correct_count(0);

  const int batch_size = 16;
  GError* err = NULL;
  GThreadPool* threadpool = g_thread_pool_new(thread_pool_dispatcher, nullptr, 8, TRUE, &err);
  if (err != NULL) {
    fprintf(stderr, "Unable to create thread pool: %s\n", err->message);
    g_error_free(err);
    return -1;
  }
  assert(threadpool != nullptr);

  const int channels = 3;
  size_t output_data_len = image_size * image_size * channels;
  float* output_data = new float[output_data_len * batch_size];

  std::mutex m;
  std::condition_variable cond_var;
  std::atomic<int> finished(0);
  // printf("loading %s\n", s.c_str());
  size_t remain = std::min<size_t>(image_file_paths.size(), batch_size);
  auto file_names_begin = image_file_paths.data();

  InceptionPreprocessing prepro(image_size, image_size, channels);
  AsyncRingBuffer buffer(160, threadpool, image_file_paths, &prepro);
  buffer.StartDownloadTasks();
  sleep(100000L);
  // if ((completed) % 160 == 0) {
  // printf("Top-1 Accuracy: %f\n", ((float)top_1_correct_count / completed));
  // printf("finished %f\n", ((float)completed / image_file_paths.size()));
  //}
  printf("Top-1 Accuracy %f\n", ((float)top_1_correct_count / image_file_paths.size()));
  OrtReleaseSessionOptions(session_option);
  OrtReleaseSession(session);
  OrtReleaseEnv(env);
  return 0;
}

int main(int argc, ORTCHAR_T* argv[]) {
  try {
    return real_main(argc, argv);
  } catch (const std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    return -1;
  }
}