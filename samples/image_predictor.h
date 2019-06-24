
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "image_loader.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"


class ImagePredictor : public DataProcessing {
 private:
  OrtSession* session;

 public:


  void operator()(const void* input_data, void* output_data) override{
    const OrtValue* input_tensor = reinterpret_cast<const OrtValue*>(input_data);
    const char* input_name = "input:0";
    const char* output_name = "InceptionV4/Logits/Predictions:0";
    OrtValue* output_tensor = NULL;
    ORT_THROW_ON_ERROR(OrtRun(session, NULL, &input_name, &input_tensor, 1, &output_name, 1, &output_tensor));
    float* probs;
    ORT_THROW_ON_ERROR(OrtGetTensorMutableData(output_tensor, (void**)&probs));
    for (size_t i = 0; i != remain; ++i) {
      float max_prob = probs[1];
      int max_prob_index = 1;
      for (int i = max_prob_index + 1; i != output_class_count; ++i) {
        if (probs[i] > max_prob) {
          max_prob = probs[i];
          max_prob_index = i;
        }
      }
      // TODO:extract number from filename, to index validation_data
      auto s = file_names_begin[i];
      int test_data_id = ExtractImageNumberFromFileName(s);
      // printf("%d\n",(int)max_prob_index);
      // printf("%s\n",labels[max_prob_index - 1].c_str());
      // printf("%s\n",validation_data[test_data_id - 1].c_str());
      if (labels[max_prob_index - 1] == validation_data[test_data_id - 1]) {
        ++top_1_correct_count;
      }
      probs += output_class_count;
 }
};