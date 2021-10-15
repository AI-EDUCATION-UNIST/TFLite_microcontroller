/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/common.h"

/* TensorFlow Lite for Microcontroller 라이브러리 header를 include하세요 */
#include ?
#include ?
#include ?
#include ?

/* 모델 header 파일(tensorflow/lite/micro/models/person_detect_model_data.h)을 include하세요 */
#include ?

/* Include test data */
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/testdata/person_image_data.h"

/* Unit test header를 include합니다 */
#include ?

/* 테스트 매크로를 시작(BEGIN)하세요 */

TF_LITE_MICRO_TESTS_?

TF_LITE_MICRO_TEST(TestInvoke) {
  // Set up logging.
  /* 로깅 설정을 위해 micro_error_reporter의 포인터(error_reporter)를 생성하세요. */
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = ?

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  /* person_detect_model_data.h에 선언된 모델 배열을 로드하세요. */
  const tflite::Model* model = ?
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.

  /* MicroMutableOpResolver 인스턴스를 만들고, 모델 수행에 필요한 아래 5개의 operation을 추가합니다 */
  /* AddAveragePool2D, AddConv2D, AddDepthwiseConv2D, AddReshape, AddSoftmax */
  tflite::?<5> ?;
  micro_op_resolver.?;
  micro_op_resolver.?;
  micro_op_resolver.?;
  micro_op_resolver.?;
  micro_op_resolver.?;

  // Create an area of memory to use for input, output, and intermediate arrays.
  /* 136 * 1024 크기의 int 배열을 만드세요 (이름: tensor_arena). */
  constexpr int tensor_arena_size = ?;
  uint8_t ?[tensor_arena_size];

  // Build an interpreter to run the model with.
  /* 아래 변수들을 사용하여 interpreter를 build하세요. */
  /* micro_op_resolver, tensor_arena, tensor_arena_size, error_reporter */
  tflite::? ?(model, ?, ?, ?, ?);

  /* AllocateTensors()를 사용하여 tensor 메모리를 할당하세요. */
  interpreter.?;

  // Get information about the memory area to use for the model's input.
  /* interpreter의 input 멤버의 0번째 element를 통해 input tensor의 포인터를 받아오세요. */
  TfLiteTensor* input = ?.?(?);

  // Make sure the input has the properties we expect.
  /* input 포인터가 nullptr이 아닌지 확인하세요. */
  TF_LITE_MICRO_EXPECT_NE(?, ?);
  /* input tensor의 dims size가 4인지 확인하세요. */
  TF_LITE_MICRO_EXPECT_EQ(?, ?);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);

  // Copy an image with a person into the memory area used for the input.
  TFLITE_DCHECK_EQ(input->bytes, static_cast<size_t>(g_person_image_data_size));

  /* 입력 텐서(input->data.int8)에 input->bytes 크기의 g_person_image_data를 copy하세요. */
  memcpy(?, ?, ?);

  // Run the model on this input and make sure it succeeds.
  /* interpreter의 invoke()를 호출하여 모델을 실행하세요. */
  TfLiteStatus invoke_status = ?.?();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  /* interpreter의 output 멤버의 0번째 element를 통해 output tensor의 포인터를 받아오세요. */
  TfLiteTensor* output = ?.?(?);
 
  /* output tensor의 dims size가 2인지 확인하세요. */
  TF_LITE_MICRO_EXPECT_EQ(?, ?);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

  /* person_score를 output->data.int8[kPersonIndex]에서 받아오세요. */
  int8_t person_score = ?->?[?];
  /* no_person_score를 output->data.int8[kNotAPersonIndex]에서 받아오세요. */
  int8_t no_person_score = ?->?[?];
  TF_LITE_REPORT_ERROR(&micro_error_reporter,
                       "person data.  person score: %d, no person score: %d\n",
                       person_score, no_person_score);

  // Make sure that the expected "Person" score is higher than the other class.
  /* person_score가 no_person_score보다 높은지 확인하세요. */
  TF_LITE_MICRO_EXPECT_?(?, ?);

  /* 입력 텐서(input->data.int8)에 input->bytes 크기의 g_no_person_image_data를 copy하세요. */
  memcpy(?, ?, ?);

  // Run the model on this "No Person" input.
  /* interpreter의 invoke()를 호출하여 모델을 실행하세요. */
  invoke_status = ?.?();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  /* interpreter의 output 멤버의 0번째 element를 통해 output tensor의 포인터를 받아오세요. */
  output = ?.?(?);

  /* output tensor의 dims size가 2인지 확인하세요. */
  TF_LITE_MICRO_EXPECT_EQ(?, ?);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);

  /* person_score를 output->data.int8[kPersonIndex]에서 받아오세요. */
  person_score = ?->?.?[?];
  /* no_person_score를 output->data.int8[kNotAPersonIndex]에서 받아오세요. */
  no_person_score = ?->?.?[?];
  TF_LITE_REPORT_ERROR(
      &micro_error_reporter,
      "no person data.  person score: %d, no person score: %d\n", person_score,
      no_person_score);
  
  // Make sure that the expected "No Person" score is higher.
  /* no_person_score가 person_score보다 높은지 확인하세요. */
  TF_LITE_MICRO_EXPECT_?(?, ?);

  TF_LITE_REPORT_ERROR(&micro_error_reporter, "Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
