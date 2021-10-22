// Use of this source code is governed by a Apache-style
// license that can be found in the LICENSE file.

#ifndef onnx_capi_h_
#define onnx_capi_h_

#include <stddef.h>
#include <stdint.h>

#include <onnxruntime_c_api.h>

typedef struct {
	OrtEnv* env;
	OrtSessionOptions* session_options;
	OrtSession* session;
} OnnxEnv;

OnnxEnv* OnnxNewOrtSession(const char* model_path);

void OnnxDeleteOrtSession(OnnxEnv* env);

OrtValue* OnnxRunInference(OnnxEnv* env, 
			float* model_input, size_t model_input_len, 
			int64_t* input_shape, size_t input_shape_len,
			const char* input_names[], const char* output_names[]);

void OnnxReleaseTensor(OrtValue* tensor);

size_t OnnxTensorNumDims(OrtValue*  tensor);

int64_t OnnxTensorDim(OrtValue*  tensor, int index);

void OnnxTensorCopyToBuffer(OrtValue*  tensor, void * value, size_t size);

// Array helper
char** MakeCharArray(int size);

void SetArrayString(char **a, char *s, int n);

void FreeCharArray(char **a, int size);

#endif // onnx_capi_h_
