// Use of this source code is governed by a Apache-style
// license that can be found in the LICENSE file.

#include "onnx_capi.h"

#include <stdio.h>
#include <assert.h>

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      printf("%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

const OrtApi* g_ort = NULL;

void VerifyInputOutputCount(OrtSession* session) {
  size_t count;
  ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  assert(count == 1);
  ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  assert(count == 1);
}

OnnxEnv* OnnxNewOrtSession(const char* model_path){
	int ret = 0;

	if(g_ort == NULL){
		g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
		if (!g_ort) {
			printf("runtime init error!\n");
			return NULL;
		}
	}

	OnnxEnv* onnx_env=(OnnxEnv*)malloc(sizeof(OnnxEnv));

	ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "infer", &onnx_env->env));

	
	ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&onnx_env->session_options));

	ORT_ABORT_ON_ERROR(g_ort->CreateSession(onnx_env->env, model_path, onnx_env->session_options, &onnx_env->session));

	VerifyInputOutputCount(onnx_env->session);

  	return onnx_env;
}

void OnnxDeleteOrtSession(OnnxEnv* env){
	if(g_ort){
		g_ort->ReleaseSessionOptions(env->session_options);
		g_ort->ReleaseSession(env->session);
		g_ort->ReleaseEnv(env->env);
	}
}

OrtValue* OnnxRunInference(OnnxEnv* env, 
					float * model_input, size_t model_input_len, 
					int64_t* input_shape, size_t input_shape_len,
					const char* input_names[], const char* output_names[]){

	OrtMemoryInfo* memory_info;
	ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
	
	OrtValue* input_tensor = NULL;
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
															input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
															&input_tensor));
	assert(input_tensor != NULL);

	int is_tensor;
	ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
	assert(is_tensor);

	g_ort->ReleaseMemoryInfo(memory_info);

	OrtValue* output_tensor = NULL;
	ORT_ABORT_ON_ERROR(g_ort->Run(env->session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
									&output_tensor));
	assert(output_tensor != NULL);

	ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
	assert(is_tensor);

  	OnnxReleaseTensor(input_tensor);
	return output_tensor;
}

void OnnxReleaseTensor(OrtValue*  tensor){
	g_ort->ReleaseValue(tensor);
}

size_t OnnxTensorNumDims(OrtValue*  tensor){
	struct OrtTensorTypeAndShapeInfo* shape_info;
	ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(tensor, &shape_info));
  	
	size_t dim_count;
  	ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(shape_info, &dim_count));
	return dim_count;
}

int64_t OnnxTensorDim(OrtValue*  tensor, int index){
	struct OrtTensorTypeAndShapeInfo* shape_info;
  	ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(tensor, &shape_info));
  	
	size_t dim_count;
  	ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(shape_info, &dim_count));

	int64_t* dims = (int64_t*)malloc(dim_count*sizeof(int64_t));
  	ORT_ABORT_ON_ERROR(g_ort->GetDimensions(shape_info, dims, dim_count));
	int64_t ret = *(dims+index);
	free(dims);
	return ret;
}

void OnnxTensorCopyToBuffer(OrtValue* tensor, void * value, size_t size){
	float* f;
  	ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(tensor, (void**)&f));
	memcpy(value, f, size);
}

char** MakeCharArray(int size) {
        return calloc(sizeof(char*), size);
}

void SetArrayString(char **a, char *s, int n) {
        a[n] = s;
}

void FreeCharArray(char **a, int size) {
        int i;
        for (i = 0; i < size; i++)
                free(a[i]);
        free(a);
}