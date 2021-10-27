// Use of this source code is governed by a Apache-style
// license that can be found in the LICENSE file.

package onnx

/*
#cgo LDFLAGS: -lonnxruntime
#cgo arm64 CFLAGS: -DARMNN=1
#include "onnx_capi.h"
*/
import "C"
import (
	"reflect"
	"unsafe"
)

type Model struct {
	env *C.OnnxEnv
}

type Tensor struct {
	t *C.OrtValue
}

type EP int

const (
	CPU EP = iota
	CUDA
	ROCM
	ARMNN
	TENSORRT
)

func NewModel(model_path string, shape []int64, inputNames []string, outputNames []string, mode EP) *Model {
	ptr := C.CString(model_path)
	defer C.free(unsafe.Pointer(ptr))

	t := C.OnnxNewOrtSession(ptr, C.int(mode))

	t.input_shape_len = C.size_t(len(shape))
	for i, s := range shape {
		t.input_shape[i] = C.int64_t(s)
	}

	t.input_names_len = C.size_t(len(inputNames))
	for i, s := range inputNames {
		t.input_names[i] = C.CString(s)
	}

	t.output_names_len = C.size_t(len(outputNames))
	for i, s := range outputNames {
		t.output_names[i] = C.CString(s)
	}

	return &Model{env: t}
}

// Invoke invoke the task.
func (m *Model) RunInference(data []float32) *Tensor {
	t := C.OnnxRunInference(m.env,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)*4),
	)
	return &Tensor{t: t}
}

func (m *Model) Delete() {
	if m != nil {
		C.OnnxDeleteOrtSession(m.env)
	}
}

func (t *Tensor) NumDims() int {
	return int(C.OnnxTensorNumDims(t.t))
}

// Dim return dimension of the element specified by index.
func (t *Tensor) Dim(index int) int64 {
	return int64(C.OnnxTensorDim(t.t, C.int32_t(index)))
}

// Shape return shape of the tensor.
func (t *Tensor) Shape() []int64 {
	shape := make([]int64, t.NumDims())
	for i := 0; i < t.NumDims(); i++ {
		shape[i] = t.Dim(i)
	}
	return shape
}

func (t *Tensor) Delete() {
	if t != nil {
		C.OnnxReleaseTensor(t.t)
	}
}

func (t *Tensor) CopyToBuffer(b interface{}, size int) {
	C.OnnxTensorCopyToBuffer(t.t, unsafe.Pointer(reflect.ValueOf(b).Pointer()), C.size_t(size))
}
