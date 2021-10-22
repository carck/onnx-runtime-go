// Use of this source code is governed by a Apache-style
// license that can be found in the LICENSE file.

package onnx

/*
#cgo LDFLAGS: -lonnxruntime
#include "onnx_capi.h"
*/
import "C"
import (
	"reflect"
	"unsafe"
)

type Model struct {
	env         *C.OnnxEnv
	shape       []int64
	inputNames  []string
	outputNames []string
}

type Tensor struct {
	t *C.OrtValue
}

func NewModel(model_path string, shape []int64, inputNames []string, outputNames []string) *Model {
	ptr := C.CString(model_path)
	defer C.free(unsafe.Pointer(ptr))

	t := C.OnnxNewOrtSession(ptr)

	return &Model{env: t, shape: shape, inputNames: inputNames, outputNames: outputNames}
}

// Invoke invoke the task.
func (m *Model) RunInference(data []float32) *Tensor {
	inputNames := C.MakeCharArray(C.int(len(m.inputNames)))
	defer C.FreeCharArray(inputNames, C.int(len(m.inputNames)))
	for i, s := range m.inputNames {
		C.SetArrayString(inputNames, C.CString(s), C.int(i))
	}

	outputNames := C.MakeCharArray(C.int(len(m.outputNames)))
	defer C.FreeCharArray(outputNames, C.int(len(m.outputNames)))
	for i, s := range m.outputNames {
		C.SetArrayString(outputNames, C.CString(s), C.int(i))
	}

	t := C.OnnxRunInference(m.env,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)*4),
		(*C.int64_t)(unsafe.Pointer(&m.shape[0])),
		C.size_t(len(m.shape)),
		inputNames,
		outputNames,
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
