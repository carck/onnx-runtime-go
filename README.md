## Introduction

Go binding for onnx runtime, only support 1 input and 1 output in current phase

## Build

- export CGO_CFLAGS=-I/source/onnxruntime-osx/include
- export CGO_LDFLAGS=-L/source/onnxruntime-osx/lib

## Example
```
package main

import (
	"github.com/carck/onnx-runtime-go"
	"log"
)

func main() {
	shape := []int64{1, 3, 112, 112}
	inputNames := []string{"input.1"}
	outputNames := []string{"683"}
	model := onnx.NewModel("facenet.onnx", shape, inputNames, outputNames, onnx.CPU)
	defer model.Delete()

	data := make([]float32, 1*3*112*112)
	output := model.RunInference(data)
	defer output.Delete()

	log.Println("num dims: %s", output.NumDims())
	log.Println("dim1: %s", output.Dim(1))
	
	res := make([]float32, 512)
	output.CopyToBuffer(res, 512*4)
	fmt.Printf("%v", res)
}
```
