// Use of this source code is governed by a Apache-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"github.com/carck/onnx-runtime-go"
	"log"
)

func main() {
	shape := []int64{1, 3, 112, 112}
	inputNames := []string{"input.1"}
	outputNames := []string{"683"}

	model := onnx.NewModel("facenet.onnx", shape, inputNames, outputNames, onnx.ARMNN)
	defer model.Delete()

	for i := 1; i <= 10; i++ {
		data := make([]float32, 1*3*112*112)
		output := model.RunInference(data)
		defer output.Delete()

		log.Println("num dims: %s", output.NumDims())

		res := make([]float32, 512)
		output.CopyToBuffer(res, 512*4)
		fmt.Printf("%v", res)
	}
}
