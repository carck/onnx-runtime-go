// Use of this source code is governed by a Apache-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"github.com/carck/onnx-runtime-go"
	"log"
	"time"
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
		fmt.Printf("%v\n", res)
	}

	d1 := make([]float32, 512)
	d2 := make([]float32, 512)
	for j := 0; j < 512; j++ {
		d1[j] = float32(0.1)
		d2[j] = float32(0.9)
	}
	s := time.Now().UnixNano()
	for j := 1; j <= 1000000; j++ {
		onnx.EuclideanDistance512(d1, d2)
	}
	fmt.Printf("%d\n", time.Now().UnixNano()-s)

	s = time.Now().UnixNano()
	for j := 1; j <= 1000000; j++ {
		onnx.EuclideanDistance512C(d1, d2)
	}
	fmt.Printf("%d\n", time.Now().UnixNano()-s)
}
