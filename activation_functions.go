package main

import (
	"gonum.org/v1/gonum/mat"
)

type ActivationFunction func(*mat.Dense) *mat.Dense

func Relu(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	results := []float64{}
	for i := 0; i < rows; i++ {
		xRow := x.RowView(i)
		for j := 0; j < cols; j++ {
			v := xRow.At(i, j)
			if v < 0.0 {
				v = 0.0
			}

			results = append(results, v)
		}
	}

	return mat.NewDense(rows, cols, results)
}
