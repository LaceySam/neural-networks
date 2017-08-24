package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type LossFunction func(x, y *mat.Dense) *mat.VecDense

func L2Loss(x, y *mat.Dense) *mat.VecDense {
	rows, cols := x.Dims()
	vals := []float64{}
	for i := 0; i < rows; i++ {
		squareSum := 0.0
		row := x.RowView(i)
		for j := 0; j < cols; j++ {
			squareSum = squareSum + math.Pow(row.At(j, 0), 2)
		}

		vals = append(vals, squareSum/(float64(cols)*2.0))
	}

	return mat.NewVecDense(rows, vals)
}
