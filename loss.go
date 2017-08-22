package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type LossFunction func(x, y *mat.Dense) float64

func L2Loss(x, y *mat.Dense) float64 {
	x.Sub(x, y)

	sum := 0.0
	rows, cols := x.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum = sum + math.Pow(x.At(i, j), 2)
		}
	}

	return 0.5 * sum
}
