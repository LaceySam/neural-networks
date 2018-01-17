package main_test

import (
	"testing"

	"gonum.org/v1/gonum/mat"

	"neural-networks"
)

func TestL2Loss(t *testing.T) {
	testX := mat.NewDense(1, 3, []float64{-4, 2, -3})
	testY := mat.NewDense(1, 3, []float64{1, 1, 1})

	expected := mat.NewVecDense(1, []float64{7})

	gotCost := main.L2Loss(testX, testY)
	err := compareDense(expected, gotCost)
	if err != nil {
		t.Error(err)
	}
}
