package main_test

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"

	"neural-networks"
)

func compareDense(x, y mat.Matrix) error {
	xRows, xCols := x.Dims()
	yRows, yCols := y.Dims()

	if xRows != yRows {
		return fmt.Errorf("Comparison row mismatch X: %d, y: %d", xRows, yRows)
	}

	if xCols != yCols {
		return fmt.Errorf("Comparison column mismatch X: %d, y: %d", xCols, yCols)
	}

	for i := 0; i < xCols; i++ {
		for j := 0; j < xRows; j++ {
			if x.At(j, i) != y.At(j, i) {
				return fmt.Errorf(
					"Comparision fail at point (%d, %d). X: %v, y: %v",
					i,
					j,
					x,
					y,
				)
			}
		}
	}

	return nil
}

func TestGetPositiveColumnMean(t *testing.T) {
	matrix := mat.NewDense(4, 3, []float64{5, 6, 7, 3, 5, 6, 6, 6, 7, 4, 3, 2})
	expectedMean := mat.NewVecDense(3, []float64{4.5, 5, 5.5})

	gotMean := main.GetColumnMean(matrix)
	err := compareDense(expectedMean, gotMean)
	if err != nil {
		t.Error(err)
	}
}

func TestGetNegativeColumnMean(t *testing.T) {
	matrix := mat.NewDense(4, 3, []float64{-5, -6, -7, -3, -5, -6, -6, -6, -7, -4, -3, -2})
	expectedMean := mat.NewVecDense(3, []float64{-4.5, -5, -5.5})

	gotMean := main.GetColumnMean(matrix)
	err := compareDense(expectedMean, gotMean)
	if err != nil {
		t.Error(err)
	}
}

func TestGetMixedColumnMean(t *testing.T) {
	matrix := mat.NewDense(4, 3, []float64{5, -6, 7, -3, 5, -6, 6, -6, 7, -4, 3, -2})
	expectedMean := mat.NewVecDense(3, []float64{1, -1, 1.5})

	gotMean := main.GetColumnMean(matrix)
	err := compareDense(expectedMean, gotMean)
	if err != nil {
		t.Error(err)
	}
}

func TestAddColumnVector(t *testing.T) {
	matrix := mat.NewDense(2, 2, []float64{1, 1, 1, 1})
	columnVector := mat.NewVecDense(2, []float64{1, 2})
	expectedResult := mat.NewDense(2, 2, []float64{2, 3, 2, 3})

	gotResult, err := main.AddRowVector(matrix, columnVector)
	if err != nil {
		t.Error(err)
	}

	err = compareDense(expectedResult, gotResult)
	if err != nil {
		t.Error(err)
	}
}

func TestSubtractColumnVector(t *testing.T) {
	matrix := mat.NewDense(2, 2, []float64{1, 1, 1, 1})
	columnVector := mat.NewVecDense(2, []float64{1, 2})
	expectedResult := mat.NewDense(2, 2, []float64{0, -1, 0, -1})

	gotResult, err := main.SubtractRowVector(matrix, columnVector)
	if err != nil {
		t.Error(err)
	}

	err = compareDense(expectedResult, gotResult)
	if err != nil {
		t.Error(err)
	}
}

func TestGetColumnStdDev(t *testing.T) {
	matrix := mat.NewDense(4, 4, []float64{4, 2, 3, 5, 4, 2, 3, 5, -4, -2, -3, -5, -4, -2, -3, -5})
	expectedStdDev := mat.NewVecDense(4, []float64{4, 2, 3, 5})

	mean := main.GetColumnMean(matrix)
	gotStdDev := main.GetColumnStdDev(matrix, mean)

	err := compareDense(expectedStdDev, gotStdDev)
	if err != nil {
		t.Error(err)
	}
}

func TestGetColumnStdDevLobsided(t *testing.T) {
	matrix := mat.NewDense(4, 2, []float64{4, 2, 4, 2, -4, -2, -4, -2})
	expectedStdDev := mat.NewVecDense(2, []float64{4, 2})

	mean := main.GetColumnMean(matrix)
	gotStdDev := main.GetColumnStdDev(matrix, mean)

	err := compareDense(expectedStdDev, gotStdDev)
	if err != nil {
		t.Error(err)
	}
}

func TestDivideColumnVectorLobsided(t *testing.T) {
	matrix := mat.NewDense(4, 2, []float64{4, 2, 4, 2, -4, -2, -4, -2})
	divisor := mat.NewVecDense(2, []float64{4, 2})
	expectedResult := mat.NewDense(4, 2, []float64{1, 1, 1, 1, -1, -1, -1, -1})

	gotResult, err := main.DivideColumnVector(matrix, divisor)
	if err != nil {
		t.Error(err)
	}

	err = compareDense(expectedResult, gotResult)
	if err != nil {
		t.Error(err)
	}
}

func TestHadamardProduct(t *testing.T) {
	a := mat.NewDense(1, 2, []float64{1, 2})
	b := mat.NewDense(1, 2, []float64{3, 4})
	expectedResult := mat.NewDense(1, 2, []float64{3, 8})

	gotResult, err := main.HadamardProduct(a, b)
	if err != nil {
		t.Error(err)
	}

	err = compareDense(expectedResult, gotResult)
	if err != nil {
		t.Error(err)
	}
}
