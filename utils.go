package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func MultiplyMatrices(x, y *mat.Dense) (*mat.Dense, error) {
	xRows, xCols := x.Dims()
	yRows, yCols := y.Dims()

	if xCols != yRows {
		err := fmt.Errorf("X Cols (%d) != Y Rows (%d), can't calculate product", xCols, yRows)
		return nil, err
	}

	result := []float64{}
	for i := 0; i < xRows; i++ {
		for j := 0; j < yCols; j++ {
			xRow := x.RowView(i)
			yCol := y.ColView(j)
			result = append(result, mat.Dot(xRow, yCol))
		}
	}

	return mat.NewDense(xRows, yCols, result), nil
}

func AddRowToMatrix(matrix, row *mat.Dense) (*mat.Dense, error) {
	matrixRows, matrixCols := matrix.Dims()
	rowRows, rowCols := row.Dims()

	if rowRows > 1 {
		err := fmt.Errorf("Row has more than 1 row")
		return nil, err
	}

	if rowCols != matrixCols {
		err := fmt.Errorf("matrix cols (%d) != row cols (%d), can't calculate add", matrixCols, rowCols)
		return nil, err
	}

	result := []float64{}
	for i := 0; i < matrixRows; i++ {
		for j := 0; j < rowCols; j++ {
			result = append(result, matrix.At(i, j)+row.At(0, j))
		}
	}

	return mat.NewDense(matrixRows, matrixCols, result), nil
}

func AddColumnVector(x *mat.Dense, y *mat.VecDense) (*mat.Dense, error) {
	rows, cols := x.Dims()
	results := []float64{}
	for i := 0; i < cols; i++ {
		column := x.ColView(i)
		for j := 0; j <= rows; j++ {
			results = append(results, column.At(0, j)-y.At(0, j))
		}
	}

	return mat.NewDense(rows, cols, results), nil
}

func DivideColumnVector(x *mat.Dense, y *mat.VecDense) (*mat.Dense, error) {
	rows, cols := x.Dims()
	results := []float64{}
	for i := 0; i < cols; i++ {
		column := x.ColView(i)
		for j := 0; j <= rows; j++ {
			results = append(results, column.At(0, j)/y.At(0, j))
		}
	}

	return mat.NewDense(rows, cols, results), nil
}

func SubtractColumnVector(x *mat.Dense, y *mat.VecDense) (*mat.Dense, error) {
	_, cols := y.Dims()
	z := mat.NewVecDense(cols, nil)
	z.ScaleVec(-1, y)
	return AddColumnVector(x, z)
}

func GetColumnMean(x *mat.Dense) *mat.VecDense {
	rows, cols := x.Dims()
	colMeans := []float64{}
	for i := 0; i < cols; i++ {
		mean := 0.0
		column := x.ColView(i)
		for j := 0; j <= rows; j++ {
			mean = mean + column.At(0, j)
		}

		colMeans = append(colMeans, mean)
	}

	return mat.NewVecDense(cols, colMeans)
}

func CalculateColumnStdDev(x *mat.Dense, mean *mat.VecDense) *mat.VecDense {
	rows, cols := x.Dims()
	colStdDev := []float64{}
	for i := 0; i < cols; i++ {
		sum := 0.0
		column := x.ColView(i)
		for j := 0; j <= rows; j++ {
			sum = sum + math.Pow(column.At(0, j)-mean.At(0, j), 2)
		}

		stdDev := math.Sqrt(sum / (float64(rows) - 1.0))
		colStdDev = append(colStdDev, stdDev)
	}

	return mat.NewVecDense(cols, colStdDev)
}
