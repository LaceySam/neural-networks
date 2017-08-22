package main

import (
	"fmt"

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
