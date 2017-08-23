package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Batch struct {
	X    *mat.Dense
	Y    *mat.Dense
	Size int
}

type DataSet struct {
	Input           [][]float64 `json:"input"`
	Output          [][]float64 `json:"output"`
	input           *mat.Dense
	output          *mat.Dense
	Batches         []Batch
	batchCount      int
	currentLocation int
	BatchSize       int
	dataLength      int
	InputCount      int
	OutputCount     int
}

func (d *DataSet) createBatches() (*mat.VecDense, *mat.VecDense) {
	// Load data into a giant Matrix
	Xvals := []float64{}
	for i := 0; i < len(d.Input); i++ {
		for j := 0; j < len(d.Input[i]); j++ {
			Xvals = append(Xvals, d.Input[i][j])
		}
	}

	input := mat.NewDense(len(d.Input), d.InputCount, Xvals)

	Yvals := []float64{}
	for i := 0; i < len(d.Output); i++ {
		for j := 0; j < len(d.Output[i]); j++ {
			Yvals = append(Yvals, d.Output[i][j])
		}
	}

	output := mat.NewDense(len(d.Output), d.OutputCount, Yvals)

	// Compress
	xCompressionMean := GetColumnMean(input)
	d.input, _ = SubtractRowVector(input, xCompressionMean)
	xCompressionStandardDeviation := GetColumnStdDev(d.input, xCompressionMean)
	d.input, _ = DivideColumnVector(d.input, xCompressionStandardDeviation)

	yCompressionMean := GetColumnMean(output)
	d.output, _ = SubtractRowVector(output, yCompressionMean)
	yCompressionStandardDeviation := GetColumnStdDev(d.output, yCompressionMean)
	d.output, _ = DivideColumnVector(d.output, yCompressionStandardDeviation)

	// Now split big matrixes into batches
	rows, xCols := d.input.Dims()
	_, yCols := d.output.Dims()
	batchNumber := 0
	i := 0
	k := 0
	for k != rows {
		i = batchNumber * d.BatchSize
		k = (batchNumber + 1) * d.BatchSize
		if k > rows {
			k = rows
		}

		sX := d.input.Slice(i, k, 0, xCols)
		sY := d.output.Slice(i, k, 0, yCols)

		X := mat.NewDense(k-i, xCols, nil)
		Y := mat.NewDense(k-i, yCols, nil)

		X.Copy(sX)
		Y.Copy(sY)

		batch := Batch{X: X, Y: Y, Size: k - i}
		d.Batches = append(d.Batches, batch)

		batchNumber++
	}

	d.batchCount = len(d.Batches)

	return yCompressionMean, yCompressionStandardDeviation
}

func (d *DataSet) Setup(BatchSize int) (*mat.VecDense, *mat.VecDense) {
	inLength := len(d.Input)
	outLength := len(d.Output)

	if inLength != outLength {
		panic(fmt.Sprintf("Data mismatch, input %d, output %d", d.Input, d.Output))
	}

	d.BatchSize = BatchSize
	d.dataLength = inLength
	d.InputCount = len(d.Input[0])
	d.OutputCount = len(d.Output[0])
	d.Batches = []Batch{}

	return d.createBatches()
}

func (d *DataSet) Compress() (*mat.VecDense, *mat.VecDense) {
	return nil, nil
}

func (d *DataSet) Reset() {
	d.currentLocation = 0
}

func (d *DataSet) Completed() bool {
	return d.batchCount == d.currentLocation
}

func (d *DataSet) GetBatch() (*mat.Dense, *mat.Dense) {
	batch := d.Batches[d.currentLocation]
	X := mat.NewDense(batch.Size, d.InputCount, nil)
	y := mat.NewDense(batch.Size, d.OutputCount, nil)

	X.Copy(batch.X)
	y.Copy(batch.Y)

	d.currentLocation++
	return X, y
}
