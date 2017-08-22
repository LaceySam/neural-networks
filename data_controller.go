package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type DataSet struct {
	Input           [][]float64 `json:"input"`
	Output          [][]float64 `json:"output"`
	currentLocation int
	BatchSize       int
	dataLength      int
	InputCount      int
	OutputCount     int
}

func (d *DataSet) Setup(BatchSize int) {
	inLength := len(d.Input)
	outLength := len(d.Output)

	if inLength != outLength {
		panic(fmt.Sprintf("Data mismatch, input %d, output %d", d.Input, d.Output))
	}

	d.BatchSize = BatchSize
	d.dataLength = inLength
	d.InputCount = len(d.Input[0])
	d.OutputCount = len(d.Output[0])
}

func (d *DataSet) Reset() {
	d.currentLocation = 0
}

func (d *DataSet) Completed() bool {
	return d.dataLength == d.currentLocation
}

func (d *DataSet) GetBatch() (*mat.Dense, *mat.Dense) {
	stop := d.currentLocation + d.BatchSize

	XBatch := []float64{}
	YBatch := []float64{}
	for i := d.currentLocation; i < stop; i++ {
		for j := 0; j < len(d.Input[i]); j++ {
			XBatch = append(XBatch, d.Input[i][j])
		}

		for j := 0; j < len(d.Output[i]); j++ {
			YBatch = append(YBatch, d.Output[i][j])
		}

		if i >= d.dataLength-1 {
			stop = d.dataLength
			break
		}
	}

	currentBatchSize := stop - d.currentLocation
	d.currentLocation = stop
	return mat.NewDense(currentBatchSize, d.InputCount, XBatch), mat.NewDense(currentBatchSize, d.OutputCount, YBatch)
}
