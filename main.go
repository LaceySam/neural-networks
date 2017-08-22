package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	Weight             *mat.Dense
	Bias               *mat.Dense
	activationFunction ActivationFunction
	weightInitialiser  Initialiser
	biasInitialiser    Initialiser
}

func (l *Layer) Seed() {
	Seed(l.Weight, l.weightInitialiser)
	Seed(l.Bias, l.biasInitialiser)
}

type NeuralNetwork struct {
	HiddenLayerCount             int
	HiddenLayerSize              int
	Layers                       []*Layer
	InputCount                   int
	OutputCount                  int
	LossFunction                 LossFunction
	CompressionMean              float64
	CompressionStandardDeviation float64
	BatchSize                    int
	DataSet                      *DataSet
}

func NewNeuralNetwork(
	hiddenLayerCount int,
	hiddenLayerSize int,
	lossFunction LossFunction,
) *NeuralNetwork {
	return &NeuralNetwork{
		HiddenLayerCount: hiddenLayerCount,
		HiddenLayerSize:  hiddenLayerSize,
		Layers:           []*Layer{},
		LossFunction:     lossFunction,
	}
}

func (
	nn *NeuralNetwork,
) AddDenseLayer(
	activationFunction ActivationFunction,
	weightInitialiser Initialiser,
	biasInitialiser Initialiser,
) bool {
	var r, c int
	if len(nn.Layers) == 0 {
		// Input layer
		r = nn.DataSet.InputCount
		c = nn.HiddenLayerSize
	} else if len(nn.Layers) == nn.HiddenLayerCount+1 {
		// Output layer
		r = nn.HiddenLayerSize
		c = nn.DataSet.OutputCount
	} else if len(nn.Layers) > nn.HiddenLayerCount+1 {
		// Neural network already built...maybe panic instead?
		return false
	} else {
		// Add another hidden layer
		r = nn.HiddenLayerSize
		c = nn.HiddenLayerSize
	}

	layer := &Layer{
		Weight:             mat.NewDense(r, c, nil),
		Bias:               mat.NewDense(1, c, nil),
		activationFunction: activationFunction,
		weightInitialiser:  weightInitialiser,
		biasInitialiser:    biasInitialiser,
	}

	layer.Seed()

	nn.Layers = append(nn.Layers, layer)

	return true
}

func (nn *NeuralNetwork) Forward(X *mat.Dense) (*mat.Dense, error) {

	//var result *mat.Dense
	var err error

	for i := 0; i < len(nn.Layers); i++ {
		layer := nn.Layers[i]
		X, err = MultiplyMatrices(X, layer.Weight)
		if err != nil {
			return nil, err
		}

		X, err = AddRowToMatrix(X, layer.Bias)
		if err != nil {
			return nil, err
		}

		r, c := X.Dims()
		fmt.Printf("BATCH: ROWS: %d, COLS: %d\n", r, c)

	}

	return X, nil
}

func (nn *NeuralNetwork) TrainBatch(X, y *mat.Dense) error {
	X, _ = nn.Forward(X)
	loss := nn.LossFunction(X, y)
	fmt.Println("Loss", loss)

	return nil
}

func (nn *NeuralNetwork) TrainEpochs(epochs int) error {

	for i := 0; i < epochs; i++ {
		for !nn.DataSet.Completed() {
			X, y := nn.DataSet.GetBatch()

			err := nn.TrainBatch(X, y)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (nn *NeuralNetwork) LoadData(batchSize int, data []byte) error {
	nn.DataSet = &DataSet{}
	err := json.Unmarshal(data, nn.DataSet)
	if err != nil {
		return err
	}

	nn.DataSet.Setup(batchSize)
	return nil
}

func main() {

	file, e := ioutil.ReadFile("./data.json")
	if e != nil {
		fmt.Printf("File error: %v\n", e)
		os.Exit(1)
	}

	hiddenLayers := 4
	hiddenLayerSize := 30
	batchSize := 27
	nn := NewNeuralNetwork(hiddenLayers, hiddenLayerSize, L2Loss)
	nn.LoadData(batchSize, file)

	o := NewOnes()
	for i := 0; i < hiddenLayers+2; i++ {
		var g *GlorotNormal
		if i == 0 {
			g = NewGlorotNormal(nn.DataSet.InputCount, hiddenLayerSize)
		} else if i == (hiddenLayers + 1) {
			g = NewGlorotNormal(hiddenLayerSize, nn.DataSet.OutputCount)
		} else {
			g = NewGlorotNormal(hiddenLayerSize, hiddenLayerSize)
		}

		nn.AddDenseLayer(Relu, g, o)
	}

	nn.TrainEpochs(1)
}
