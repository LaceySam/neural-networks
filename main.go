package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"gonum.org/v1/gonum/mat"
)

type Data struct {
	Input  [][]float64 `json:"input"`
	Output [][]float64 `json:"output"`
}

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
	HiddenLayerCount int
	HiddenLayerSize  int
	Layers           []*Layer
	InputCount       int
	OutputCount      int
}

func NewNeuralNetwork(hiddenLayerCount, hiddenLayerSize, inputCount, outputCount int) *NeuralNetwork {
	return &NeuralNetwork{
		HiddenLayerCount: hiddenLayerCount,
		HiddenLayerSize:  hiddenLayerSize,
		InputCount:       inputCount,
		OutputCount:      outputCount,
		Layers:           []*Layer{},
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
		r = nn.InputCount
		c = nn.HiddenLayerSize
	} else if len(nn.Layers) == nn.HiddenLayerCount+1 {
		// Output layer
		r = nn.HiddenLayerSize
		c = nn.OutputCount
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

func (nn *NeuralNetwork) Forward(data *mat.Dense) (*mat.Dense, error) {

	//var result *mat.Dense
	var err error

	for i := 0; i < len(nn.Layers); i++ {
		layer := nn.Layers[i]
		data, err = MultiplyMatrices(data, layer.Weight)
		if err != nil {
			return nil, err
		}

		data, err = AddRowToMatrix(data, layer.Bias)
		if err != nil {
			return nil, err
		}

	}

	return data, nil

}

func main() {

	file, e := ioutil.ReadFile("./data.json")
	if e != nil {
		fmt.Printf("File error: %v\n", e)
		os.Exit(1)
	}

	data := &Data{}
	json.Unmarshal(file, data)

	hiddenLayers := 4
	hiddenLayerSize := 30
	inputCount := 7
	outputCount := 3
	nn := NewNeuralNetwork(hiddenLayers, hiddenLayerSize, inputCount, outputCount)

	o := NewOnes()
	for i := 0; i < hiddenLayers+2; i++ {
		var g *GlorotNormal
		if i == 0 {
			g = NewGlorotNormal(inputCount, hiddenLayerSize)
		} else if i == (hiddenLayers + 1) {
			g = NewGlorotNormal(hiddenLayerSize, outputCount)
		} else {
			g = NewGlorotNormal(hiddenLayerSize, hiddenLayerSize)
		}

		nn.AddDenseLayer(Relu, g, o)
	}

	slice := data.Input[0:10]
	batchItems := []float64{}
	for _, example := range slice {
		for _, i := range example {
			batchItems = append(batchItems, i)
		}
	}

	batch := mat.NewDense(len(slice), inputCount, batchItems)
	r, c := batch.Dims()
	fmt.Printf("BATCH: ROWS: %d, COLS: %d\n", r, c)

	x, _ := nn.Forward(batch)
	fmt.Println(x)
}
