package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Layer struct {
	Weight                  *mat.Dense
	Bias                    *mat.Dense
	ActivationFunction      ActivationFunction
	ActivationFunctionPrime ActivationFunctionPrime
	weightInitialiser       Initialiser
	biasInitialiser         Initialiser
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
	CompressionMean              *mat.VecDense
	CompressionStandardDeviation *mat.VecDense
	BatchSize                    int
	DataSet                      *DataSet
	LearningRate                 float64
}

func NewNeuralNetwork(
	hiddenLayerCount int,
	hiddenLayerSize int,
	lossFunction LossFunction,
	learningRate float64,
) *NeuralNetwork {
	return &NeuralNetwork{
		HiddenLayerCount: hiddenLayerCount,
		HiddenLayerSize:  hiddenLayerSize,
		Layers:           []*Layer{},
		LossFunction:     lossFunction,
		LearningRate:     learningRate,
	}
}

func (
	nn *NeuralNetwork,
) AddDenseLayer(
	activationFunction ActivationFunction,
	ActivationFunctionPrime ActivationFunctionPrime,
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
		Weight:                  mat.NewDense(r, c, nil),
		Bias:                    mat.NewDense(1, c, nil),
		ActivationFunction:      activationFunction,
		ActivationFunctionPrime: ActivationFunctionPrime,
		weightInitialiser:       weightInitialiser,
		biasInitialiser:         biasInitialiser,
	}

	layer.Seed()

	nn.Layers = append(nn.Layers, layer)

	return true
}

func (nn *NeuralNetwork) Forward(X *mat.Dense) (*mat.Dense, []*mat.Dense, error) {
	var err error
	Xsteps := []*mat.Dense{X}

	for i := 0; i < len(nn.Layers); i++ {
		layer := nn.Layers[i]

		X, err = MultiplyMatrices(X, layer.Weight)
		if err != nil {
			return nil, nil, err
		}

		if layer.Bias != nil {
			X, err = AddRowToMatrix(X, layer.Bias)
			if err != nil {
				return nil, nil, err
			}
		}

		// Apply the layers activation function
		X.Apply(nn.Layers[i].ActivationFunction, X)
		Xsteps = append(Xsteps, X)

		r, c := X.Dims()
		fmt.Printf("BATCH: ROWS: %d, COLS: %d\n", r, c)
	}

	return X, Xsteps, nil
}

func (nn *NeuralNetwork) Backward(Xsteps []*mat.Dense, X, y *mat.Dense) error {

	cost := mat.DenseCopyOf(Xsteps[len(Xsteps)-1])
	cost.Sub(Xsteps[len(Xsteps)-1], y)

	// Need differential error
	newWeights := []*mat.Dense{}

	for i := len(nn.Layers) - 1; i >= 0; i-- {
		delta := mat.DenseCopyOf(Xsteps[i+1])

		delta.Apply(nn.Layers[i].ActivationFunctionPrime, delta)
		delta.MulElem(delta, cost)

		weightCost, err := MultiplyMatrices(Transpose(Xsteps[i]), delta)
		if err != nil {
			fmt.Println("mult", err)
		}

		weightCost.Scale(nn.LearningRate, weightCost)
		weightCost.Sub(nn.Layers[i].Weight, weightCost)

		newWeights = append([]*mat.Dense{weightCost}, newWeights...)

		cost, err = MultiplyMatrices(delta, Transpose(nn.Layers[i].Weight))
		if err != nil {
			fmt.Println("mult2", err)
		}

	}

	for i := len(nn.Layers) - 1; i > 0; i-- {
		if i == len(nn.Layers)-1 {
			//fmt.Println("NEW", i, ":", newWeights[i])
		}
		nn.Layers[i].Weight = newWeights[i]
	}

	return nil
}

func (nn *NeuralNetwork) TrainBatch(X, y *mat.Dense) error {
	var Xsteps []*mat.Dense
	X, Xsteps, _ = nn.Forward(X)
	loss := nn.LossFunction(X, y)
	fmt.Println("Loss", loss)
	count, _ := loss.Dims()
	fmt.Println("Mean loss", mat.Sum(loss)/float64(count))

	nn.Backward(Xsteps, X, y)

	return nil
}

func (nn *NeuralNetwork) TrainEpochs(epochs int) error {

	for i := 0; i < epochs; i++ {
		fmt.Printf("Epoch %d, Completed: %v\n\n\n\n\n\n", i, nn.DataSet.Completed())

		z := 0
		for !nn.DataSet.Completed() {
			X, y := nn.DataSet.GetBatch()

			err := nn.TrainBatch(X, y)
			if err != nil {
				return err
			}

			z++
		}

		time.Sleep(time.Second)
		nn.DataSet.Reset()

	}

	return nil
}

func (nn *NeuralNetwork) LoadData(batchSize int, data []byte) error {
	nn.DataSet = &DataSet{}
	err := json.Unmarshal(data, nn.DataSet)
	if err != nil {
		return err
	}

	nn.CompressionMean, nn.CompressionStandardDeviation = nn.DataSet.Setup(batchSize)
	return nil
}

func main() {

	file, e := ioutil.ReadFile("./data.json")
	if e != nil {
		fmt.Printf("File error: %v\n", e)
		os.Exit(1)
	}

	hiddenLayers := 1
	hiddenLayerSize := 3
	batchSize := 1
	learningRate := 0.1
	nn := NewNeuralNetwork(hiddenLayers, hiddenLayerSize, L2Loss, learningRate)
	nn.LoadData(batchSize, file)

	o := NewZeros()
	for i := 0; i < hiddenLayers+2; i++ {
		var g *GlorotNormal
		if i == 0 {
			g = NewGlorotNormal(nn.DataSet.InputCount, hiddenLayerSize)
		} else if i == (hiddenLayers + 1) {
			g = NewGlorotNormal(hiddenLayerSize, nn.DataSet.OutputCount)
		} else {
			g = NewGlorotNormal(hiddenLayerSize, hiddenLayerSize)
		}

		nn.AddDenseLayer(Relu, ReluPrime, g, o)
	}

	nn.TrainEpochs(2)
}
