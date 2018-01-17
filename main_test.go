package main_test

import (
	"testing"

	"gonum.org/v1/gonum/mat"

	"neural-networks"
)

func TestForwardPropagation(t *testing.T) {
	hiddenLayers := 1
	hiddenLayerSize := 2
	learningRate := 0.1
	nn := main.NewNeuralNetwork(hiddenLayers, hiddenLayerSize, main.L2Loss, learningRate)
	nn.OutputCount = 1

	nn.Layers = []*main.Layer{
		&main.Layer{
			Weight:             mat.NewDense(2, 3, []float64{1, 1, 1, 2, 1, 1}),
			ActivationFunction: main.Relu,
		},
	}

	m := mat.NewDense(3, 2, []float64{3, 5, 5, 1, 10, 2})
	gotResult, _, _ := nn.Forward(m)

	expectedResult := mat.NewDense(3, 3, []float64{13, 8, 8, 7, 6, 6, 14, 12, 12})
	err := compareDense(expectedResult, gotResult)
	if err != nil {
		t.Error(err)
	}
}

func TestBackWardPropagation(t *testing.T) {

	hiddenLayers := 2
	hiddenLayerSize := 3
	learningRate := 1
	nn := main.NewNeuralNetwork(hiddenLayers, hiddenLayerSize, main.L2Loss, learningRate)

	nn.OutputCount = 3

	nn.Layers = []*main.Layer{
		&main.Layer{
			Weight:             mat.NewDense(3, 3, []float64{0.1, 0.2, 0.3, 0.3, 0.2, 0.7, 0.4, 0.3, 0.9}),
			ActivationFunction: main.Relu,
		},
		&main.Layer{
			Weight:             mat.NewDense(3, 3, []float64{0.2, 0.3, 0.5, 0.3, 0.5, 0.7, 0.6, 0.4, 0.8}),
			ActivationFunction: main.Relu,
		},
		&main.Layer{
			Weight:             mat.NewDense(3, 3, []float64{0.1, 0.4, 0.8, 0.3, 0.7, 0.2, 0.5, 0.2, 0.9}),
			ActivationFunction: main.Relu,
		},
	}
}
