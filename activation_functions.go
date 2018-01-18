package main

import (
	"math"
)

type ActivationFunction func(int, int, float64) float64

func Relu(i, j int, v float64) float64 {
	return math.Max(0.0, v)
}

type ActivationFunctionPrime func(int, int, float64) float64

func ReluPrime(i, j int, v float64) float64 {
	if v <= 0 {
		return 0
	}

	return 1
}

func Sigmoid(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func SigmoidPrime(i, j int, v float64) float64 {
	return Sigmoid(i, j, v) * (1 - Sigmoid(i, j, v))
}
