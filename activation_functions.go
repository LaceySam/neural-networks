package main

type ActivationFunction func(int, int, float64) float64

func Relu(i, j int, v float64) float64 {
	if v < 0.0 {
		return 0.0
	}

	return v
}

type ActivationFunctionPrime func(int, int, float64) float64

func ReluPrime(i, j int, v float64) float64 {
	if v < 0.0 {
		return 0.0
	}

	return 1
}
