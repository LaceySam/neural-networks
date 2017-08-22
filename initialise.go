package main

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Initialiser interface {
	Generate() float64
}

type Zeros struct{}

func (z *Zeros) Generate() float64 {
	return 0.0
}

func NewZeros() *Zeros {
	return &Zeros{}
}

type Ones struct{}

func (o *Ones) Generate() float64 {
	return 1.0
}

func NewOnes() *Ones {
	return &Ones{}
}

type Constant struct {
	value float64
}

func (c *Constant) Generate() float64 {
	return c.value
}

func NewConstant() *Constant {
	return &Constant{}
}

type GlorotNormal struct {
	stddev float64
}

func NewGlorotNormal(in, out int) *GlorotNormal {
	return &GlorotNormal{stddev: math.Sqrt(2 / (float64(in) + float64(out)))}
}

func (g *GlorotNormal) Generate() float64 {
	return rand.NormFloat64() * g.stddev
}

func Seed(x *mat.Dense, initialiser Initialiser) {
	rows, cols := x.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			x.Set(i, j, initialiser.Generate())
		}
	}
}
