<p align="center">
  <img src="https://user-images.githubusercontent.com/22550143/159121969-05822663-73d9-4439-ac7c-3f1e3e50cf83.svg#gh-light-mode-only" width="300px" height="300px">
  <img src="https://user-images.githubusercontent.com/22550143/159121951-4cca90be-c08b-4e32-8f8c-0724812a545f.svg#gh-dark-mode-only" width="300px" height="300px">
</p>

<hr />
![Unit Tests](https://github.com/krylowicz/tinynet/workflows/Unit%20Tests/badge.svg) <br />
tinynet is yet another deep learning framework built solely for the purpose of learning about pytorch, autograd technique and deep learning

## Installation
```
pip3 install git+https://github.com/krylowicz/tinynet.git --upgrade

# for development
git clone https://github.com/krylowicz/tinynet.git
cd tinynet
python setup.py develop
```

## How?
<p align="center">
  <img src="https://user-images.githubusercontent.com/22550143/159122001-658f38d0-3a39-4d47-848a-402bdfec31b7.svg#gh-light-mode-only" width="550px" height="550px">
  <img src="https://user-images.githubusercontent.com/22550143/159121985-4cd03924-e050-45a3-9a47-0c81cb2639ed.svg#gh-dark-mode-only" width="550px" height="550px">

I am planning to create a second Tensor class that is a wrapper around my other [tiny](http://github.com/krylowicz/tinydot) project

### Functions
tinynet supports (not yet) these 16 ops, both on cpu and gpu (using PyOpenCL)
```
Relu, Log, Exp                  # unary ops
Add, Sub, Mul, Pow, Matmul      # binary ops
Sum, Max                        # reduce ops
Reshape, Transpose, Slice       # movement ops
Matmul, Conv1D, Conv2D          # processing ops
```
The rest of the ops can be implemented as a combination of others.