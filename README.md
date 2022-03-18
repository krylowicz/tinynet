<p align="center">
  <img src="https://user-images.githubusercontent.com/22550143/159065534-86370222-8c2d-47e1-a680-f5b70bc76a3d.svg#gh-light-mode-only" width="300px" height="300px">
  <img src="https://user-images.githubusercontent.com/22550143/159065617-2c5544e3-1d8b-4cc4-b4d2-d626ec9495e8.svg#gh-dark-mode-only" width="300px" height="300px">
</p>

<hr />

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
  <img src="https://user-images.githubusercontent.com/22550143/158054958-332315a0-8863-4585-a69e-ed60dfc6a597.svg#gh-light-mode-only" width="550px" height="550px">
  <img src="https://user-images.githubusercontent.com/22550143/159060876-4ed5ba97-956d-4789-833e-0c43b8c3bee2.svg#gh-dark-mode-only" width="550px" height="550px">

I am planning to create a second Tensor class that is a wrapper around my other [tiny](http://github.com/krylowicz/tinydot) project

### Functions
tinynet supports (not yet) these 14 ops, both on cpu and gpu (using PyOpenCL)
```
Relu, Log, Exp                  # unary ops
Sum, Max                        # reduce ops
Add, Sub, Mul, Pow              # binary ops
Reshape, Transpose, Slice       # movement ops
Matmul, Conv2D                  # processing ops
```
