<p align="center">
  <img src="https://user-images.githubusercontent.com/22550143/158031439-cddc9686-7ae6-4c5b-b1bb-d76d3c933f88.svg#gh-light-mode-only" width="300px" height="300px">
    <img src="https://user-images.githubusercontent.com/22550143/158031439-cddc9686-7ae6-4c5b-b1bb-d76d3c933f88.svg#gh-dark-mode-only" width="300px" height="300px">
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
  <img src="https://user-images.githubusercontent.com/22550143/158054958-332315a0-8863-4585-a69e-ed60dfc6a597.svg" width="550px" height="550px">
</p>

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
