# Naive Neural Network Utilities
This repository contains naive implementation of forward path funcitons of DNNs.

Intension is to understand and reproduce fundamental concepts for building hardware specifc libraries on ACAP and FPGAs.

The following are implemented and tested:

* conv2D
* activations
    * relu
    * leaky_relu
    * selu
    * sigmoid
    * tanh
    * swish

Purposefully left the code without vectorization for naiveness.

**To do: C/C++ bindings**

## Test outputs for activations
```bash
DNN functions
inputs :  [ 0.59811103  0.18673458  0.15836048  0.19046645 -0.09635726]
ifm.shape :  (5, 5, 3)
relu   :  [0.59811103 0.18673458 0.15836048 0.19046645 0.        ]
ifm.shape :  (5, 5, 3)
leaky_relu   :  [ 0.59811103  0.18673458  0.15836048  0.19046645 -0.00096357]
ifm.shape :  (5, 5, 3)
selu   :  [ 0.62843584  0.19620221  0.16638951  0.20012329 -0.16149985]
ifm.shape :  (5, 5, 3)
sigmoid   :  [0.64522402 0.54654846 0.53950759 0.54747318 0.47592931]
ifm.shape :  (5, 5, 3)
tanh   :  [ 0.53570405  0.18459397  0.15704983  0.18819618 -0.09606015]
ifm.shape :  (5, 5, 3)
swish   :  [ 0.3859156   0.1020595   0.08543668  0.10427527 -0.04585924]
```

