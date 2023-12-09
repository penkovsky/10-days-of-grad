# Automatic differentiation

Previously we have calculated our backpropagation by hand.  Now, we think
how this process can be simplified with automatic differentiation.  [The
tutorial](https://penkovsky.com/neural-networks/day3/) about neural
networks.


## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Run interactively

     ```
     $ stack --install-ghc ghci --resolver lts-11.9 \
     --package backprop-0.2.2.0 --package random \
     haskell-guide-to-neural-networks.lhs
     ```

     ```
     > main
     ```

```
> forwardMultiplyGate (-2) 3
-6.0

> localSearch 0.01 (-2, 3, inf_) (3 times)
(-1.8033467421002856,2.8876934853094722,-5.207512638917056)
(-1.7821631869698582,2.8577273328918866,-5.0929364510774775)
(-1.6720792902677883,2.9497743119517112,-4.932256537978371)

Testing automatic differentiation:
gradients for parameters a, b, c, x, and y:

> neuron1 [(1, 0), (2, 1), (-3, 0), (-1, 0), (3, 0)]
(0.8807970779778823,0.3149807562105195)
> neuron1 [(1, 0), (2, 1), (-3, 0), (-1, 0), (3, 0)]
(0.8807970779778823,0.3149807562105195)
> [(1, 0), (2, 0), (-3, 1), (-1, 0), (3, 0)]
(0.8807970779778823,0.1049935854035065)
> neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 1), (3, 0)]
(0.8807970779778823,0.1049935854035065)
> neuron1 [(1, 0), (2, 0), (-3, 0), (-1, 0), (3, 1)]
(0.8807970779778823,0.209987170807013)

Testing backprop library
> forwardNeuron [1, 2, (-3), (-1), 3]
0.8807970779778823

> backwardNeuron [1, 2, (-3), (-1), 3]
[-0.1049935854035065,0.3149807562105195,0.1049935854035065,0.1049935854035065,0.209987170807013]
```
