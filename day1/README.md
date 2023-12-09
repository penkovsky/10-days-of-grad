# Neural network basics

[The tutorial](http://penkovsky.com/post/neural-networks/) about neural networks in Haskell.

## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Install open-blas from https://www.openblas.net/ (needed for hmatrix package)

3. Run

     ```
     $ stack --resolver lts-10.6 --install-ghc runghc --package hmatrix-0.18.2.0 Iris.hs
     ```

```
Initial loss 172.22632766592446
Loss after training 57.831447523758634
Some predictions by an untrained network:
(5><3)
 [ 0.7689030658896214,  3.267038113313446e-2, 0.8739296059573654
 , 0.7629285524847399,  5.015314974962409e-2, 0.8573561277784868
 , 0.7507810216934356,  4.335165787597786e-2, 0.8565466018222087
 , 0.7511138360826368, 5.2565979139007867e-2, 0.8372358524449345
 , 0.7640230289265144, 3.1061429210096483e-2, 0.8708746285872909 ]
Some predictions by a trained network:
(5><3)
 [ 0.7521951759895588, 0.11088638025047737, 0.16321750158553122
 , 0.7168435997393378, 0.14956018384577013, 0.19502771068176775
 , 0.7308700847782362, 0.13048874572295668,  0.1853343482314066
 , 0.6975730028131855,  0.1479627188257003, 0.19303989727800958
 , 0.7534115942423942, 0.10502945965496263,  0.1590179398299323 ]
Targets
(5><3)
 [ 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0
 , 1.0, 0.0, 0.0 ]
```
