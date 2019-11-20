# Convolutional Neural Networks

![Handwritten digit](http://penkovsky.com/img/posts/mnist/mnist-five.png)

This project's aim is to build a classical convolutional neural network (LeNet)
in Haskell and test it on handwritten digits recognition task. Relevant
documentation can be found in
[the tutorial about convolutional neural networks](http://penkovsky.com/neural-networks/day5).

## How To Build

1. Install stack:

     ```
     $ wget -qO- https://get.haskellstack.org/ | sh
     ```

(alternatively, `curl -sSL https://get.haskellstack.org/ | sh`)

2. Build and run MNIST benchmark on all available cores:

     ```
     $ ./run.sh
     ```

```
1 Training accuracy 86.6  Validation accuracy 87.3
2 Training accuracy 96.6  Validation accuracy 96.6
3 Training accuracy 98.2  Validation accuracy 98.1
4 Training accuracy 98.7  Validation accuracy 98.4
5 Training accuracy 98.9  Validation accuracy 98.4
6 Training accuracy 99.1  Validation accuracy 98.5
7 Training accuracy 99.2  Validation accuracy 98.6
8 Training accuracy 99.3  Validation accuracy 98.6
9 Training accuracy 99.5  Validation accuracy 98.7
10 Training accuracy 99.5  Validation accuracy 98.8
11 Training accuracy 99.5  Validation accuracy 98.7
12 Training accuracy 99.6  Validation accuracy 98.9
13 Training accuracy 99.6  Validation accuracy 98.8
14 Training accuracy 99.7  Validation accuracy 98.8
15 Training accuracy 99.7  Validation accuracy 98.9
16 Training accuracy 99.7  Validation accuracy 98.8
17 Training accuracy 99.7  Validation accuracy 99.0
18 Training accuracy 99.6  Validation accuracy 98.9
19 Training accuracy 99.6  Validation accuracy 98.8
20 Training accuracy 99.9  Validation accuracy 99.0
21 Training accuracy 99.8  Validation accuracy 99.0
22 Training accuracy 99.9  Validation accuracy 98.8
23 Training accuracy 99.8  Validation accuracy 98.8
24 Training accuracy 99.8  Validation accuracy 99.0
25 Training accuracy 99.8  Validation accuracy 99.0
26 Training accuracy 99.9  Validation accuracy 99.1
27 Training accuracy 99.8  Validation accuracy 98.9
28 Training accuracy 99.9  Validation accuracy 99.0
29 Training accuracy 100.0  Validation accuracy 99.1
30 Training accuracy 99.9  Validation accuracy 99.0
```
