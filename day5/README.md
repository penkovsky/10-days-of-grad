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
1 Training accuracy 10.4  Validation accuracy 10.3
2 Training accuracy 85.7  Validation accuracy 86.3
3 Training accuracy 95.9  Validation accuracy 96.2
4 Training accuracy 97.6  Validation accuracy 97.4
5 Training accuracy 98.4  Validation accuracy 98.1
6 Training accuracy 98.3  Validation accuracy 98.0
7 Training accuracy 99.0  Validation accuracy 98.8
8 Training accuracy 99.0  Validation accuracy 98.7
9 Training accuracy 99.1  Validation accuracy 98.7
10 Training accuracy 99.2  Validation accuracy 98.7
11 Training accuracy 99.3  Validation accuracy 98.8
12 Training accuracy 98.0  Validation accuracy 97.6
13 Training accuracy 99.3  Validation accuracy 98.7
14 Training accuracy 99.5  Validation accuracy 98.7
15 Training accuracy 99.3  Validation accuracy 98.6
16 Training accuracy 99.5  Validation accuracy 98.7
17 Training accuracy 99.6  Validation accuracy 98.8
18 Training accuracy 99.6  Validation accuracy 98.7
19 Training accuracy 99.5  Validation accuracy 98.7
20 Training accuracy 99.6  Validation accuracy 98.8
21 Training accuracy 99.7  Validation accuracy 98.8
22 Training accuracy 99.6  Validation accuracy 98.7
23 Training accuracy 99.8  Validation accuracy 98.9
24 Training accuracy 99.7  Validation accuracy 98.8
25 Training accuracy 99.8  Validation accuracy 98.9
26 Training accuracy 99.8  Validation accuracy 99.0
27 Training accuracy 99.8  Validation accuracy 98.7
28 Training accuracy 99.9  Validation accuracy 98.9
29 Training accuracy 99.7  Validation accuracy 98.6
30 Training accuracy 99.9  Validation accuracy 98.9
```
