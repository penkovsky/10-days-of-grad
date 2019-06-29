# Batch normalization

![Handwritten digit](http://penkovsky.com/img/posts/mnist/mnist-five.png)

This project's aim is to demonstrate batch normalization using handwritten
digits recognition task as an example. Relevant documentation is in
[the tutorial about neural networks](http://penkovsky.com/post/neural-networks-4/).

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
10 training epochs

Training accuracy (SGD + batchnorm) 100.0
Validation accuracy (SGD + batchnorm) 98.1

Training accuracy (SGD) 85.7
Validation accuracy (SGD) 85.8
```
