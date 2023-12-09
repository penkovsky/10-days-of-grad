# Hasktorch MNIST

[The documentation](http://penkovsky.com/neural-networks/day7).

## How To Run

First, download MNIST data:

    $ ./download.sh

Next, launch Hasktorch Docker container

    docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/home/ubuntu/data htorch/hasktorch-jupyter:latest-cu11

Finally, open localhost:8888 and find the `MNIST.ipynb` notebook in data/.
