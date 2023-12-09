# Model Uncertainty Estimation

[The documentation](http://penkovsky.com/neural-networks/day8).

## How To Run

First, download MNIST data:

    $ ./download.sh

Next, launch Hasktorch Docker container

    docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/home/ubuntu/data htorch/hasktorch-jupyter:latest-cu11

or if you don't have a GPU

    docker run -it --rm -p 8888:8888 -v $(pwd):/home/ubuntu/data htorch/hasktorch-jupyter:latest-cpu

Finally, open localhost:8888 and find the `Uncertainty.ipynb` notebook in data/.
