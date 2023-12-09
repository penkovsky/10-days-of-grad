""" Naive Numpy 1.17.1 benchmark (single core), just for fun.

1 channel -> 3 channels
"""
from __future__ import print_function
import numpy as np
import time
print(np.__version__)


def conv_forward(X, W):
    '''
    Based on https://gist.githubusercontent.com/mayankgrwl97/3fa1e7baac12828076c8908125beee72/raw/258fc6913f1f9f4c204e1e8fe0098c46b4b9b9b9/forward_pass.py
    The forward computation for a convolution function

    Arguments:
    X -- output activations of the previous layer, numpy array of shape (n_H_prev, n_W_prev) assuming input channels = 1
    W -- Weights, numpy array of size (f, f) assuming number of filters = 1

    Returns:
    H -- conv output, numpy array of size (n_H, n_W)
    '''

    # Retrieving dimensions from X's shape
    (n_H_prev, n_W_prev) = X.shape

    # Retrieving dimensions from W's shape
    (f, f) = W.shape

    # Compute the output dimensions assuming no padding and stride = 1
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1

    # Initialize the output H with zeros
    H = np.zeros((n_H, n_W))

    # Looping over vertical(h) and horizontal(w) axis of output volume
    for h in range(n_H):
        for w in range(n_W):
            x_slice = X[h:h+f, w:w+f]
            H[h,w] = np.sum(x_slice * W)

    return H

def conv2d(batch, ws):
    bs = batch.shape[0]
    cout = ws.shape[0]
    for i in range(bs):
        image = batch[i, 0]  # Hardcoded channel dim
        # results = []
        for k in range(cout):
            kernel = ws[k, 0]  # Hardcoded channel dim
            conv_forward(image, kernel)

            # No use, just a benchmark
            # results.append(conv_forward(image, kernel))

    return None

bs = 64
N = 50  # It is very slow, don't waste your time on 1000

# Numpy 1.17.1
rng = np.random.default_rng()
# Already 'padded' results to 32x32
fm0 = rng.random((bs, 1, 32, 32))
ws = rng.random((3, 1, 5, 5))

start = time.process_time()
for i in range(N):
    r = conv2d(fm0, ws)
end = time.process_time()
print(1000 * (end - start)/N, "ms")
# Dell Precision 5820 desktop (1 core):
# ~640 ms
