from __future__ import print_function
import torch
import torch.nn as nn
import time

net1 = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, bias=False)
net2 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=0, bias=False)

bs = 64
N = 1000

# CPU
fm0 = [torch.randn(bs, 1, 28, 28) for i in range(N)]
fm1 = [torch.randn(bs, 3, 28, 28) for i in range(N)]

start = time.process_time()
for i in range(N):
    r = net1(fm0[i])
end = time.process_time()
print(1000 * (end - start)/N, "ms")
# Dell Precision 5820 desktop (all 12 cores):
#
# PyTorch 0.4.1
# 2.0470764999999997 ms

# PyTorch 1.2.0
# 2.1088856049999998 ms

start = time.process_time()
for i in range(N):
    r = net2(fm1[i])
end = time.process_time()
print(1000 * (end - start)/N, "ms")

# PyTorch 0.4.1
# 4.599914508 ms

# PyTorch 1.2.0
# 1.9183940690000005 ms
