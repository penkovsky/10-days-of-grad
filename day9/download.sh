#!/usr/bin/env bash
set -euo pipefail

datadir=data
trainset=train-images-idx3-ubyte
trainlab=train-labels-idx1-ubyte
testset=t10k-images-idx3-ubyte
testlab=t10k-labels-idx1-ubyte

# yann.lecun.com/exdb/mnist now returns 404s; use PyTorch's S3 mirror.
url=https://ossci-datasets.s3.amazonaws.com/mnist

mkdir -p $datadir
echo "Attempting to download MNIST data"
for f in $trainset $trainlab $testset $testlab; do
  echo "  $f.gz"
  curl --fail --location -o "$datadir/$f.gz" "$url/$f.gz"
  # Reject error pages: the file must be a real gzip stream.
  if ! gzip -t "$datadir/$f.gz" 2>/dev/null; then
    echo "ERROR: $datadir/$f.gz is not a valid gzip file (download failed)." >&2
    exit 1
  fi
done
echo "MNIST data downloaded to $datadir/"
