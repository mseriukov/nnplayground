#!/usr/bin/env zsh

MNIST_DIR="../models/mnist"
FILES=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

mkdir -p "$MNIST_DIR"
pushd "$MNIST_DIR" || exit 1

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        curl -OL "https://raw.githubusercontent.com/fgnt/mnist/master/$file.gz"
        gunzip "$file.gz"
    fi
done

popd