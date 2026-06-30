# Neural Network MNIST Classifier

A C++ feedforward neural network trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset, with a [raylib](https://www.raylib.com/) GUI for drawing digits and running live predictions.

The network is implemented from scratch using [Eigen](https://eigen.tuxfamily.org/).

### Flow Chart
![Flow Chart](assets/Neural_Net_Flowchart(2).png)

### Video Explanation
![img src="/assets/neural_net_vid.png"](https://youtu.be/fKm7jqmdumg)

## Features

- **Custom MLP** with configurable layer sizes, ReLU hidden layers, and softmax output
- **Cross-entropy training** with mini-batch gradient descent (batch size 32)
- **Early stopping** on a held-out validation split with best-weight checkpointing
- **Model persistence** — save and load weights to `data/best_model.bin`
- **Interactive GUI** — draw digits with the mouse, preprocess like MNIST, and classify in real time
- **Background training** — model loads or trains on a worker thread while the window stays responsive


## Architecture

Default network: **784 → 256 → 64 → 10**

Input:28×28 grayscale image
Hidden layers:ReLU function
Output layer:softmax function
Loss:Cross-entropy
Optimizer:stochastic gradient descent

Training uses 48000 MNIST images and 12000 are left for validation. Test accuracy is reported on the official 10000 image test set.

## Hand-drawn digit preprocessing

To match MNIST normalization image processing:

1. Capture the 280×280 drawing canvas
2. Find the bounding box of white ink pixels
3. Crop to that box
4. Scale to fit inside a 20×20 region (aspect ratio preserved)
5. Paste into a 28×28 black image, centered by center of mass at (13.5, 13.5)
6. Feed the resulting vector to the network

The right panel in the GUI shows the processed 28×28 input the network actually sees.


## Build

```bash
cmake -B build
cmake --build build
```

The executable is `build/main`.

## Run

```bash
./build/main
```
You can choose to either load a saved model from `data/best_model.bin` or train your own network by editing line 23 of `src/main.cpp`: `NeuralNet net({784, 256, 64, 10}, 32, true, true)`. You can change the layer size list (first entry must be `784`, last must be `10`), batch size (`32`), and the two flags at the end. Set **`save_read`** to `true` to read weights from file on startup (skipped if the file is missing or the layer list does not match), or `false` to always reinitialize and train; set **`save_write`** to `true` to overwrite and save `data/best_model.bin` on quit when training improves validation accuracy, or `false` to leave the file unchanged.

Wait for **"Training not done, please wait"** to disappear before submitting a drawing.

## Controls


**Left mouse drag**: Draw on the canvas (white on black) 
**Enter**: Preprocess drawing and run prediction 
**C**: Clear the canvas 
**Q**: Quit (saves model weights on exit) 

Predictions appear below the processed-image preview on the right.


## Model file format

`data/best_model.bin` format is as follows:

```
[float]  best validation accuracy
[int]    number of layer sizes
[int]    layer_sizes[]        
per layer:
    [double] weights
    [double] biases
```
