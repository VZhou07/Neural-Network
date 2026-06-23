#include <iostream>
#include "neural_net.hpp"

int main() {
    NeuralNet net({784, 10, 10}, 32);
    float learning_rate=0.01;
    float accuracy=net.train(learning_rate);
    std::cout << "Accuracy: " << accuracy << std::endl;
    return 0;
}
