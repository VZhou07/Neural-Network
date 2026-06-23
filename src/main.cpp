#include <iostream>
#include <raylib.h>
#include "neural_net.hpp"

int main() {
    
    NeuralNet net({784,10, 10}, 32);
    bool success=net.load("data/best_model.bin");
    if (!success){
        net.reinitialize_weights();
        float learning_rate=0.01;
        float accuracy=net.train(learning_rate);
        std::cout << "Accuracy: " << accuracy << std::endl;

    }
    return 0;
}
