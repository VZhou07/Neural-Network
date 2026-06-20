#include "neural_net.hpp"

Layer::Layer(int current_layer_size, int next_layer_size) : current_layer_size(current_layer_size), next_layer_size(next_layer_size) {
    weights = std::vector<std::vector<double>>(next_layer_size, std::vector<double>(current_layer_size));
    biases = std::vector<double>(next_layer_size);
    initialize_weights_and_biases(-1.0, 1.0);
}

void Layer::initialize_weights_and_biases (double min, double max) {
    for (int i=0;i<next_layer_size;i++){
        for (int j=0;j<current_layer_size;j++){
            this->weights[i][j] = (double)std::rand()/(double)RAND_MAX*(max-min)+min;
        }
        this->biases[i] = 0.0;
    }
}