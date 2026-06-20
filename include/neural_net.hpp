#pragma once


class Layer{
    public:
    Layer (int current_layer_size, int next_layer_size);
    void initialize_weights_and_biases(double min, double max);
    void forward(const std::vector<double>& input);
    void backward(const std::vector<double>& output);
    void update_weights_and_biases(double learning_rate);
private:
    int current_layer_size;
    int next_layer_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};