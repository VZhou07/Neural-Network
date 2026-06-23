#pragma once
#include <Eigen/Dense>
#include <mnist_loader.hpp>

class Layer{
    public:
    Layer (int previous_layer_size, int next_layer_size, bool is_output_layer=false, int batchSize=32);
    void forward(Eigen::MatrixXd& A_prev);
    Eigen::MatrixXd backward(Eigen::MatrixXd& dA, Eigen::MatrixXd& activations_prev_layer, float learning_rate);
    bool is_output_layer;
    int previous_layer_size;
    int current_layer_size;
    int batchSize;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::MatrixXd activations;
    Eigen::MatrixXd pre_activations;
    Eigen::MatrixXd A_prev;
    Eigen::MatrixXd dZ;
    Eigen::MatrixXd dW;
    Eigen::VectorXd db;
    Eigen::MatrixXd best_weights;
    Eigen::VectorXd best_biases;
    void sigmoid();
};    

class NeuralNet{
    public:
    NeuralNet(std::vector<int> layer_sizes, int batchSize=32);
    float train(float learning_rate);
    int input_size;
    std::vector<int> layer_sizes;
    int output_size;
    float best_accuracy;
    bool keep_training;
    float epoch_accuracy;
    std::vector<float> epoch_accuracies;
    int epoch;
    int batchSize;
    std::vector<Layer> layers;
    MNISTDataset dataset;
    int batch_counter;
};    