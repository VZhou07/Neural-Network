#include "neural_net.hpp"
#include <mnist_loader.hpp>
#include <Eigen/Dense>
#include <filesystem>
#include <cmath>
#include <iostream>

Layer::Layer(int previous_layer_size, int current_layer_size, bool is_output_layer, int batchSize) : previous_layer_size(previous_layer_size), current_layer_size(current_layer_size), is_output_layer(is_output_layer), batchSize(batchSize) {
    weights = Eigen::MatrixXd::Random(current_layer_size, previous_layer_size)*std::sqrt(1.0/previous_layer_size);
    biases = Eigen::VectorXd::Random(current_layer_size);
    activations = Eigen::MatrixXd::Zero(current_layer_size,batchSize);
    pre_activations = Eigen::MatrixXd::Zero(current_layer_size,batchSize);
    best_weights = Eigen::MatrixXd::Random(current_layer_size, previous_layer_size);
    best_biases = Eigen::VectorXd::Random(current_layer_size);
}

void Layer::sigmoid(){
    activations = (1.0 / (1.0 + (-pre_activations.array()).exp())).matrix();
}

void Layer::forward(Eigen::MatrixXd& activations_prev_layer){
    pre_activations = (weights*activations_prev_layer).colwise() + biases;
    sigmoid();
}

Eigen::MatrixXd Layer::backward(Eigen::MatrixXd& dA, Eigen::MatrixXd& activations_prev_layer, float learning_rate){
    this->dZ=(dA.array()*(activations.array()*(1.0-activations.array()))).matrix();
    this->dW=float(1.0/batchSize)*(dZ*activations_prev_layer.transpose());
    this->db=float(1.0/batchSize)*dZ.rowwise().sum();
    Eigen::MatrixXd dA_prev=(weights.transpose()*dZ);
    weights=weights-(learning_rate*dW);
    biases=biases-(learning_rate*db);
    return dA_prev;
}

NeuralNet::NeuralNet(std::vector<int> layer_sizes, int batchSize): layer_sizes(layer_sizes), batchSize(batchSize) {
    for(int i=0;i<layer_sizes.size();i++){
        if(i==0){
        }
        else if (i!=layer_sizes.size()-1){
            layers.push_back(Layer(layer_sizes[i-1], layer_sizes[i], false, batchSize));
        }
        else{
            layers.push_back(Layer(layer_sizes[i-1], layer_sizes[i], true, batchSize));
        }
    }
    this->best_accuracy=-1.0;
    this->epoch_accuracy=0.0;
    this->epoch=0;
    this->batch_counter=0;
    this->dataset=MNISTDataset::load(
        (std::filesystem::path("data") / "train-images-idx3-ubyte").string(),
        (std::filesystem::path("data") / "train-labels-idx1-ubyte").string(),
        (std::filesystem::path("data") / "t10k-images-idx3-ubyte").string(),
        (std::filesystem::path("data") / "t10k-labels-idx1-ubyte").string()
    );
    this->dataset.shuffle_images(this->dataset.size_of_training_set);
    this->epoch_accuracies.push_back(0.0);
    this->epoch_accuracies.push_back(0.0);
    this->keep_training=true;
}

float NeuralNet::train(float learning_rate){
    // while the accuracy is not improving, train the network
    int not_improving=0;
    int epoch_counter=0;
    while (keep_training){
        batch_counter=0;
        while(batch_counter<dataset.size_of_training_set/batchSize){
            Eigen::MatrixXd input_activations = Eigen::MatrixXd::Zero(784, batchSize);
            for(int j=0;j<784;j++){
                for (int k=0;k<batchSize;k++){
                    input_activations(j,k)=dataset.images[(batch_counter*batchSize+k)*784+j]/255.0;
                }
            }
            layers[0].forward(input_activations);
            for(int j=1;j<layers.size();j++){
                layers[j].forward(layers[j-1].activations);
            }
            Eigen::MatrixXd output_activations = layers[layers.size()-1].activations;
            //one hot encode
            Eigen::MatrixXd output_labels = Eigen::MatrixXd::Zero(10, batchSize);
            Eigen::VectorXd y_batch = Eigen::VectorXd::Zero(batchSize);
            for (int j=0;j<batchSize;j++){
                y_batch(j)=dataset.labels[batch_counter*batchSize+j];
            }
            for (int j=0;j<batchSize;j++){
                output_labels(static_cast<int>(y_batch(j)),j)=1.0;
            }
            Eigen::MatrixXd dA=output_activations-output_labels;
            for(int j=layers.size()-1;j>=0;j--){
                if (j==0){
                    dA=layers[j].backward(dA, input_activations, learning_rate);
                }
                else{
                    dA=layers[j].backward(dA, layers[j-1].activations, learning_rate);
                }
            }

            batch_counter++;
        }
        // validate
        int correct=0;
        for (int i=0;i<dataset.size_of_validation_set;i++){
            Eigen::MatrixXd input_activations = Eigen::MatrixXd::Zero(784, 1);
            for(int j=0;j<784;j++){
                input_activations(j,0)=dataset.images[(dataset.size_of_training_set+i)*784+j]/255.0;
            }
            layers[0].forward(input_activations);
            for(int j=1;j<layers.size();j++){
                layers[j].forward(layers[j-1].activations);
            }
            Eigen::MatrixXd output_activations = layers[layers.size()-1].activations;
            int index_of_max_row, index_of_max_col;
            float prediction=output_activations.maxCoeff(&index_of_max_row, &index_of_max_col);
            if (index_of_max_row==static_cast<int>(dataset.labels[dataset.size_of_training_set+i])){
                correct++;
            }
        }
        epoch_accuracy=float(correct)/dataset.size_of_validation_set;
        epoch_accuracies.push_back(epoch_accuracy);
        if (epoch_accuracy>best_accuracy){
            not_improving=0;
            best_accuracy=epoch_accuracy;
            for(int j=0;j<layers.size();j++){
                layers[j].best_weights=layers[j].weights;
                layers[j].best_biases=layers[j].biases;
            }
        }
        else{
            not_improving++;
            if (not_improving>10){
                keep_training=false;
                std::cout<<"Not improving for 10 epochs, stopping training"<<std::endl;
                break;
            }
        }
        epoch_counter++;
        std::cout<<"epoch: "<<epoch_counter<<", accuracy: "<<epoch_accuracy<<std::endl;
    }

    //load best weights and biases
    for(int j=0;j<layers.size();j++){
        layers[j].weights=layers[j].best_weights;
        layers[j].biases=layers[j].best_biases;
    }
    //test
    int correct=0;
    for (int i=0;i<dataset.size_of_test_set;i++){
        Eigen::MatrixXd input_activations = Eigen::MatrixXd::Zero(784, 1);
        for(int j=0;j<784;j++){
            input_activations(j,0)=dataset.test_images[i*784+j]/255.0;
        }
        layers[0].forward(input_activations);
        for(int j=1;j<layers.size();j++){
            layers[j].forward(layers[j-1].activations);
        }
        Eigen::MatrixXd output_activations = layers[layers.size()-1].activations;
        int index_of_max_row, index_of_max_col;
        int prediction=output_activations.maxCoeff(&index_of_max_row, &index_of_max_col);
        if (index_of_max_row==static_cast<int>(dataset.test_labels[i])){
            correct++;
        }
    }
    float test_accuracy=float(correct)/dataset.size_of_test_set;
    return test_accuracy;
}




