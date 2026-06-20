#include <iostream>
#include <filesystem>

#include "mnist_loader.hpp"

int main() {
    MNISTDataset data = MNISTDataset::load(
        (std::filesystem::path("data") / "train-images-idx3-ubyte").string(),
        (std::filesystem::path("data") / "train-labels-idx1-ubyte").string()
    );
    std::cout << "Magic number: " << data.magic_number << std::endl;
    std::cout << "Number of images: " << data.number_of_images << std::endl;
    std::cout << "Rows: " << data.rows << std::endl;
    std::cout << "Columns: " << data.columns << std::endl;
    std::cout << "Labels magic number: " << data.labels_magic_number << std::endl;
    std::cout << "Number of labels: " << data.number_of_labels << std::endl;
    std::cout << "Labels: " << data.labels.size() << std::endl;
    for (int i=0;i<data.labels.size();i++){
        std::cout << "Label " << i << ": " << (int)data.labels[i] << std::endl;
    }
    return 0;
}
