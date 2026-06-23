#include "mnist_loader.hpp"

#include <fstream>
#include <stdexcept>
#include <bits/stdc++.h>
#include <algorithm>

/*
Constructor for the MNISTDataset class
*/
MNISTDataset::MNISTDataset(std::vector<uint8_t> images, 
                           std::vector<uint8_t> labels, 
                           std::vector<uint8_t> test_images, 
                           std::vector<uint8_t> test_labels) :
                           images(images),
                           labels(labels),
                           test_images(test_images),
                           test_labels(test_labels) {
}

/*
Most CPUs use little-endian but MNIST is stored in big-endian so flip it
*/
uint32_t MNISTDataset::flip_endian(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0xFF000000) >> 24);
}
/*
Fisher Yates block shuffle
*/
void MNISTDataset::shuffle_images(int size_of_dataset){
    size_t numBlocks = size_of_dataset;
    //cuz images are 28x28 and labels are 1 byte
    size_t blockSize_images = 784; 
    size_t blockSize_labels = 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t i = numBlocks - 1; i > 0; i--) {
        size_t j = gen() % (i + 1);
        std::swap_ranges(images.begin()+i*blockSize_images, images.begin()+i*blockSize_images+blockSize_images, images.begin()+j*blockSize_images);
        std::swap_ranges(labels.begin()+i*blockSize_labels, labels.begin()+i*blockSize_labels+blockSize_labels, labels.begin()+j*blockSize_labels);

    }
}


/*
Load the MNIST dataset from a file
*/
MNISTDataset MNISTDataset::load(const std::string& filename, const std::string& labelpath, const std::string& test_filename, const std::string& test_labelpath) {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t rows;
    uint32_t columns;
    uint32_t labels_magic_number;
    uint32_t number_of_labels;
    uint32_t test_magic_number;
    uint32_t number_of_test_images;
    uint32_t test_rows;
    uint32_t test_columns;
    uint32_t test_labels_magic_number;
    uint32_t number_of_test_labels;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = flip_endian(magic_number);
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = flip_endian(number_of_images);
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    rows = flip_endian(rows);
    file.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    columns = flip_endian(columns);
    // rest of bytes are image
    std::vector<uint8_t> images(number_of_images * rows * columns);
    file.read(reinterpret_cast<char*>(images.data()), images.size());
    file.close();
    
    std::ifstream labels_file(labelpath, std::ios::binary);
    if (!labels_file.is_open()) {
        throw std::runtime_error("Failed to open labels file"+labelpath);
    }
    labels_file.read(reinterpret_cast<char*>(&labels_magic_number),sizeof(labels_magic_number));
    labels_magic_number=flip_endian(labels_magic_number);
    labels_file.read(reinterpret_cast<char*>(&number_of_labels),sizeof(number_of_labels));
    number_of_labels=flip_endian(number_of_labels);
    std::vector<uint8_t> labels(number_of_labels);
    labels_file.read(reinterpret_cast<char*>(labels.data()),number_of_labels);
    labels_file.close();

    std::ifstream test_file(test_filename, std::ios::binary);
    if (!test_file.is_open()) {
        throw std::runtime_error("Failed to open test file: " + test_filename);
    }
    test_file.read(reinterpret_cast<char*>(&test_magic_number), sizeof(test_magic_number));
    test_magic_number=flip_endian(test_magic_number);
    test_file.read(reinterpret_cast<char*>(&number_of_test_images), sizeof(number_of_test_images));
    number_of_test_images=flip_endian(number_of_test_images);
    test_file.read(reinterpret_cast<char*>(&test_rows), sizeof(test_rows));
    test_rows=flip_endian(test_rows);
    test_file.read(reinterpret_cast<char*>(&test_columns), sizeof(test_columns));
    test_columns=flip_endian(test_columns);
    std::vector<uint8_t> test_images(number_of_test_images*test_rows*test_columns);
    test_file.read(reinterpret_cast<char*>(test_images.data()),test_images.size());
    test_file.close();

    std::ifstream test_labels_file(test_labelpath, std::ios::binary);
    if (!test_labels_file.is_open()) {
        throw std::runtime_error("Failed to open test labels file: " + test_labelpath);
    }
    test_labels_file.read(reinterpret_cast<char*>(&test_labels_magic_number), sizeof(test_labels_magic_number));
    test_labels_magic_number=flip_endian(test_labels_magic_number);
    test_labels_file.read(reinterpret_cast<char*>(&number_of_test_labels), sizeof(number_of_test_labels));
    number_of_test_labels=flip_endian(number_of_test_labels);
    std::vector<uint8_t> test_labels(number_of_test_labels);
    test_labels_file.read(reinterpret_cast<char*>(test_labels.data()),number_of_test_labels);
    test_labels_file.close();
    return {images, labels, test_images, test_labels};
}
