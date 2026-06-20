#include "mnist_loader.hpp"

#include <fstream>
#include <stdexcept>

/*
Constructor for the MNISTDataset class
*/
MNISTDataset::MNISTDataset(uint32_t magic_number, 
                           uint32_t number_of_images,
                           uint32_t rows, 
                           uint32_t columns, 
                           std::vector<uint8_t> images, 
                           uint32_t labels_magic_number, 
                           uint32_t number_of_labels, 
                           std::vector<uint8_t> labels) :
                            magic_number(magic_number), 
                            number_of_images(number_of_images), 
                            rows(rows), 
                            columns(columns), 
                            images(images), 
                            labels_magic_number(labels_magic_number), 
                            number_of_labels(number_of_labels), 
                            labels(labels) {
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
Load the MNIST dataset from a file
*/
MNISTDataset MNISTDataset::load(const std::string& filename, const std::string& labelpath) {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t rows;
    uint32_t columns;
    uint32_t labels_magic_number;
    uint32_t number_of_labels;

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
    return {magic_number, number_of_images, rows, columns, images, labels_magic_number, number_of_labels, labels};
}
