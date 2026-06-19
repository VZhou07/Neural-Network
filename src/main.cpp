#include <iostream>
#include <fstream>
#include <cstdint>
#include <filesystem>
#include <vector>


struct MNISTDataset() {
    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::vector<uint8_t> images = {};
};
/* 
Most CPUs use little-endian but MNIST is stored in big-endian so flip it
*/
uint32_t flip_endian(uint32_t value) {
    return ((value & 0x000000FF) << 24) | // Grab byte 4 and shift it to position 1
           ((value & 0x0000FF00) << 8)  | // Grab byte 3 and shift it to position 2
           ((value & 0x00FF0000) >> 8)  | // Grab byte 2 and shift it to position 3
           ((value & 0xFF000000) >> 24);  // Grab byte 1 and shift it to position 4
}
/*
Function to unpack the MNIST dataset
*/
MNISTDataset unpack_mnist_dataset(const std::string& filename){
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t rows;
    uint32_t columns;
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filename);
    }
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = flip_endian(magic_number);
    file.read(reinterpret_cast<char*>(&number_of_images),sizeof(number_of_images));
    number_of_images = flip_endian(number_of_images);
    file.read(reinterpret_cast<char*>(&rows),sizeof(rows));
    rows = flip_endian(rows);
    file.read(reinterpret_cast<char*>(&columns),sizeof(columns));
    columns = flip_endian(columns);
    //rest of bytes are image
    std::vector<uint8_t> images(number_of_images * rows * columns);
    file.read(reinterpret_cast<char*>(images.data()), images.size());
    file.close();
    return {magic_number, number_of_images, rows, columns, images};
}

int main(){
    MNISTDataset data = ;
    data=unpack_mnist_dataset((std::filesystem::path("data")/"train-images-idx3-ubyte").string());
    std::cout << "Magic number: " << data.magic_number << std::endl;
    std::cout << "Number of images: " << data.number_of_images << std::endl;
    std::cout << "Rows: " << data.rows << std::endl;
    std::cout << "Columns: " << data.columns << std::endl;
    return 0;
}