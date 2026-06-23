#pragma once

#include <cstdint>
#include <string>
#include <vector>

class MNISTDataset {
public:
    MNISTDataset(
         std::vector<uint8_t> images={}, 
         std::vector<uint8_t> labels={},
         std::vector<uint8_t> test_images={},
         std::vector<uint8_t> test_labels={});

    static MNISTDataset load(const std::string& filename, const std::string& labelpath, const std::string& test_filename, const std::string& test_labelpath);
    static uint32_t flip_endian(uint32_t value);
    void shuffle_images(int size_of_dataset);
    std::vector<uint8_t> images;
    std::vector<uint8_t> labels;
    std::vector<uint8_t> test_images;
    std::vector<uint8_t> test_labels;

    size_t size_of_training_set=48000;
    size_t size_of_validation_set=12000;
    size_t size_of_test_set=10000;
};
