#pragma once

#include <cstdint>
#include <string>
#include <vector>

class MNISTDataset {
public:
    MNISTDataset(uint32_t magic_number=0,
         uint32_t number_of_images=0, 
         uint32_t rows=0, 
         uint32_t columns=0, 
         std::vector<uint8_t> images={}, 
         uint32_t labels_magic_number=0, 
         uint32_t number_of_labels=0, 
         std::vector<uint8_t> labels={});

    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t labels_magic_number = 0;
    uint32_t number_of_labels = 0;
    std::vector<uint8_t> images;
    std::vector<uint8_t> labels;

    static MNISTDataset load(const std::string& filename, const std::string& labelpath);
    static void shuffle_images();

    

private:
    static uint32_t flip_endian(uint32_t value);
};
