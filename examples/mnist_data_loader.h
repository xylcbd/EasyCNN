#pragma once
#include <vector>
#include <cstdint>

struct image_t
{
	int width, height, channels;
	std::vector<uint8_t> data;
};
bool load_mnist_images(const std::string& file_path, std::vector<image_t>& images);

struct label_t
{
	uint8_t data;
};
bool load_mnist_labels(const std::string& file_path, std::vector<label_t>& labels);
