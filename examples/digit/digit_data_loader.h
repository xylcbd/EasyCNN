#pragma once
#include <vector>
#include <cstdint>

struct image_t
{
	size_t width, height, channels;
	std::vector<uint8_t> data;
};
struct label_t
{
	uint8_t data;
};
bool load_digit_images(const std::string& dir_path, std::vector<image_t>& images, std::vector<label_t>& labels);
