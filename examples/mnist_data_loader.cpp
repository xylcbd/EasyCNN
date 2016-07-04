#include <fstream>
#include <cassert>
#include "mnist_data_loader.h"

template<typename T>
T reverse_endian(T p) {
	std::reverse(reinterpret_cast<char*>(&p), reinterpret_cast<char*>(&p)+sizeof(T));
	return p;
}

inline bool is_little_endian() {
	int x = 1;
	return *(char*)&x != 0;
}

bool load_mnist_images(const std::string& file_path, std::vector<image_t>& images)
{
	images.clear();
	std::ifstream ifs(file_path,std::ios::binary);
	if (!ifs.is_open())
	{
		return false;
	}
	//detect platform information
	const bool is_little_endian_flag = is_little_endian();
	//magic number
	uint32_t magic_number;
	ifs.read((char*)&magic_number, sizeof(magic_number));
	if (is_little_endian_flag)
	{
		magic_number = reverse_endian<uint32_t>(magic_number);
	}
	const bool magic_number_validate = (magic_number == 0x00000803);
	assert(magic_number_validate);
	if (!magic_number_validate)
	{
		return false;
	}
	//count
	uint32_t images_total_count = 0;
	ifs.read((char*)&images_total_count, sizeof(images_total_count));
	//image property
	uint32_t width = 0, height = 0;
	ifs.read((char*)&height, sizeof(height));
	ifs.read((char*)&width, sizeof(width));
	if (is_little_endian_flag)
	{
		images_total_count = reverse_endian<uint32_t>(images_total_count);
		width = reverse_endian<uint32_t>(width);
		height = reverse_endian<uint32_t>(height);
	}
	//images
	for (uint32_t i = 0; i < images_total_count;i++)
	{
		image_t image;
		image.channels = 1;
		image.width = width;
		image.height = height;
		image.data.resize(width*height);
		ifs.read((char*)&image.data[0], width*height);
		images.push_back(image);
	}
	return true;
}

bool load_mnist_labels(const std::string& file_path, std::vector<label_t>& labels)
{
	labels.clear();
	std::ifstream ifs(file_path, std::ios::binary);
	if (!ifs.is_open())
	{
		return false;
	}//detect platform information
	const bool is_little_endian_flag = is_little_endian();
	//magic number
	uint32_t magic_number;
	ifs.read((char*)&magic_number, sizeof(magic_number));
	if (is_little_endian_flag)
	{
		magic_number = reverse_endian<uint32_t>(magic_number);
	}
	const bool magic_number_validate = (magic_number == 0x00000801);
	assert(magic_number_validate);
	if (!magic_number_validate)
	{
		return false;
	}
	//count
	uint32_t labels_total_count = 0;
	ifs.read((char*)&labels_total_count, sizeof(labels_total_count)); 
	if (is_little_endian_flag)
	{
		labels_total_count = reverse_endian<uint32_t>(labels_total_count);
	}
	//labels
	for (uint32_t i = 0; i < labels_total_count; i++)
	{
		label_t label;		
		ifs.read((char*)&label.data, sizeof(label.data));
		labels.push_back(label);
	}
	return true;
}