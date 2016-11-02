#include <fstream>
#include <cassert>
#include "digit_data_loader.h"
#include "../common/utils.h"
#include <opencv2/opencv.hpp>

template<typename T>
T reverse_endian(T p) {
	std::reverse(reinterpret_cast<char*>(&p), reinterpret_cast<char*>(&p)+sizeof(T));
	return p;
}

inline bool is_little_endian() {
	int x = 1;
	return *(char*)&x != 0;
}

static label_t get_label_from_path(const std::string& file_path)
{
	const auto begin_pos = file_path.find_last_of('\\')+1;
	const auto end_pos = file_path.find_last_of('_');
	const std::string subStr = file_path.substr(begin_pos, end_pos - begin_pos);
	label_t label;
	label.data = convert<std::string, int>(subStr);
	return label;
}
bool load_digit_images(const std::string& dir_path, std::vector<image_t>& images, std::vector<label_t>& labels)
{
	images.clear();
	labels.clear();

	const std::vector<std::string> all_files = get_files_in_dir(dir_path);
	//images
	for (uint32_t i = 0; i < all_files.size(); i++)
	{
		const std::string& file_path = all_files[i];
		const cv::Mat srcImg = cv::imread(file_path, 0);
		image_t image;
		image.channels = 1;
		image.width = srcImg.cols;
		image.height = srcImg.rows;
		image.data.resize(image.width*image.height);
		for (int y = 0; y < srcImg.rows;y++)
		{
			const uchar* srcRow = srcImg.data + y*srcImg.step[0];
			uchar* dstRow = (uchar*)&image.data[0] + y*image.width;
			memcpy(dstRow, srcRow, image.width);
		}
		images.push_back(image);
		labels.push_back(get_label_from_path(file_path));
	}
	return true;
}
