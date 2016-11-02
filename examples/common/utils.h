#pragma once
#include <string>
#include <vector>
#include <sstream>

template<typename SRC_TYPE,typename DST_TYPE>
DST_TYPE convert(const SRC_TYPE& val)
{
	std::stringstream ss;
	ss << val;
	DST_TYPE result;
	ss >> result;
	return result;
}
std::vector<std::string> get_files_in_dir(const std::string& dir_path);