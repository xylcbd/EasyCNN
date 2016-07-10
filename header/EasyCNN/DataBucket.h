#pragma once
#include <vector>
#include "EasyCNN/Configure.h"
#include "EasyCNN/Logger.h"

namespace EasyCNN
{
	class DataBucket
	{
	public:
		DataBucket();
		virtual ~DataBucket();
	private:
		int channels;
		int width;
		int height;
		std::vector<data_type> data;
	};
}