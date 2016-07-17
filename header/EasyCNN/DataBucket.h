#pragma once
#include <memory>
#include "EasyCNN/Configure.h"
#include "EasyCNN/EasyLogger.h"
#include "EasyCNN/EasyAssert.h"

namespace EasyCNN
{
	struct BucketSize
	{
	public:
		BucketSize() = default;
		BucketSize(const int _number,const int _channels, const int _width, const int _height)
			:number(_number),channels(_channels), width(_width), height(_height){}
		inline int totalSize() const { return number*channels*width*height; }
		inline bool operator==(const BucketSize& other) const{ 
			return other.number == number && other.channels == channels && other.width == width && other.height == height;
		}
		int number = 0;
		int channels = 0;
		int width = 0;
		int height = 0;
	};
	class DataBucket
	{
	public:
		DataBucket(const BucketSize _size);
		virtual ~DataBucket();		
		BucketSize getSize() const;
		std::shared_ptr<data_type> getData() const;
		void cloneTo(DataBucket& target);
	private:
		BucketSize size;
		std::shared_ptr<data_type> data;
	};
}