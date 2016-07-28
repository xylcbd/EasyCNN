#pragma once
#include <memory>
#include "EasyCNN/Configure.h"
#include "EasyCNN/EasyLogger.h"
#include "EasyCNN/EasyAssert.h"

namespace EasyCNN
{
	struct DataSize
	{
	public:
		DataSize() = default;
		DataSize(const int _number,const int _channels, const int _width, const int _height)
			:number(_number),channels(_channels), width(_width), height(_height){}
		inline size_t totalSize() const { return number*channels*width*height; }
		inline size_t singleSize() const { return channels*width*height; }
		inline bool operator==(const DataSize& other) const{ 
			return other.number == number && other.channels == channels && other.width == width && other.height == height;
		}
		inline int getIndex(const int in, const int ic, const int ih, const int iw) const{
			return in*channels*height*width + ic*height*width + ih*width + iw;
		}
		inline int getIndex(const int ic, const int ih, const int iw) const{
			return ic*height*width + ih*width + iw;
		}
		int number = 0;
		int channels = 0;
		int width = 0;
		int height = 0;
	};
	class DataBucket
	{
	public:
		DataBucket(const DataSize _size);
		virtual ~DataBucket();		
	public:
		DataSize getSize() const;
		std::shared_ptr<float> getData() const;
		void fillData(float item);
		void cloneTo(DataBucket& target);
	private:
		DataSize size;
		std::shared_ptr<float> data;
	};
}