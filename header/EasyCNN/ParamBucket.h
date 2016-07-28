#pragma once
#include <memory>
#include "EasyCNN/Configure.h"
#include "EasyCNN/EasyLogger.h"
#include "EasyCNN/EasyAssert.h"

namespace EasyCNN
{
	struct ParamSize
	{
	public:
		ParamSize() = default;
		ParamSize(const int _number, const int _channels, const int _width, const int _height)
			:number(_number), channels(_channels), width(_width), height(_height){}
		inline int totalSize() const { return number*channels*width*height; }
		inline bool operator==(const ParamSize& other) const{
			return other.number == number && other.channels == channels && other.width == width && other.height == height;
		}
		inline int getIdx(const int in, const int ic, const int ih, const int iw){
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
	class ParamBucket
	{
	public:
		ParamBucket(const ParamSize _size);
		virtual ~ParamBucket();
		ParamSize getSize() const;
		std::shared_ptr<float> getData() const;
		void cloneTo(ParamBucket& target);
	private:
		ParamSize size;
		std::shared_ptr<float> data;
	};
}