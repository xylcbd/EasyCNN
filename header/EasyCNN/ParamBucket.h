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
		ParamSize(const size_t _number, const size_t _channels, const size_t _width, const size_t _height)
			:number(_number), channels(_channels), width(_width), height(_height){}
		inline size_t _4DSize() const { return number*channels*width*height; }
		inline size_t _3DSize() const { return channels*width*height; }
		inline size_t _2DSize() const { return width*height; }
		inline bool operator==(const ParamSize& other) const{
			return other.number == number && other.channels == channels && other.width == width && other.height == height;
		}
		inline size_t getIndex(const size_t in, const size_t ic, const size_t ih, const size_t iw) const{
			return in*channels*height*width + ic*height*width + ih*width + iw;
		}
		inline size_t getIndex(const size_t ic, const size_t ih, const size_t iw) const{
			return ic*height*width + ih*width + iw;
		}
		size_t number = 0;
		size_t channels = 0;
		size_t width = 0;
		size_t height = 0;
	};
	class ParamBucket
	{
	public:
		ParamBucket(const ParamSize _size);
		virtual ~ParamBucket();
		ParamSize getSize() const;
		std::shared_ptr<float> getData() const;
		void fillData(const float item);
		void cloneTo(ParamBucket& target);
	private:
		ParamSize size;
		std::shared_ptr<float> data;
	};
}