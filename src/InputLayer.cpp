#include "EasyCNN/InputLayer.h"

namespace EasyCNN
{
	InputLayer::InputLayer()
	{

	}
	InputLayer::~InputLayer()
	{

	}
	DEFINE_LAYER_TYPE(InputLayer, "InputLayer");
	std::string InputLayer::serializeToString() const
	{
		const std::string spliter = " ";
		std::stringstream ss;
		//layer enc
		ss << getLayerType() << spliter
			<< inputSize.channels << spliter << inputSize.width << spliter << inputSize.height << spliter;
		return ss.str();
	}
	void InputLayer::serializeFromString(const std::string content)
	{
		//layer desc
		std::string _layerType;
		int _number = 1;
		int _channels = 0;
		int _width = 0;
		int _height = 0;
		std::stringstream ss(content);
		ss >> _layerType >> _channels >> _width >> _height;
		easyAssert(_layerType == getLayerType(), "layer type is invalidate.");

		DataSize mapSize;
		mapSize.number = _number;
		mapSize.channels = _channels;
		mapSize.width = _width;
		mapSize.height = _height;		
		setInputBucketSize(mapSize);
		setOutpuBuckerSize(mapSize);
		solveInnerParams();
	}
	std::string InputLayer::getLayerType() const
	{
		return layerType;
	}
	void InputLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		prev->cloneTo(*next);
	}
	void InputLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		//data layer : nop
	}
}//namespace