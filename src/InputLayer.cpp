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