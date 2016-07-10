#include "EasyCNN/InputLayer.h"

EasyCNN::InputLayer::InputLayer()
{

}
EasyCNN::InputLayer::~InputLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::InputLayer, "InputLayer");
std::string EasyCNN::InputLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::InputLayer::forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}
void EasyCNN::InputLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}