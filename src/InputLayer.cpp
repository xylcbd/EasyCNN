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
void EasyCNN::InputLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	prevDataBucket->cloneTo(*nextDataBucket);
}
void EasyCNN::InputLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	//data layer : nop
}