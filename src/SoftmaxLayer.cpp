#include "EasyCNN/SoftmaxLayer.h"

EasyCNN::SoftmaxLayer::SoftmaxLayer()
{

}
EasyCNN::SoftmaxLayer::~SoftmaxLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::SoftmaxLayer, "SoftmaxLayer");
std::string EasyCNN::SoftmaxLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::SoftmaxLayer::forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}
void EasyCNN::SoftmaxLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}