#include "EasyCNN/ConvolutionLayer.h"

EasyCNN::ConvolutionLayer::ConvolutionLayer()
{

}
EasyCNN::ConvolutionLayer::~ConvolutionLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::ConvolutionLayer, "ConvolutionLayer");
std::string EasyCNN::ConvolutionLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::ConvolutionLayer::forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}
void EasyCNN::ConvolutionLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}