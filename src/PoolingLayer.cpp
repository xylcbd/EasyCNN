#include "EasyCNN/PoolingLayer.h"

EasyCNN::PoolingLayer::PoolingLayer()
{

}
EasyCNN::PoolingLayer::~PoolingLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::PoolingLayer, "PoolingLayer");
std::string EasyCNN::PoolingLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::PoolingLayer::forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}
void EasyCNN::PoolingLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}