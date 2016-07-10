#include "EasyCNN/ActivationLayer.h"


//
EasyCNN::SigmodLayer::SigmodLayer()
{

}
EasyCNN::SigmodLayer::~SigmodLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::SigmodLayer, "SigmodLayer");
std::string EasyCNN::SigmodLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::SigmodLayer::forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}
void EasyCNN::SigmodLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}

EasyCNN::TanhLayer::TanhLayer()
{

}
EasyCNN::TanhLayer::~TanhLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::TanhLayer, "TanhLayer");
std::string EasyCNN::TanhLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::TanhLayer::forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}
void EasyCNN::TanhLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}

EasyCNN::ReluLayer::ReluLayer()
{

}
EasyCNN::ReluLayer::~ReluLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::ReluLayer, "ReluLayer");
std::string EasyCNN::ReluLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::ReluLayer::forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}
void EasyCNN::ReluLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{

}