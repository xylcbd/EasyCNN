#include "EasyCNN/BatchNormalizationLayer.h"
#include "EasyCNN/CommonTools.h"

EasyCNN::BatchNormalizationLayer::BatchNormalizationLayer()
{

}
EasyCNN::BatchNormalizationLayer::~BatchNormalizationLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::BatchNormalizationLayer, "BatchNormalizationLayer");
std::string EasyCNN::BatchNormalizationLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::BatchNormalizationLayer::setParamaters()
{
	
}
std::string EasyCNN::BatchNormalizationLayer::serializeToString() const
{
	const std::string spliter = " ";
	std::stringstream ss;
	//layer desc
	ss << getLayerType() << spliter;
	return ss.str();
}
void EasyCNN::BatchNormalizationLayer::serializeFromString(const std::string content)
{
	std::stringstream ss(content);
	//layer desc
	std::string _layerType;
	ss >> _layerType;
	easyAssert(_layerType == getLayerType(), "layer type is invalidate.");
}
void EasyCNN::BatchNormalizationLayer::solveInnerParams()
{
	setOutpuBuckerSize(getInputBucketSize());
}
void EasyCNN::BatchNormalizationLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	easyAssert(prevDataSize == nextDataSize, "size must be equal!");

	//TODO
}
void EasyCNN::BatchNormalizationLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket)
{
	easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const DataSize nextDiffSize = nextDiffBucket->getSize();
	easyAssert(prevDataSize == nextDataSize, "size must be equal!");

	//////////////////////////////////////////////////////////////////////////
	//update prevDiff data
	const DataSize prevDiffSize(prevDataSize);
	std::shared_ptr<DataBucket> prevDiffBucket(std::make_shared<DataBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);
	const float* nextDiff = nextDiffBucket->getData().get();
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff && multiply next diff
	//TODO

	//update params
	//TODO

	//////////////////////////////////////////////////////////////////////////
	//chain goto previous layer
	nextDiffBucket = prevDiffBucket;
}