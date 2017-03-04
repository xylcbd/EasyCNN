#include "EasyCNN/BatchNormalizationLayer.h"
#include "EasyCNN/CommonTools.h"

namespace EasyCNN
{
	BatchNormalizationLayer::BatchNormalizationLayer()
	{

	}
	BatchNormalizationLayer::~BatchNormalizationLayer()
	{

	}
	DEFINE_LAYER_TYPE(BatchNormalizationLayer, "BatchNormalizationLayer");
	std::string BatchNormalizationLayer::getLayerType() const
	{
		return layerType;
	}
	void BatchNormalizationLayer::setParamaters()
	{

	}
	std::string BatchNormalizationLayer::serializeToString() const
	{
		const std::string spliter = " ";
		std::stringstream ss;
		//layer desc
		ss << getLayerType() << spliter;
		return ss.str();
	}
	void BatchNormalizationLayer::serializeFromString(const std::string content)
	{
		std::stringstream ss(content);
		//layer desc
		std::string _layerType;
		ss >> _layerType;
		easyAssert(_layerType == getLayerType(), "layer type is invalidate.");
	}
	void BatchNormalizationLayer::solveInnerParams()
	{
		setOutpuBuckerSize(getInputBucketSize());
	}
	void BatchNormalizationLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevDataSize = prev->getSize();
		const DataSize nextDataSize = next->getSize();
		easyAssert(prevDataSize == nextDataSize, "size must be equal!");

		//TODO
	}
	void BatchNormalizationLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();		
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//////////////////////////////////////////////////////////////////////////
		//update prevDiff
		
		prevDiff->fillData(0.0f);
		//calculate current inner diff && multiply next diff
		//TODO

		//update params
		//TODO
	}
}//namespace