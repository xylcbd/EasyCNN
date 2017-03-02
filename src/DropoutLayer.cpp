#include <random>
#include <ctime>
#include "EasyCNN/DropoutLayer.h"
#include "EasyCNN/CommonTools.h"

EasyCNN::DropoutLayer::DropoutLayer()
{
	
}
EasyCNN::DropoutLayer::DropoutLayer(const float _rate)
{
	setParamaters(_rate);
}
EasyCNN::DropoutLayer::~DropoutLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::DropoutLayer, "DropoutLayer");
std::string EasyCNN::DropoutLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::DropoutLayer::setParamaters(const float _rate)
{
	rate = _rate;	
}
std::string EasyCNN::DropoutLayer::serializeToString() const
{
	const std::string spliter = " ";
	std::stringstream ss;
	//layer desc
	ss << getLayerType() << rate << spliter;
	return ss.str();
}
void EasyCNN::DropoutLayer::serializeFromString(const std::string content)
{
	std::stringstream ss(content);
	//layer desc
	std::string _layerType;
	ss >> _layerType
		>> rate;
	easyAssert(_layerType == getLayerType(), "layer type is invalidate.");
}
void EasyCNN::DropoutLayer::solveInnerParams()
{
	//nothing
	if (!maskData.get())
	{
		ParamSize maskSize = getInputBucketSize();
		maskSize.number = 1;
		maskData.reset(new ParamBucket(maskSize));
		const_distribution_init(maskData->getData().get(), maskSize.totalSize(), 1.0f);
	}
	setOutpuBuckerSize(getInputBucketSize());
}
void EasyCNN::DropoutLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	easyAssert(prevDataSize == nextDataSize, "size must be equal!");

	//init rand seed
	std::srand((unsigned int)std::time(nullptr));
	
	if (getPhase() == Phase::Train)
	{
		//fill mask
		float* mask = maskData->getData().get();
		std::random_device rd;
		std::mt19937 engine(rd());
		std::bernoulli_distribution random_distribution(rate);
		for (size_t i = 0; i < maskData->getSize().totalSize(); i++) {
			mask[i] = (float)(random_distribution(engine));
		}

		const float* prevData = prevDataBucket->getData().get();
		float* nextData = nextDataBucket->getData().get();
		for (size_t i = 0; i < nextDataSize.number; i++)
		{
			for (size_t j = 0; j < nextDataSize._3DSize();j++)
			{
				const int dataIdx = i*nextDataSize._3DSize() + j;
				nextData[dataIdx] = prevData[dataIdx] * mask[j] / rate;
			}			
		}
	}
	else
	{
		const float* prevData = prevDataBucket->getData().get();
		float* nextData = nextDataBucket->getData().get();
		for (size_t i = 0; i < nextDataSize.totalSize(); i++)
		{
			nextData[i] = prevData[i];
		}
	}
}
void EasyCNN::DropoutLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket)
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
	const float* mask = maskData->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff && multiply next diff
	for (size_t i = 0; i < nextDataSize.number; i++)
	{
		for (size_t j = 0; j < nextDataSize._3DSize(); j++)
		{
			const int dataIdx = i*nextDataSize._3DSize() + j;
			prevDiff[dataIdx] = nextDiff[dataIdx] * mask[j] / rate;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//chain goto previous layer
	nextDiffBucket = prevDiffBucket;
}