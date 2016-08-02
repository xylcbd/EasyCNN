#include <algorithm>
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
void EasyCNN::SoftmaxLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();

	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{
		const float* prevData = prevDataBucket->getData().get() + nn*prevDataSize._3DSize();
		float* nextData = nextDataBucket->getData().get() + nn*nextDataSize._3DSize();

		//step1 : find max value
		float maxVal = prevData[0];
		for (size_t prevDataIdx = 0; prevDataIdx < prevDataSize._3DSize(); prevDataIdx++)
		{
			maxVal = std::max(maxVal, prevData[prevDataIdx]);
		}
		//step2 : sum
		float sum = 0;
		for (size_t prevDataIdx = 0; prevDataIdx < prevDataSize._3DSize(); prevDataIdx++)
		{
			nextData[prevDataIdx] = std::exp(prevData[prevDataIdx] - maxVal);
			sum += nextData[prevDataIdx];
		}
		//step3 : div
		for (size_t prevDataIdx = 0; prevDataIdx < prevDataSize._3DSize(); prevDataIdx++)
		{
			nextData[prevDataIdx] = nextData[prevDataIdx] / sum;
		}
	}
}
void EasyCNN::SoftmaxLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const ParamSize nextDiffSize = nextDiffBucket->getSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	easyAssert(prevDataSize == nextDataSize, "data size must be equal!");

	//update prevDiff data
	const ParamSize prevDiffSize(1, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	easyAssert(prevDiffSize == nextDiffSize, "diff size must be equal!");
	std::shared_ptr<ParamBucket> prevDiffBucket(std::make_shared<ParamBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);	
	float* prevDiff = prevDiffBucket->getData().get();
	for (size_t pn = 0; pn < prevDataSize.number; pn++)
	{
		for (size_t prevDiffIdx = 0; prevDiffIdx < prevDiffSize._3DSize(); prevDiffIdx++)
		{
			for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
			{
				const size_t nextDataIdx = pn*nextDataSize._3DSize() + nextDiffIdx;
				if (nextDiffIdx == prevDiffIdx)
				{
					prevDiff[prevDiffIdx] += nextData[prevDiffIdx] * (1.0f - nextData[prevDiffIdx]) * nextDiff[nextDiffIdx] / nextDataSize.number;
				}
				else
				{
					prevDiff[prevDiffIdx] -= nextData[prevDiffIdx] * nextData[nextDiffIdx] * nextDiff[nextDiffIdx] / nextDataSize.number;
				}
			}
		}
	}

	//update this layer's param
	//softmax layer : nop

	nextDiffBucket = prevDiffBucket;
}