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
void EasyCNN::SoftmaxLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket)
{
	easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const DataSize nextDiffSize = nextDiffBucket->getSize();
	easyAssert(prevDataSize == nextDataSize, "data size must be equal!");
	easyAssert(nextDiffSize == nextDataSize, "next data's and diff's size must be equal! ");

	//update prevDiff data
	const DataSize prevDiffSize(prevDataSize.number, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	easyAssert(prevDiffSize == nextDiffSize, "diff size must be equal!");
	std::shared_ptr<DataBucket> prevDiffBucket(std::make_shared<DataBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);		
	for (size_t pn = 0; pn < prevDataSize.number; pn++)
	{
		const float* prevData = prevDataBucket->getData().get() + pn*prevDataSize._3DSize();
		const float* nextData = nextDataBucket->getData().get() + pn*nextDataSize._3DSize();
		const float* nextDiff = nextDiffBucket->getData().get() + pn*nextDiffSize._3DSize();
		float* prevDiff = prevDiffBucket->getData().get() + pn*prevDiffSize._3DSize();		
		for (size_t prevDiffIdx = 0; prevDiffIdx < prevDiffSize._3DSize(); prevDiffIdx++)
		{
			for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
			{
				if (nextDiffIdx == prevDiffIdx)
				{
					prevDiff[prevDiffIdx] += nextData[prevDiffIdx] * (1.0f - nextData[prevDiffIdx]) * nextDiff[nextDiffIdx];
				}
				else
				{
					prevDiff[prevDiffIdx] -= nextData[prevDiffIdx] * nextData[nextDiffIdx] * nextDiff[nextDiffIdx];
				}
			}
		}
	}

	//update this layer's param
	//softmax layer : nop

	nextDiffBucket = prevDiffBucket;
}