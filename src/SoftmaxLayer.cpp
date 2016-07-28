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
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");
	easyAssert(outputSize.number >0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1,
		"outputSize is invalidate.");

	for (int on = 0; on < outputSize.number; on++)
	{
		const float* prevRawData = prevDataBucket->getData().get() + on*inputSize.channels*inputSize.height*inputSize.width;
		float* nextRawData = nextDataBucket->getData().get() + on*outputSize.channels*outputSize.height*outputSize.width;

		//step1 : find max value
		float maxVal = prevRawData[0];
		for (int ic = 0; ic < inputSize.channels; ic++)
		{
			maxVal = std::max(maxVal, prevRawData[ic]);
		}
		//step2 : sum
		float sum = 0;
		for (int ic = 0; ic < inputSize.channels; ic++)
		{
			nextRawData[ic] = std::exp(prevRawData[ic] - maxVal);
			sum += nextRawData[ic];
		}
		//step3 : div
		for (int ic = 0; ic < inputSize.channels; ic++)
		{
			nextRawData[ic] = nextRawData[ic] / sum;
		}
	}
}
void EasyCNN::SoftmaxLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const DataSize nextDiffSize = nextDiffBucket->getSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	easyAssert(prevDataSize == nextDataSize, "size must be equal!");
	easyAssert(nextDiffSize == nextDataSize, "size must be equal!");

	//update prevDiff data
	std::shared_ptr<DataBucket> prevDiffBucket(std::make_shared<DataBucket>(prevDataSize));
	prevDiffBucket->fillData(0.0f);
	const DataSize prevDiffSize = prevDiffBucket->getSize();
	float* prevDiff = prevDiffBucket->getData().get();
	for (int pn = 0; pn < prevDiffSize.number;pn++)
	{
		for (int nc = 0; nc < nextDataSize.channels;nc++)
		{
			for (int pc = 0; pc < prevDiffSize.channels;pc++)
			{
				if (nc == pc)
				{
					prevDiff[pc] += nextData[nc] * (1.0f - nextData[nc]);
				}
				else
				{
					prevDiff[pc] -= nextData[nc] * nextData[pc];
				}
			}
		}
	}
	for (size_t i = 0; i < prevDiffBucket->getSize().totalSize();i++)
	{
		prevDiff[i] *= nextDiff[i];
	}
	nextDiffBucket = prevDiffBucket;

	//update this layer's param
	//softmax layer : nop
}