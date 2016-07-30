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

	for (size_t on = 0; on < outputSize.number; on++)
	{
		const float* prevRawData = prevDataBucket->getData().get() + on*inputSize._3DSize();
		float* nextRawData = nextDataBucket->getData().get() + on*outputSize._3DSize();

		//step1 : find max value
		float maxVal = prevRawData[0];
		for (size_t ic = 0; ic < inputSize._3DSize(); ic++)
		{
			maxVal = std::max(maxVal, prevRawData[ic]);
		}
		//step2 : sum
		float sum = 0;
		for (size_t ic = 0; ic < inputSize._3DSize(); ic++)
		{
			nextRawData[ic] = std::exp(prevRawData[ic] - maxVal);
			sum += nextRawData[ic];
		}
		//step3 : div
		for (size_t ic = 0; ic < inputSize._3DSize(); ic++)
		{
			nextRawData[ic] = nextRawData[ic] / sum;
		}
	}
}
void EasyCNN::SoftmaxLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
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
	for (size_t pidx = 0; pidx < prevDiffSize._4DSize(); pidx++)
	{
		for (size_t nn = 0; nn < nextDataSize.number; nn++)
		{
			for (size_t nidx = 0; nidx < nextDataSize._3DSize(); nidx++)
			{
				const size_t dataIdx = nn*nextDataSize._3DSize() + nidx;
				if (nidx == pidx)
				{
					prevDiff[pidx] += nextData[dataIdx] * (1.0f - nextData[dataIdx]);
				}
				else
				{
					prevDiff[pidx] -= nextData[dataIdx] * nextData[dataIdx];
				}
			}
		}
	}
	for (size_t i = 0; i < prevDiffBucket->getSize()._4DSize();i++)
	{
		prevDiff[i] *= nextDiff[i];
	}	

	//update this layer's param
	//softmax layer : nop

	nextDiffBucket = prevDiffBucket;
}