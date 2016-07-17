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
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");
	easyAssert(outputSize.number >0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1,
		"outputSize is invalidate.");

	for (int in = 0; in < inputSize.number;in++)
	{
		const data_type* prevRawData = prevDataBucket->getData().get() + in*inputSize.channels*inputSize.height*inputSize.width;
		data_type* nextRawData = nextDataBucket->getData().get() + in*outputSize.channels*outputSize.height*outputSize.width;

		//step1 : find max value
		data_type maxVal = prevRawData[0];
		for (int ic = 0; ic < inputSize.channels; ic++)
		{
			maxVal = std::max(maxVal, prevRawData[ic]);
		}
		//step2 : sum
		data_type sum = 0;
		for (int ic = 0; ic < inputSize.channels; ic++)
		{
			nextRawData[ic] = std::exp(prevRawData[ic] - maxVal);
			sum += nextRawData[ic];
		}
		//step3 : div
		for (int ic = 0; ic < inputSize.channels; ic++)
		{
			nextRawData[ic] = nextRawData[ic] / maxVal;
		}
	}
}
void EasyCNN::SoftmaxLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}