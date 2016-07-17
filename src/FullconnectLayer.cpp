#include "EasyCNN/FullconnectLayer.h"

EasyCNN::FullconnectLayer::FullconnectLayer()
{

}
EasyCNN::FullconnectLayer::~FullconnectLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::FullconnectLayer, "FullconnectLayer");
std::string EasyCNN::FullconnectLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::FullconnectLayer::setParamaters(const bool _enabledBias)
{
	enabledBias = _enabledBias;
}
void EasyCNN::FullconnectLayer::solveInnerParams()
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(inputSize.number > 0 && inputSize.channels > 0 && inputSize.width > 0 && inputSize.height > 0, "input size or step is invalidate.");
	easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1, "output size is invalidate.");
	weightsData.reset(new DataBucket(BucketSize(1,inputSize.totalSize()*outputSize.totalSize(),1, 1)));
	if (enabledBias)
	{
		biasData.reset(new DataBucket(BucketSize(1,outputSize.channels,1, 1)));
	}
}
void EasyCNN::FullconnectLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");
	easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1,
		"outputSize is invalidate.");

	const data_type* weightsRawData = weightsData->getData().get();
	const data_type* biasRawData = enabledBias ? biasData->getData().get() : nullptr;
	for (int on = 0; on < outputSize.number;on++)
	{
		const data_type* prevRawData = prevDataBucket->getData().get() + on*inputSize.channels*inputSize.height*inputSize.width;
		data_type* nextRawData = nextDataBucket->getData().get() + on*outputSize.channels*outputSize.height*outputSize.width;
		for (int oc = 0; oc < outputSize.channels; oc++)
		{
			const int outIdx = oc*outputSize.height*outputSize.width;
			data_type sum = 0;
			for (int ic = 0; ic < inputSize.channels; ic++)
			{
				for (int ih = 0; ih < inputSize.height; ih++)
				{
					for (int iw = 0; iw < inputSize.width; iw++)
					{
						const int inIdx = ic*inputSize.height*inputSize.width + ih*inputSize.width + iw;
						sum += prevRawData[inIdx] * weightsRawData[outIdx*inputSize.channels*inputSize.height*inputSize.width + inIdx];
					}
				}
			}
			if (enabledBias)
			{
				sum += biasRawData[outIdx];
			}
			nextRawData[outIdx] = sum;
		}//oc
	}//on
}
void EasyCNN::FullconnectLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}