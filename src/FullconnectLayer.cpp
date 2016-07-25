#include "EasyCNN/FullconnectLayer.h"
#include "EasyCNN/CommonTools.h"

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
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(inputSize.number > 0 && inputSize.channels > 0 && inputSize.width > 0 && inputSize.height > 0, "input size or step is invalidate.");
	easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1, "output size is invalidate.");
	weightsData.reset(new ParamBucket(ParamSize(1,(inputSize.totalSize()*outputSize.totalSize())/(inputSize.number*outputSize.number),1, 1)));
	normal_distribution_init(weightsData->getData().get(), weightsData->getSize().totalSize(), 0.0f, 01.f);
	if (enabledBias)
	{
		biasData.reset(new ParamBucket(ParamSize(1, outputSize.channels, 1, 1)));
		const_distribution_init(biasData->getData().get(), biasData->getSize().totalSize(), 0.0f);
	}
}
void EasyCNN::FullconnectLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");
	easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1,
		"outputSize is invalidate.");

	const float* weightsRawData = weightsData->getData().get();
	const float* biasRawData = biasData->getData().get();
	for (int on = 0; on < outputSize.number;on++)
	{
		const float* prevRawData = prevDataBucket->getData().get() + on*inputSize.channels*inputSize.height*inputSize.width;
		float* nextRawData = nextDataBucket->getData().get() + on*outputSize.channels*outputSize.height*outputSize.width;
		for (int oc = 0; oc < outputSize.channels; oc++)
		{
			const int outIdx = oc;
			float sum = 0;
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