#include <algorithm>
#include "EasyCNN/ActivationLayer.h"

EasyCNN::SigmodLayer::SigmodLayer()
{

}
EasyCNN::SigmodLayer::~SigmodLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::SigmodLayer, "SigmodLayer");
std::string EasyCNN::SigmodLayer::getLayerType() const
{
	return layerType;
}
//f(x)=1/(1+e^(-x))
static inline EasyCNN::data_type sigmodOperator(const EasyCNN::data_type x)
{
	EasyCNN::data_type result = 0;
	result = 1.0f / (1.0f + std::exp(-1.0f*x));
	return result;
}
void EasyCNN::SigmodLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");

	const data_type* prevRawData = prevDataBucket->getData().get();
	data_type* nextRawData = nextDataBucket->getData().get();
	for (int on = 0; on < outputSize.number; on++)
	{
		for (int oc = 0; oc < outputSize.channels; oc++)
		{
			for (int oh = 0; oh < outputSize.height; oh++)
			{
				for (int ow = 0; ow < outputSize.width; ow++)
				{
					const int idx = on*outputSize.channels*outputSize.height*outputSize.width +
						oc*outputSize.height*outputSize.width + 
						oh*outputSize.width + 
						ow;
					nextRawData[idx] = sigmodOperator(prevRawData[idx]);
				}
			}
		}
	}
}
void EasyCNN::SigmodLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}

EasyCNN::TanhLayer::TanhLayer()
{

}
EasyCNN::TanhLayer::~TanhLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::TanhLayer, "TanhLayer");
std::string EasyCNN::TanhLayer::getLayerType() const
{
	return layerType;
}
//f(x)=(e^x-e^(-x))/(e^x+e^(-x))
static inline EasyCNN::data_type tanhOperator(const EasyCNN::data_type x)
{
	EasyCNN::data_type result = 0;
	const EasyCNN::data_type ex = std::exp(x);
	const EasyCNN::data_type efx = std::exp(-x);
	result = (ex - efx) / (ex + efx);
	return result;
}
void EasyCNN::TanhLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");

	const data_type* prevRawData = prevDataBucket->getData().get();
	data_type* nextRawData = nextDataBucket->getData().get();
	for (int on = 0; on < outputSize.number; on++)
	{
		for (int oc = 0; oc < outputSize.channels; oc++)
		{
			for (int oh = 0; oh < outputSize.height; oh++)
			{
				for (int ow = 0; ow < outputSize.width; ow++)
				{
					const int idx = on*outputSize.channels*outputSize.height*outputSize.width +
						oc*outputSize.height*outputSize.width +
						oh*outputSize.width +
						ow;
					nextRawData[idx] = tanhOperator(prevRawData[idx]);
				}
			}
		}
	}
}
void EasyCNN::TanhLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}

EasyCNN::ReluLayer::ReluLayer()
{

}
EasyCNN::ReluLayer::~ReluLayer()
{

}
DEFINE_LAYER_TYPE(EasyCNN::ReluLayer, "ReluLayer");
std::string EasyCNN::ReluLayer::getLayerType() const
{
	return layerType;
}
//f(x)=max(x,0)
static inline EasyCNN::data_type reluOperator(const EasyCNN::data_type x)
{
	EasyCNN::data_type result = std::max(x, 0.0f);
	return result;
}
void EasyCNN::ReluLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");

	const data_type* prevRawData = prevDataBucket->getData().get();
	data_type* nextRawData = nextDataBucket->getData().get();
	for (int on = 0; on < outputSize.number; on++)
	{
		for (int oc = 0; oc < outputSize.channels; oc++)
		{
			for (int oh = 0; oh < outputSize.height; oh++)
			{
				for (int ow = 0; ow < outputSize.width; ow++)
				{
					const int idx = on*outputSize.channels*outputSize.height*outputSize.width +
						oc*outputSize.height*outputSize.width +
						oh*outputSize.width +
						ow;
					nextRawData[idx] = reluOperator(prevRawData[idx]);
				}
			}
		}
	}
}
void EasyCNN::ReluLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}