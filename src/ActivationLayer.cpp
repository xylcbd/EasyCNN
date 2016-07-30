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
static inline float sigmodOperator(const float x)
{
	float result = 0;
	result = 1.0f / (1.0f + std::exp(-1.0f*x));
	return result;
}
//f'(x) = x(1-x)
static inline float sigmodDfOperator(const float x)
{
	return x*(1.0f - x);
}
void EasyCNN::SigmodLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");

	const float* prevRawData = prevDataBucket->getData().get();
	float* nextRawData = nextDataBucket->getData().get();
	for (size_t on = 0; on < outputSize.number; on++)
	{
		for (size_t oc = 0; oc < outputSize.channels; oc++)
		{
			for (size_t oh = 0; oh < outputSize.height; oh++)
			{
				for (size_t ow = 0; ow < outputSize.width; ow++)
				{
					const size_t idx = outputSize.getIndex(on, oc, oh, ow);
					nextRawData[idx] = sigmodOperator(prevRawData[idx]);
				}
			}
		}
	}
}
void EasyCNN::SigmodLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const ParamSize nextDiffSize = nextDiffBucket->getSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	easyAssert(prevDataSize == nextDataSize, "size must be equal!");

	//update prevDiff data
	const ParamSize prevDiffSize(1, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	std::shared_ptr<ParamBucket> prevDiffBucket(std::make_shared<ParamBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);	
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff
	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{
		for (size_t pc = 0; pc < prevDiffSize.channels; pc++)
		{
			for (size_t ph = 0; ph < prevDiffSize.height; ph++)
			{
				for (size_t pw = 0; pw < prevDiffSize.width; pw++)
				{
					const size_t dataIdx = nextDataSize.getIndex(nn, pc, ph, pw);
					const size_t paramIdx = prevDiffSize.getIndex(pc, ph, pw);
					prevDiff[paramIdx] += sigmodDfOperator(nextData[dataIdx]);
				}
			}
		}
	}
	//multiply next diff
	for (size_t i = 0; i < prevDiffBucket->getSize()._4DSize(); i++)
	{
		prevDiff[i] *= nextDiff[i];
	}
	nextDiffBucket = prevDiffBucket;

	//update this layer's param
	//Tanh layer : nop
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
static inline float tanhOperator(const float x)
{
	float result = 0;
	const float ex = std::exp(x);
	const float efx = std::exp(-x);
	result = (ex - efx) / (ex + efx);
	return result;
}
//f'(x)=1-x^(1/2)
static inline float tanhDfOperator(const float x)
{
	return 1.0f - std::sqrt(x);
}
void EasyCNN::TanhLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");

	const float* prevRawData = prevDataBucket->getData().get();
	float* nextRawData = nextDataBucket->getData().get();
	for (size_t on = 0; on < outputSize.number; on++)
	{
		for (size_t oc = 0; oc < outputSize.channels; oc++)
		{
			for (size_t oh = 0; oh < outputSize.height; oh++)
			{
				for (size_t ow = 0; ow < outputSize.width; ow++)
				{
					const size_t idx = outputSize.getIndex(on, oc, oh, ow);
					nextRawData[idx] += tanhOperator(prevRawData[idx]);
				}
			}
		}
	}
}
void EasyCNN::TanhLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const ParamSize nextDiffSize = nextDiffBucket->getSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	easyAssert(prevDataSize == nextDataSize, "size must be equal!");

	//update prevDiff data
	const ParamSize prevDiffSize(1, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	std::shared_ptr<ParamBucket> prevDiffBucket(std::make_shared<ParamBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff
	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{
		for (size_t pc = 0; pc < prevDiffSize.channels; pc++)
		{
			for (size_t ph = 0; ph < prevDiffSize.height; ph++)
			{
				for (size_t pw = 0; pw < prevDiffSize.width; pw++)
				{
					const size_t dataIdx = nextDataSize.getIndex(nn, pc, ph, pw);
					const size_t paramIdx = prevDiffSize.getIndex(pc, ph, pw);
					prevDiff[paramIdx] += tanhDfOperator(nextData[dataIdx]);
				}
			}
		}
	}
	//multiply next diff
	for (size_t i = 0; i < prevDiffBucket->getSize()._4DSize(); i++)
	{
		prevDiff[i] *= nextDiff[i];
	}
	nextDiffBucket = prevDiffBucket;

	//update this layer's param
	//Tanh layer : nop
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
static inline float reluOperator(const float x)
{
	float result = std::max(x, 0.0f);
	return result;
}
//f'(x)=0(x<=0),1(x>0)
static inline float reluDfOperator(const float x)
{
	//note : too small df is not suitable.
	return x <= 0.0f ? 0.01f : 1.0f;
}
void EasyCNN::ReluLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(inputSize == outputSize, "outputSize must be equals with inputSize.");

	const float* prevRawData = prevDataBucket->getData().get();
	float* nextRawData = nextDataBucket->getData().get();
	for (size_t on = 0; on < outputSize.number; on++)
	{
		for (size_t oc = 0; oc < outputSize.channels; oc++)
		{
			for (size_t oh = 0; oh < outputSize.height; oh++)
			{
				for (size_t ow = 0; ow < outputSize.width; ow++)
				{
					const size_t idx = outputSize.getIndex(on, oc, oh, ow);
					nextRawData[idx] = reluOperator(prevRawData[idx]);
				}
			}
		}
	}
}
void EasyCNN::ReluLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const ParamSize nextDiffSize = nextDiffBucket->getSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	easyAssert(prevDataSize == nextDataSize, "size must be equal!");

	//update prevDiff data
	const ParamSize prevDiffSize(1, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	std::shared_ptr<ParamBucket> prevDiffBucket(std::make_shared<ParamBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff
	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{
		for (size_t pc = 0; pc < prevDiffSize.channels; pc++)
		{
			for (size_t ph = 0; ph < prevDiffSize.height; ph++)
			{
				for (size_t pw = 0; pw < prevDiffSize.width; pw++)
				{
					const size_t dataIdx = nextDataSize.getIndex(nn, pc, ph, pw);
					const size_t paramIdx = prevDiffSize.getIndex(pc, ph, pw);
					prevDiff[paramIdx] += reluDfOperator(nextData[dataIdx]);
				}
			}
		}
	}
	//multiply next diff
	for (size_t i = 0; i < prevDiffBucket->getSize()._4DSize(); i++)
	{
		prevDiff[i] *= nextDiff[i];
	}
	nextDiffBucket = prevDiffBucket;

	//update this layer's param
	//RELU layer : nop
}