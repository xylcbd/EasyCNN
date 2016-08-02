#include "EasyCNN/LossFunction.h"

//////////////////////////////////////////////////////////////////////////
//cross entropy
float EasyCNN::CrossEntropyFunctor::getLoss(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket)
{
	const auto outputSize = outputDataBucket->getSize();
	const float* labelData = labelDataBucket->getData().get();
	const float* outputData = outputDataBucket->getData().get();
	float loss = 0.0f;
	for (size_t i = 0; i < outputSize._4DSize(); i++)
	{
		loss -= labelData[i] * std::log(outputData[i]) / outputSize.number;
	}
	return loss;
}
std::shared_ptr<EasyCNN::ParamBucket> EasyCNN::CrossEntropyFunctor::getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket)
{
	const auto outputSize = outputDataBucket->getSize();
	const float* labelData = labelDataBucket->getData().get();
	const float* outputData = outputDataBucket->getData().get();

	const ParamSize nextDiffSize = ParamSize(1, outputSize.channels, outputSize.height, outputSize.width);
	std::shared_ptr<ParamBucket> nextDiffBucket(std::make_shared<ParamBucket>(nextDiffSize));
	nextDiffBucket->fillData(0.0f);
	float* nextDiff = nextDiffBucket->getData().get();
	for (size_t on = 0; on < outputSize.number; on++)
	{
		for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
		{
			const size_t dataIdx = on*outputSize._3DSize() + nextDiffIdx;
			nextDiff[nextDiffIdx] -= ((labelData[dataIdx] / (outputData[dataIdx]))) / outputSize.number;
		}
	}
	return nextDiffBucket;
}


//////////////////////////////////////////////////////////////////////////
//MSE
float EasyCNN::MSEFunctor::getLoss(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket)
{
	const auto outputSize = outputDataBucket->getSize();
	const float* labelData = labelDataBucket->getData().get();
	const float* outputData = outputDataBucket->getData().get();
	float loss = 0.0f;
	for (size_t i = 0; i < outputSize._4DSize(); i++)
	{
		loss += std::pow(outputData[i] - labelData[i],2) / outputSize.number;
	}
	return loss;
}
std::shared_ptr<EasyCNN::ParamBucket> EasyCNN::MSEFunctor::getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket)
{
	const auto outputSize = outputDataBucket->getSize();
	const float* labelData = labelDataBucket->getData().get();
	const float* outputData = outputDataBucket->getData().get();

	const ParamSize nextDiffSize = ParamSize(1, outputSize.channels, outputSize.height, outputSize.width);
	std::shared_ptr<ParamBucket> nextDiffBucket(std::make_shared<ParamBucket>(nextDiffSize));
	nextDiffBucket->fillData(0.0f);
	float* nextDiff = nextDiffBucket->getData().get();
	for (size_t on = 0; on < outputSize.number; on++)
	{
		for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
		{
			const size_t dataIdx = on*outputSize._3DSize() + nextDiffIdx;
			nextDiff[nextDiffIdx] += 2.0f*(outputData[dataIdx]-labelData[dataIdx]) / outputSize.number;
		}
	}
	return nextDiffBucket;
}