#include <cmath>
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
	for (size_t i = 0; i < outputSize.totalSize(); i++)
	{
		loss -= labelData[i] * std::log(outputData[i]) / outputSize.number;
	}
	return loss;
}
std::shared_ptr<EasyCNN::DataBucket> EasyCNN::CrossEntropyFunctor::getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket)
{
	const DataSize labelSize = labelDataBucket->getSize();
	const DataSize outputSize = outputDataBucket->getSize();
	const DataSize nextDiffSize(outputSize.number, outputSize.channels, outputSize.height, outputSize.width);
	std::shared_ptr<DataBucket> nextDiffBucket(std::make_shared<DataBucket>(nextDiffSize));
	nextDiffBucket->fillData(0.0f);	
	for (size_t on = 0; on < outputSize.number; on++)
	{
		const float* labelData = labelDataBucket->getData().get() + on*labelSize._3DSize();
		const float* outputData = outputDataBucket->getData().get() + on*outputSize._3DSize();
		float* nextDiff = nextDiffBucket->getData().get() + on*nextDiffSize._3DSize();
		for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
		{
			const size_t dataIdx = nextDiffIdx;
			nextDiff[nextDiffIdx] -= ((labelData[dataIdx] / (outputData[dataIdx])));
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
	for (size_t i = 0; i < outputSize.totalSize(); i++)
	{
		loss += (outputData[i] - labelData[i])*(outputData[i] - labelData[i]) / outputSize.number;
	}
	return loss;
}
std::shared_ptr<EasyCNN::DataBucket> EasyCNN::MSEFunctor::getDiff(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const std::shared_ptr<EasyCNN::DataBucket> outputDataBucket)
{
	const DataSize labelSize = labelDataBucket->getSize();
	const DataSize outputSize = outputDataBucket->getSize();
	const DataSize nextDiffSize(outputSize.number, outputSize.channels, outputSize.height, outputSize.width);
	std::shared_ptr<DataBucket> nextDiffBucket(std::make_shared<DataBucket>(nextDiffSize));
	nextDiffBucket->fillData(0.0f);	
	for (size_t on = 0; on < outputSize.number; on++)
	{
		const float* labelData = labelDataBucket->getData().get() + on*labelSize._3DSize();
		const float* outputData = outputDataBucket->getData().get() + on*outputSize._3DSize();
		float* nextDiff = nextDiffBucket->getData().get() + on*nextDiffSize._3DSize();
		for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
		{
			const size_t dataIdx = nextDiffIdx;
			nextDiff[nextDiffIdx] += 2.0f*(outputData[dataIdx]-labelData[dataIdx]);
		}
	}
	return nextDiffBucket;
}