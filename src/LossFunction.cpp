#include <cmath>
#include "EasyCNN/LossFunction.h"
#include "EasyCNN/MathFunctions.h"

namespace EasyCNN
{
	//////////////////////////////////////////////////////////////////////////
	//cross entropy
	float CrossEntropyFunctor::getLoss(const std::shared_ptr<DataBucket> labelDataBucket,
		const std::shared_ptr<DataBucket> outputDataBucket)
	{
		const auto outputSize = outputDataBucket->getSize();
		const float* labelData = labelDataBucket->getData().get();
		const float* outputData = outputDataBucket->getData().get();
		float loss = 0.0f;
		for (size_t i = 0; i < outputSize.totalSize(); i++)
		{
			const float curLoss = -labelData[i] * std::log(outputData[i]);
			loss = moving_average(loss, i+1, curLoss);
		}		
		return loss*outputSize._3DSize();
	}
	void CrossEntropyFunctor::getDiff(const std::shared_ptr<DataBucket> labelDataBucket,
		const std::shared_ptr<DataBucket> outputDataBucket, std::shared_ptr<DataBucket>& diff)
	{
		const DataSize labelSize = labelDataBucket->getSize();
		const DataSize outputSize = outputDataBucket->getSize();
		const DataSize diffSize = diff->getSize();
		diff->fillData(0.0f);
		for (size_t on = 0; on < outputSize.number; on++)
		{
			const float* labelData = labelDataBucket->getData().get() + on*labelSize._3DSize();
			const float* outputData = outputDataBucket->getData().get() + on*outputSize._3DSize();
			float* diffData = diff->getData().get() + on*diffSize._3DSize();
			for (size_t nextDiffIdx = 0; nextDiffIdx < diffSize._3DSize(); nextDiffIdx++)
			{
				const size_t dataIdx = nextDiffIdx;
				diffData[nextDiffIdx] -= ((labelData[dataIdx] / (outputData[dataIdx])));
			}
		}
	}


	//////////////////////////////////////////////////////////////////////////
	//MSE
	float MSEFunctor::getLoss(const std::shared_ptr<DataBucket> labelDataBucket,
		const std::shared_ptr<DataBucket> outputDataBucket)
	{
		const auto outputSize = outputDataBucket->getSize();
		const float* labelData = labelDataBucket->getData().get();
		const float* outputData = outputDataBucket->getData().get();
		float loss = 0.0f;
		for (size_t i = 0; i < outputSize.totalSize(); i++)
		{
			const float curLoss = (outputData[i] - labelData[i])*(outputData[i] - labelData[i]);
			loss = moving_average(loss, i+1, curLoss);
		}
		return loss*outputSize._3DSize();
	}
	void MSEFunctor::getDiff(const std::shared_ptr<DataBucket> labelDataBucket,
		const std::shared_ptr<DataBucket> outputDataBucket, std::shared_ptr<DataBucket>& diff)
	{
		const DataSize labelSize = labelDataBucket->getSize();
		const DataSize outputSize = outputDataBucket->getSize();
		const DataSize diffSize = diff->getSize();
		diff->fillData(0.0f);
		for (size_t on = 0; on < outputSize.number; on++)
		{
			const float* labelData = labelDataBucket->getData().get() + on*labelSize._3DSize();
			const float* outputData = outputDataBucket->getData().get() + on*outputSize._3DSize();
			float* diffData = diff->getData().get() + on*diffSize._3DSize();
			for (size_t nextDiffIdx = 0; nextDiffIdx < diffSize._3DSize(); nextDiffIdx++)
			{
				const size_t dataIdx = nextDiffIdx;
				diffData[nextDiffIdx] += 2.0f*(outputData[dataIdx] - labelData[dataIdx]);
			}
		}
	}
}//namespace