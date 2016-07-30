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
	weightsData.reset(new ParamBucket(ParamSize(1,inputSize._3DSize()*outputSize._3DSize(),1, 1)));
	normal_distribution_init(weightsData->getData().get(), weightsData->getSize()._4DSize(), 0.0f, 01.f);
	if (enabledBias)
	{
		biasData.reset(new ParamBucket(ParamSize(1, outputSize.channels, 1, 1)));
		const_distribution_init(biasData->getData().get(), biasData->getSize()._4DSize(), 0.0f);
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
	const float* biasRawData = enabledBias ? biasData->getData().get() : nullptr;
	for (size_t on = 0; on < outputSize.number; on++)
	{
		const float* prevRawData = prevDataBucket->getData().get() + on*inputSize._3DSize();
		float* nextRawData = nextDataBucket->getData().get() + on*outputSize._3DSize();
		for (size_t oc = 0; oc < outputSize.channels; oc++)
		{
			const size_t outIdx = oc;
			float sum = 0;
			for (size_t ic = 0; ic < inputSize.channels; ic++)
			{
				for (size_t ih = 0; ih < inputSize.height; ih++)
				{
					for (size_t iw = 0; iw < inputSize.width; iw++)
					{
						const size_t inIdx = inputSize.getIndex(ic, ih, iw);
						sum += prevRawData[inIdx] * weightsRawData[outIdx*inputSize._3DSize() + inIdx];
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
void EasyCNN::FullconnectLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const ParamSize nextDiffSize = nextDiffBucket->getSize();
	const ParamSize weightSize = weightsData->getSize();
	const ParamSize biasSize = enabledBias ? biasData->getSize() : ParamSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	float* weight = weightsData->getData().get();
	float* bias = enabledBias ? biasData->getData().get() : nullptr;
	easyAssert(nextDataSize.width == 1 && nextDataSize.height == 1, "use channel only!");
	easyAssert(weightSize._4DSize() == prevDataSize._3DSize() * nextDataSize._3DSize(), "weight size is invalidate!");
	if (enabledBias)
	{
		easyAssert(biasSize._4DSize() == nextDataSize._3DSize(), "bias size is invalidate!");
	}

	//////////////////////////////////////////////////////////////////////////
	//update prevDiff data
	const ParamSize prevDiffSize(1, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	std::shared_ptr<ParamBucket> prevDiffBucket(std::make_shared<ParamBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff && multiply next diff
	for (size_t pn = 0; pn < prevDataSize.number; pn++)
	{
		for (size_t pc = 0; pc < prevDiffSize.channels; pc++)
		{
			for (size_t ph = 0; ph < prevDiffSize.height; ph++)
			{
				for (size_t pw = 0; pw < prevDiffSize.width; pw++)
				{
					const size_t prevDiffIdx = prevDataSize.getIndex(pc, ph, pw);
					for (size_t nc = 0; nc < nextDiffSize.channels; nc++)
					{
						const size_t weightIdx = nc*prevDataSize._3DSize() + prevDataSize.getIndex(pc, ph, pw);
						const size_t nextDiffIdx = nc;
						prevDiff[prevDiffIdx] += weight[weightIdx] * nextDiff[nextDiffIdx];
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//update this layer's param
	//update weight
	//get weight diff
	std::shared_ptr<ParamBucket> weightDiffBucket(std::make_shared<ParamBucket>(weightSize));
	weightDiffBucket->fillData(0.0f);
	float* weightDiff = weightDiffBucket->getData().get();
	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{
		for (size_t nc = 0; nc < nextDataSize.channels; nc++)
		{
			const size_t nextDiffIdx = nc;				
			for (size_t prevData3DIdx = 0; prevData3DIdx < prevDataSize._3DSize(); prevData3DIdx++)
			{					
				const size_t weightDiffIdx = nc*prevDiffSize._3DSize() + prevData3DIdx;
				const size_t prevDataIdx = nn*prevDataSize._3DSize() + prevData3DIdx;					
				weightDiff[weightDiffIdx] += prevData[prevDataIdx] * nextDiff[nextDiffIdx];
			}
		}
	}
	//apply change
	for (size_t weightIdx = 0; weightIdx < weightSize._4DSize(); weightIdx++)
	{
		weight[weightIdx] -= getLearningRate() * weightDiff[weightIdx];
	}
	//////////////////////////////////////////////////////////////////////////
	//update bias
	if (enabledBias)
	{
		//get bias diff
		std::shared_ptr<ParamBucket> biasDiffBucket(std::make_shared<ParamBucket>(biasSize));
		biasDiffBucket->fillData(0.0f);
		float* biastDiff = biasDiffBucket->getData().get();
		for (size_t biasDiffIdx = 0; biasDiffIdx < biasSize._4DSize(); biasDiffIdx++)
		{
			biastDiff[biasDiffIdx] += 1.0f*nextDiff[biasDiffIdx];
		}
		//apply change
		for (size_t biasDiffIdx = 0; biasDiffIdx < biasSize._4DSize(); biasDiffIdx++)
		{
			biastDiff[biasDiffIdx] -= getLearningRate() * biastDiff[biasDiffIdx];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//chain goto previous layer
	nextDiffBucket = prevDiffBucket;
}