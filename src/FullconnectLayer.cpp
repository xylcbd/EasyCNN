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
	const float* biasRawData = enabledBias ? biasData->getData().get() : nullptr;
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
						const int inIdx = inputSize.getIndex(ic, ih, iw);
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
void EasyCNN::FullconnectLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const DataSize nextDiffSize = nextDiffBucket->getSize();
	const ParamSize weightSize = weightsData->getSize();
	const ParamSize biasSize = enabledBias ? biasData->getSize() : ParamSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	float* weight = weightsData->getData().get();
	float* bias = enabledBias ? biasData->getData().get() : nullptr;
	easyAssert(nextDiffSize == nextDataSize, "size must be equal!");
	easyAssert(nextDataSize.width == 1 && nextDataSize.height == 1, "use channel only!");
	easyAssert(weightSize.totalSize() == prevDataSize.singleSize() * nextDataSize.singleSize(), "weight size is invalidate!");
	if (enabledBias)
	{
		easyAssert(biasSize.totalSize() == (nextDataSize.totalSize() / (nextDataSize.number)), "bias size is invalidate!");
	}
	//update prevDiff data
	std::shared_ptr<DataBucket> prevDiffBucket(std::make_shared<DataBucket>(prevDataSize));
	prevDiffBucket->fillData(0.0f);
	const DataSize prevDiffSize = prevDiffBucket->getSize();
	float* prevDiff = prevDiffBucket->getData().get();
	for (int pn = 0; pn < prevDiffSize.number; pn++)
	{
		for (int pc = 0; pc < prevDiffSize.channels; pc++)
		{
			for (int ph = 0; ph < prevDiffSize.height; ph++)
			{
				for (int pw = 0; pw < prevDiffSize.width; pw++)
				{
					const int data_idx = prevDataSize.getIndex(pn, pc, ph, pw);
					for (int nc = 0; nc < nextDiffSize.channels; nc++)
					{						
						const int weight_idx = pn*prevDataSize.singleSize()*nextDiffSize.singleSize() + nc*prevDiffSize.totalSize() + prevDataSize.getIndex(pc, ph, pw);;
						prevDiff[data_idx] += weight[weight_idx];
					}
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	easyAssert(false, "FIXME please!");
	for (size_t i = 0; i < prevDiffBucket->getSize().totalSize(); i++)
	{
		prevDiff[i] *= nextDiff[i];
	}
	for (int pn = 0; pn < prevDiffSize.number; pn++)
	{
		for (int pc = 0; pc < prevDiffSize.channels; pc++)
		{
			for (int nc = 0; nc < nextDiffSize.channels; nc++)
			{
				const int data_idx = pc;
				const int weight_idx = nc*prevDiffSize.channels + pc;
				prevDiff[data_idx] += weight[weight_idx];
			}
		}
	}

	//update this layer's param
	//update weight
	{
		//get weight diff
		std::shared_ptr<DataBucket> weightDiffBucket(std::make_shared<DataBucket>(DataSize(1,weightSize.totalSize(),1,1)));
		weightDiffBucket->fillData(0.0f);
		float* weightDiff = weightDiffBucket->getData().get();
		for (int pn = 0; pn < prevDataSize.number; pn++)
		{
			for (int nc = 0; nc < nextDataSize.channels; nc++)
			{
				for (int pc = 0; pc < prevDataSize.channels; pc++)
				{
					const int weight_idx = nc*prevDataSize.channels + pc;
					weightDiff[weight_idx] += prevData[pc];
				}
			}
		}
		for (int nc = 0; nc < nextDataSize.channels; nc++)
		{
			for (int pc = 0; pc < prevDataSize.channels; pc++)
			{
				const int weight_idx = nc*prevDataSize.channels + pc;
				weight[weight_idx] *= nextDiff[nc];
			}
		}
		//apply change
		for (int nc = 0; nc < nextDataSize.channels; nc++)
		{
			for (int pc = 0; pc < prevDataSize.channels; pc++)
			{
				const int weight_idx = nc*prevDataSize.channels + pc;
				weight[weight_idx] -= getLearningRate() * weightDiff[pc];
			}
		}
	}
	//update bias
	if (enabledBias)
	{
		//get bias diff
		std::shared_ptr<DataBucket> biasDiffBucket(std::make_shared<DataBucket>(DataSize(1, biasSize.totalSize(), 1, 1)));
		biasDiffBucket->fillData(0.0f);
		float* biastDiff = biasDiffBucket->getData().get();
		for (int nc = 0; nc < nextDataSize.channels; nc++)
		{
			const int bias_idx = nc;
			biastDiff[bias_idx] += 1.0f;
		}
		for (int nc = 0; nc < nextDataSize.channels; nc++)
		{
			const int bias_idx = nc;
			biastDiff[bias_idx] *= nextDiff[nc];
		}
		//apply change
		for (int nc = 0; nc < nextDataSize.channels; nc++)
		{
			const int bias_idx = nc;
			biastDiff[bias_idx] -= getLearningRate() * biastDiff[bias_idx];
		}
	}

	//chain goto previous layer
	nextDiffBucket = prevDiffBucket;
}