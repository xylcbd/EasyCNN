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
void EasyCNN::FullconnectLayer::setParamaters(const ParamSize _outMapSize, const bool _enabledBias)
{
	outMapSize = _outMapSize;
	enabledBias = _enabledBias;
	DataSize outputSize;
	outputSize.number = _outMapSize.number;
	outputSize.channels = _outMapSize.channels;
	outputSize.width = _outMapSize.width;
	outputSize.height = _outMapSize.height;
	setOutpuBuckerSize(outputSize);
}
std::string EasyCNN::FullconnectLayer::serializeToString() const
{
	const std::string spliter = " ";
	std::stringstream ss;
	//layer desc
	ss << getLayerType() << spliter
		<< outMapSize.number << spliter << outMapSize.channels << spliter << outMapSize.width << spliter << outMapSize.height << spliter
		<< enabledBias << spliter;
	//weight
	const auto weight = weightsData->getData().get();
	const auto weightSize = weightsData->getSize();
	for (size_t i = 0; i < weightSize._4DSize(); i++)
	{
		ss << weight[i] << spliter;
	}
	//bias
	if (enabledBias)
	{
		const auto bias = biasData->getData().get();
		const auto biasSize = biasData->getSize();
		for (size_t i = 0; i < biasSize._4DSize(); i++)
		{
			ss << bias[i] << spliter;
		}
	}
	return ss.str();
}
void EasyCNN::FullconnectLayer::serializeFromString(const std::string content)
{
	std::stringstream ss(content);
	//layer desc
	std::string _layerType;
	ss >> _layerType
		>> outMapSize.number >> outMapSize.channels >> outMapSize.width >> outMapSize.height
		>> enabledBias;
	easyAssert(_layerType == getLayerType(), "layer type is invalidate.");
	DataSize outputSize;
	outputSize.number = outMapSize.number;
	outputSize.channels = outMapSize.channels;
	outputSize.width = outMapSize.width;
	outputSize.height = outMapSize.height;
	setOutpuBuckerSize(outputSize);
	solveInnerParams();
	//weight
	const auto weight = weightsData->getData().get();
	const auto weightSize = weightsData->getSize();
	for (size_t i = 0; i < weightSize._4DSize(); i++)
	{
		ss >> weight[i];
	}
	//bias
	if (enabledBias)
	{
		const auto bias = biasData->getData().get();
		const auto biasSize = biasData->getSize();
		for (size_t i = 0; i < biasSize._4DSize(); i++)
		{
			ss >> bias[i];
		}
	}
}
void EasyCNN::FullconnectLayer::solveInnerParams()
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(inputSize.number > 0 && inputSize.channels > 0 && inputSize.width > 0 && inputSize.height > 0, "input size or step is invalidate.");
	easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1, "output size is invalidate.");
	if (weightsData.get() == nullptr)
	{
		weightsData.reset(new ParamBucket(ParamSize(1, inputSize._3DSize()*outputSize._3DSize(), 1, 1)));
		normal_distribution_init(weightsData->getData().get(), weightsData->getSize()._4DSize(), 0.0f, 01.f);
	}
	if (enabledBias)
	{
		if (biasData.get() == nullptr)
		{
			biasData.reset(new ParamBucket(ParamSize(1, outputSize.channels, 1, 1)));
			const_distribution_init(biasData->getData().get(), biasData->getSize()._4DSize(), 0.0f);
		}
	}
}
void EasyCNN::FullconnectLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();

	const float* prevData = prevDataBucket->getData().get();
	float* nextData = nextDataBucket->getData().get();
	const float* weights = weightsData->getData().get();
	const float* bias = enabledBias ? biasData->getData().get() : nullptr;
	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{
		for (size_t nc = 0; nc < nextDataSize.channels; nc++)
		{			
			float sum = 0;
			for (size_t pc = 0; pc < prevDataSize.channels; pc++)
			{
				for (size_t ph = 0; ph < prevDataSize.height; ph++)
				{
					for (size_t pw = 0; pw < prevDataSize.width; pw++)
					{
						const size_t prevDataIdx = prevDataSize.getIndex(nn, pc, ph, pw);
						const size_t weightsIdx = nc*prevDataSize._3DSize() + prevDataSize.getIndex(pc, ph, pw);
						sum += prevData[prevDataIdx] * weights[weightsIdx];
					}
				}
			}
			if (enabledBias)
			{
				const size_t biasIdx = nc;
				sum += bias[biasIdx];
			}
			const size_t nextDataIdx = nn*nextDataSize._3DSize() + nc;
			nextData[nextDataIdx] = sum;
		}//oc
	}//on
}
void EasyCNN::FullconnectLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
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
					const size_t prevDiffIdx = prevDiffSize.getIndex(pc, ph, pw);
					for (size_t nc = 0; nc < nextDiffSize.channels; nc++)
					{
						const size_t weightIdx = nc*prevDataSize._3DSize() + prevDataSize.getIndex(pc, ph, pw);
						const size_t nextDiffIdx = nc;
						prevDiff[prevDiffIdx] += weight[weightIdx] * nextDiff[nextDiffIdx] / nextDataSize.number;
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
				weightDiff[weightDiffIdx] += prevData[prevDataIdx] * nextDiff[nextDiffIdx] / nextDataSize.number;
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
		float* biasDiff = biasDiffBucket->getData().get();
		for (size_t biasDiffIdx = 0; biasDiffIdx < biasSize._4DSize(); biasDiffIdx++)
		{
			biasDiff[biasDiffIdx] += 1.0f*nextDiff[biasDiffIdx] / nextDataSize.number;
		}
		//apply change
		for (size_t biasDiffIdx = 0; biasDiffIdx < biasSize._4DSize(); biasDiffIdx++)
		{
			bias[biasDiffIdx] -= getLearningRate() * biasDiff[biasDiffIdx];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//chain goto previous layer
	nextDiffBucket = prevDiffBucket;
}