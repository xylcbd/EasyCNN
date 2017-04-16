#include "EasyCNN/FullconnectLayer.h"
#include "EasyCNN/CommonTools.h"
#include "EasyCNN/MathFunctions.h"
#include "EasyCNN/ThreadPool.h"

namespace EasyCNN
{
	FullconnectLayer::FullconnectLayer()
	{

	}
	FullconnectLayer::~FullconnectLayer()
	{

	}
	DEFINE_LAYER_TYPE(FullconnectLayer, "FullconnectLayer");
	std::string FullconnectLayer::getLayerType() const
	{
		return layerType;
	}
	void FullconnectLayer::setParamaters(const ParamSize _outMapSize, const bool _enabledBias)
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
	std::string FullconnectLayer::serializeToString() const
	{
		const std::string spliter = " ";
		std::stringstream ss;
		//layer desc
		ss << getLayerType() << spliter
			<< outMapSize.number << spliter << outMapSize.channels << spliter << outMapSize.width << spliter << outMapSize.height << spliter
			<< enabledBias << spliter;
		//weight
		const auto weightData = weight->getData().get();
		const auto weightSize = weight->getSize();
		for (size_t i = 0; i < weightSize.totalSize(); i++)
		{
			ss << weightData[i] << spliter;
		}
		//bias
		if (enabledBias)
		{
			const auto biasData = bias->getData().get();
			const auto biasSize = bias->getSize();
			for (size_t i = 0; i < biasSize.totalSize(); i++)
			{
				ss << biasData[i] << spliter;
			}
		}
		return ss.str();
	}
	void FullconnectLayer::serializeFromString(const std::string content)
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
		const auto weightData = weight->getData().get();
		const auto weightSize = weight->getSize();
		for (size_t i = 0; i < weightSize.totalSize(); i++)
		{
			ss >> weightData[i];
		}
		//bias
		if (enabledBias)
		{
			const auto biasData = bias->getData().get();
			const auto biasSize = bias->getSize();
			for (size_t i = 0; i < biasSize.totalSize(); i++)
			{
				ss >> biasData[i];
			}
		}
	}
	void FullconnectLayer::solveInnerParams()
	{
		const DataSize inputSize = getInputBucketSize();
		DataSize outputSize = getOutputBucketSize();
		outputSize.number = inputSize.number;
		setOutpuBuckerSize(outputSize);
		easyAssert(inputSize.number > 0 && inputSize.channels > 0 && inputSize.width > 0 && inputSize.height > 0, "input size or step is invalidate.");
		easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width == 1 && outputSize.height == 1, "output size is invalidate.");
		if (weight.get() == nullptr)
		{
			weight.reset(new ParamBucket(ParamSize(1, inputSize._3DSize()*outputSize._3DSize(), 1, 1)));
			normal_distribution_init(weight->getData().get(), weight->getSize().totalSize(), 0.0f, 0.1f);
			/*
			const size_t fan_in = inputSize._3DSize();
			const size_t fan_out = outputSize._3DSize();
			xavier_init(weight->getData().get(), weight->getSize().totalSize(), fan_in, fan_out);
			*/
		}
		if (weightGradient.get() == nullptr)
		{
			weightGradient.reset(new ParamBucket(weight->getSize()));
			const_distribution_init(weightGradient->getData().get(), weightGradient->getSize().totalSize(), 0.0f);
		}
		if (enabledBias)
		{
			if (bias.get() == nullptr)
			{
				bias.reset(new ParamBucket(ParamSize(1, outputSize.channels, 1, 1)));
				const_distribution_init(bias->getData().get(), bias->getSize().totalSize(), 0.0f);
			}
			if (biasGradient.get() == nullptr)
			{
				biasGradient.reset(new ParamBucket(bias->getSize()));
				const_distribution_init(biasGradient->getData().get(), biasGradient->getSize().totalSize(), 0.0f);
			}
		}
		//parmas
		params.clear();
		params.push_back(weight);
		params.push_back(bias);
		//diffs
		gradients.clear();
		gradients.push_back(weightGradient);
		gradients.push_back(biasGradient);
	}
	void FullconnectLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();

		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();		
		const float* weightData = weight->getData().get();
		const float* biasData = enabledBias ? bias->getData().get() : nullptr;
		auto worker = [&](const size_t start, const size_t stop){
			fullconnect(prevData + start * prevSize._3DSize(), weightData, biasData, nextData + start * nextSize._3DSize(), stop-start, prevSize._3DSize(), nextSize._3DSize());
		};
		dispatch_worker(worker,prevSize.number);
	}

	void FullconnectLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const ParamSize weightSize = weight->getSize();
		const ParamSize biasSize = enabledBias ? bias->getSize() : ParamSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		const float* weightData = weight->getData().get();
		const float* biasData = enabledBias ? bias->getData().get() : nullptr;
		easyAssert(nextSize.width == 1 && nextSize.height == 1, "use channel only!");
		easyAssert(weightSize.totalSize() == prevSize._3DSize() * nextSize._3DSize(), "weight size is invalidate!");
		if (enabledBias)
		{
			easyAssert(biasSize.totalSize() == nextSize._3DSize(), "bias size is invalidate!");
		}
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//////////////////////////////////////////////////////////////////////////
		//update prevDiff
		prevDiff->fillData(0.0f);		
		//calculate current inner diff && multiply next diff
		auto worker = [&](const size_t start, const size_t stop){
			for (size_t pn = start; pn < stop; pn++)
			{
				for (size_t pidx = 0; pidx < prevDiffSize._3DSize(); pidx++)
				{
					const size_t prevDiffIdx = pn * prevDiffSize._3DSize() + pidx;
					for (size_t nc = 0; nc < nextDiffSize.channels; nc++)
					{
						const size_t weightIdx = nc*prevSize._3DSize() + pidx;
						const size_t nextDiffIdx = pn*nextDiffSize._3DSize() + nc;
						prevDiffData[prevDiffIdx] += weightData[weightIdx] * nextDiffData[nextDiffIdx];
					}
				}
			}
		};
		dispatch_worker(worker, prevSize.number);

		//////////////////////////////////////////////////////////////////////////
		//update this layer's param
		//get weight gradient
		weightGradient->fillData(0.0f);
		float* weightGradientData = weightGradient->getData().get();
		for (size_t pn = 0; pn < nextSize.number; pn++)
		{
			for (size_t nc = 0; nc < nextSize.channels; nc++)
			{
				const size_t nextDiffIdx = pn*nextDiffSize._3DSize() + nc;
				for (size_t prevData3DIdx = 0; prevData3DIdx < prevSize._3DSize(); prevData3DIdx++)
				{
					const size_t weightGradientIdx = nc*prevDiffSize._3DSize() + prevData3DIdx;
					const size_t prevDataIdx = pn*prevSize._3DSize() + prevData3DIdx;
					weightGradientData[weightGradientIdx] += prevData[prevDataIdx] * nextDiffData[nextDiffIdx];
				}
			}
		}
		//div by batch size
		div_inplace(weightGradientData, (float)nextSize.number, weightSize.totalSize());

		//////////////////////////////////////////////////////////////////////////
		//update bias
		if (enabledBias)
		{
			//get bias diff		
			biasGradient->fillData(0.0f);
			float* biasGradientData = biasGradient->getData().get();
			for (size_t nn = 0; nn < nextSize.number; nn++)
			{
				for (size_t biasDiffIdx = 0; biasDiffIdx < biasSize._3DSize(); biasDiffIdx++)
				{
					biasGradientData[biasDiffIdx] += 1.0f*nextDiffData[nn*biasSize._3DSize() + biasDiffIdx];
				}
			}
			//div by batch size
			div_inplace(biasGradientData, (float)nextSize.number, biasSize.totalSize());
		}
	}
}//namespace