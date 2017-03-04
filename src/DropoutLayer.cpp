#include <random>
#include <ctime>
#include "EasyCNN/DropoutLayer.h"
#include "EasyCNN/CommonTools.h"
#include "EasyCNN/MathFunctions.h"

namespace EasyCNN
{
	DropoutLayer::DropoutLayer()
	{

	}
	DropoutLayer::DropoutLayer(const float _rate)
	{
		setParamaters(_rate);
	}
	DropoutLayer::~DropoutLayer()
	{

	}
	DEFINE_LAYER_TYPE(DropoutLayer, "DropoutLayer");
	std::string DropoutLayer::getLayerType() const
	{
		return layerType;
	}
	void DropoutLayer::setParamaters(const float _rate)
	{
		rate = _rate;
	}
	std::string DropoutLayer::serializeToString() const
	{
		const std::string spliter = " ";
		std::stringstream ss;
		//layer desc
		ss << getLayerType() << rate << spliter;
		return ss.str();
	}
	void DropoutLayer::serializeFromString(const std::string content)
	{
		std::stringstream ss(content);
		//layer desc
		std::string _layerType;
		ss >> _layerType
			>> rate;
		easyAssert(_layerType == getLayerType(), "layer type is invalidate.");
	}
	void DropoutLayer::solveInnerParams()
	{
		if (!mask.get())
		{
			ParamSize maskSize = getInputBucketSize();
			maskSize.number = 1;
			mask.reset(new ParamBucket(maskSize));
			const_distribution_init(mask->getData().get(), maskSize.totalSize(), 1.0f);
		}
		setOutpuBuckerSize(getInputBucketSize());
	}
	void DropoutLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		easyAssert(prevSize == nextSize, "size must be equal!");

		//init rand seed
		std::srand((unsigned int)std::time(nullptr));

		if (getPhase() == Phase::Train)
		{
			//fill mask
			float* maskData = mask->getData().get();
			std::random_device rd;
			std::mt19937 engine(rd());
			std::bernoulli_distribution random_distribution(rate);
			for (size_t i = 0; i < mask->getSize().totalSize(); i++) {
				maskData[i] = (float)(random_distribution(engine));
			}

			const float* prevData = prev->getData().get();
			float* nextData = next->getData().get();
			for (size_t i = 0; i < nextSize.number; i++)
			{
				for (size_t j = 0; j < nextSize._3DSize(); j++)
				{
					const int dataIdx = i*nextSize._3DSize() + j;
					nextData[dataIdx] = prevData[dataIdx] * maskData[j] / rate;
				}
			}
		}
		else
		{
			const float* prevData = prev->getData().get();
			float* nextData = next->getData().get();
			for (size_t i = 0; i < nextSize.totalSize(); i++)
			{
				nextData[i] = prevData[i];
			}
		}
	}
	void DropoutLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = nextDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//////////////////////////////////////////////////////////////////////////
		//update prevDiff
		prevDiff->fillData(0.0f);
		const float* maskData = mask->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		//calculate current inner diff && multiply next diff
		for (size_t i = 0; i < nextSize.number; i++)
		{
			for (size_t j = 0; j < nextSize._3DSize(); j++)
			{
				const int dataIdx = i*nextSize._3DSize() + j;
				prevDiffData[dataIdx] = nextDiffData[dataIdx] * maskData[j] / rate;
			}
		}
	}
}//namespace