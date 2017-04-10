#include <algorithm>
#include "EasyCNN/SoftmaxLayer.h"


namespace EasyCNN
{
	//////////////////////////////////////////////////////////////////////////
	//normal softmax
	SoftmaxLayer::SoftmaxLayer()
	{

	}
	SoftmaxLayer::~SoftmaxLayer()
	{

	}
	DEFINE_LAYER_TYPE(SoftmaxLayer, "SoftmaxLayer");
	std::string SoftmaxLayer::getLayerType() const
	{
		return layerType;
	}
	void SoftmaxLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevDataSize = prev->getSize();
		const DataSize nextDataSize = next->getSize();

		for (size_t nn = 0; nn < nextDataSize.number; nn++)
		{
			const float* prevData = prev->getData().get() + nn*prevDataSize._3DSize();
			float* nextData = next->getData().get() + nn*nextDataSize._3DSize();

			//step1 : find max value
			float maxVal = prevData[0];
			for (size_t prevDataIdx = 0; prevDataIdx < prevDataSize._3DSize(); prevDataIdx++)
			{
				maxVal = std::max(maxVal, prevData[prevDataIdx]);
			}
			//step2 : sum
			float sum = 0;
			for (size_t prevDataIdx = 0; prevDataIdx < prevDataSize._3DSize(); prevDataIdx++)
			{
				nextData[prevDataIdx] = std::exp(prevData[prevDataIdx] - maxVal);
				sum += nextData[prevDataIdx];
			}
			//step3 : div
			for (size_t prevDataIdx = 0; prevDataIdx < prevDataSize._3DSize(); prevDataIdx++)
			{
				nextData[prevDataIdx] = nextData[prevDataIdx] / sum;
			}
		}
	}
	void SoftmaxLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		easyAssert(prevSize == nextSize, "data size must be equal!");
		easyAssert(nextDiffSize == nextSize, "next data's and diff's size must be equal! ");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");
		easyAssert(prevDiffSize == nextDiffSize, "diff size must be equal!");
		
		//update prevDiff	
		prevDiff->fillData(0.0f);
		for (size_t pn = 0; pn < prevSize.number; pn++)
		{
			const float* prevData = prev->getData().get() + pn*prevSize._3DSize();
			const float* nextData = next->getData().get() + pn*nextSize._3DSize();
			const float* nextDiffData = nextDiff->getData().get() + pn*nextDiffSize._3DSize();
			float* prevDiffData = prevDiff->getData().get() + pn*prevDiffSize._3DSize();
			for (size_t prevDiffIdx = 0; prevDiffIdx < prevDiffSize._3DSize(); prevDiffIdx++)
			{
				for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
				{
					if (nextDiffIdx == prevDiffIdx)
					{
						prevDiffData[prevDiffIdx] += nextData[prevDiffIdx] * (1.0f - nextData[prevDiffIdx]) * nextDiffData[nextDiffIdx];
					}
					else
					{
						prevDiffData[prevDiffIdx] -= nextData[prevDiffIdx] * nextData[nextDiffIdx] * nextDiffData[nextDiffIdx];
					}
				}
			}
		}

		//update this layer's param
		//softmax layer : nop
	}
}//namespace