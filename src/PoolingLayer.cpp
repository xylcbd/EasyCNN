#include <algorithm>
#include "EasyCNN/PoolingLayer.h"

EasyCNN::PoolingLayer::PoolingLayer()
{

}
EasyCNN::PoolingLayer::~PoolingLayer()
{

}
void EasyCNN::PoolingLayer::setParamaters(const PoolingType _poolingType, const BucketSize _poolingKernelSize)
{
	easyAssert(_poolingKernelSize.channels == 1 && _poolingKernelSize.width > 1 && _poolingKernelSize.height > 1, "parameters invalidate.");
	poolingKernelSize = _poolingKernelSize;
	poolingType = _poolingType;
}
DEFINE_LAYER_TYPE(EasyCNN::PoolingLayer, "PoolingLayer");
std::string EasyCNN::PoolingLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::PoolingLayer::solveInnerParams()
{
	easyAssert(poolingKernelSize.channels == 1 && poolingKernelSize.width > 1 && poolingKernelSize.height > 1, "poolingKernelSize parameters invalidate.");
	const BucketSize inputSize = getInputBucketSize();
	easyAssert(inputSize.channels > 0 && inputSize.width > poolingKernelSize.width && inputSize.height > poolingKernelSize.height, "poolingKernelSize parameters invalidate.");
	BucketSize outputSize;
	outputSize.channels = inputSize.channels;
	outputSize.width = inputSize.width / poolingKernelSize.width;
	outputSize.height = inputSize.height / poolingKernelSize.height;
	setOutpuBuckerSize(outputSize);
}
void EasyCNN::PoolingLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");

	const data_type* prevRawData = prevDataBucket->getData().get();
	data_type* nextRawData = nextDataBucket->getData().get();
	for (int oc = 0; oc < outputSize.channels; oc++)
	{
		for (int oh = 0; oh < outputSize.height; oh++)
		{
			for (int ow = 0; ow < outputSize.width; ow++)
			{
				const int inStartX = ow*poolingKernelSize.width;
				const int inStartY = oh*poolingKernelSize.height;
				const int outIdx = oc*outputSize.height*outputSize.width + oh*outputSize.width + ow;
				data_type result = 0;
				if (poolingType == PoolingType::MaxPooling)
				{
					for (int ph = 0; ph < poolingKernelSize.height; ph++)
					{
						for (int pw = 0; pw < poolingKernelSize.width; pw++)
						{
							const int inIdx = oc*inputSize.height*inputSize.width + (inStartY + ph)*inputSize.width + (inStartX + pw);
							result = std::max(result,prevRawData[inIdx]);
						}
					}
				}else if (poolingType == PoolingType::MeanPooling)
				{
					for (int ph = 0; ph < poolingKernelSize.height; ph++)
					{
						for (int pw = 0; pw < poolingKernelSize.width; pw++)
						{
							const int inIdx = oc*inputSize.height*inputSize.width + (inStartY + ph)*inputSize.width + (inStartX + pw);
							result += prevRawData[inIdx];
						}
					}
					result /= poolingKernelSize.width*poolingKernelSize.height;
				}				
				nextRawData[outIdx] = result;
			}
		}
	}
}
void EasyCNN::PoolingLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}