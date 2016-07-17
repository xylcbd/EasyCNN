#include <algorithm>
#include "EasyCNN/PoolingLayer.h"

EasyCNN::PoolingLayer::PoolingLayer()
{

}
EasyCNN::PoolingLayer::~PoolingLayer()
{

}
void EasyCNN::PoolingLayer::setParamaters(const PoolingType _poolingType, const BucketSize _poolingKernelSize,const int _widthStep, const int _heightStep)
{
	easyAssert(_poolingKernelSize.number ==1 && _poolingKernelSize.channels > 0 && _poolingKernelSize.width > 1 && _poolingKernelSize.height > 1 && _widthStep>0 && _heightStep>0,
		"parameters invalidate.");
	poolingKernelSize = _poolingKernelSize;
	poolingType = _poolingType;
	widthStep = _widthStep;
	heightStep = _heightStep;
}
DEFINE_LAYER_TYPE(EasyCNN::PoolingLayer, "PoolingLayer");
std::string EasyCNN::PoolingLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::PoolingLayer::solveInnerParams()
{
	easyAssert(poolingKernelSize.number > 0 && poolingKernelSize.channels > 0 && poolingKernelSize.width > 1 && poolingKernelSize.height > 1, "poolingKernelSize parameters invalidate.");
	const BucketSize inputSize = getInputBucketSize();
	poolingKernelSize.number = 1;
	poolingKernelSize.channels = inputSize.channels;
	easyAssert(inputSize.number && poolingKernelSize.number && inputSize.channels == poolingKernelSize.channels &&
		inputSize.width > poolingKernelSize.width && inputSize.height > poolingKernelSize.height, 
		"poolingKernelSize parameters invalidate.");
	BucketSize outputSize;
	outputSize.number = inputSize.number;
	outputSize.channels = inputSize.channels;
	outputSize.width = (inputSize.width-poolingKernelSize.width)/widthStep+1;
	outputSize.height = (inputSize.height-poolingKernelSize.height)/heightStep+1;
	setOutpuBuckerSize(outputSize);
}
void EasyCNN::PoolingLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(inputSize.number == outputSize.number && inputSize.channels == outputSize.channels &&
		inputSize.height >= outputSize.height && inputSize.width >= outputSize.height, "input size & output size is invalidate.");
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");

	for (int on = 0; on < outputSize.number;on++)
	{
		const data_type* prevRawData = prevDataBucket->getData().get() + on*inputSize.channels*inputSize.height*inputSize.width;
		data_type* nextRawData = nextDataBucket->getData().get() + on*outputSize.channels*outputSize.height*outputSize.width;
		for (int oc = 0; oc < outputSize.channels; oc++)
		{
			for (int oh = 0; oh < outputSize.height; oh++)
			{
				for (int ow = 0; ow < outputSize.width; ow++)
				{
					const int inStartX = ow*widthStep;
					const int inStartY = oh*heightStep;
					const int outIdx = oc*outputSize.height*outputSize.width + oh*outputSize.width + ow;
					data_type result = 0;
					if (poolingType == PoolingType::MaxPooling)
					{
						for (int ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (int pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const int inIdx = oc*inputSize.height*inputSize.width + (inStartY + ph)*inputSize.width + (inStartX + pw);
								result = std::max(result, prevRawData[inIdx]);
							}
						}
					}
					else if (poolingType == PoolingType::MeanPooling)
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
				}//ow
			}//oh
		}//oc
	}//on
}
void EasyCNN::PoolingLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}