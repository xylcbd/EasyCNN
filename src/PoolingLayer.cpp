#include <algorithm>
#include "EasyCNN/PoolingLayer.h"

EasyCNN::PoolingLayer::PoolingLayer()
{

}
EasyCNN::PoolingLayer::~PoolingLayer()
{

}
void EasyCNN::PoolingLayer::setParamaters(const PoolingType _poolingType, const ParamSize _poolingKernelSize, const size_t _widthStep, const size_t _heightStep)
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
	const DataSize inputSize = getInputBucketSize();
	poolingKernelSize.number = 1;
	poolingKernelSize.channels = inputSize.channels;
	easyAssert(inputSize.number && poolingKernelSize.number && inputSize.channels == poolingKernelSize.channels &&
		inputSize.width > poolingKernelSize.width && inputSize.height > poolingKernelSize.height, 
		"poolingKernelSize parameters invalidate.");
	DataSize outputSize;
	outputSize.number = inputSize.number;
	outputSize.channels = inputSize.channels;
	outputSize.width = (inputSize.width-poolingKernelSize.width)/widthStep+1;
	outputSize.height = (inputSize.height-poolingKernelSize.height)/heightStep+1;
	setOutpuBuckerSize(outputSize);

	if (poolingType == PoolingType::MaxPooling)
	{
		maxIdxBucket.reset(new ParamBucket(ParamSize(outputSize.number, outputSize.channels, outputSize.height, outputSize.width)));
	}
}
void EasyCNN::PoolingLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	const ParamSize maxIdxSize = maxIdxBucket->getSize();
	easyAssert(inputSize.number == outputSize.number && inputSize.channels == outputSize.channels &&
		inputSize.height >= outputSize.height && inputSize.width >= outputSize.height, "input size & output size is invalidate.");
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");

	for (size_t on = 0; on < outputSize.number; on++)
	{
		const float* prevRawData = prevDataBucket->getData().get() + on*inputSize._3DSize();
		float* nextRawData = nextDataBucket->getData().get() + on*outputSize._3DSize();
		float* maxIdxRawData = maxIdxBucket->getData().get() + on*maxIdxSize._3DSize();
		for (size_t oc = 0; oc < outputSize.channels; oc++)
		{
			for (size_t oh = 0; oh < outputSize.height; oh++)
			{
				for (size_t ow = 0; ow < outputSize.width; ow++)
				{
					const size_t inStartX = ow*widthStep;
					const size_t inStartY = oh*heightStep;
					const size_t outIdx = outputSize.getIndex(oc, oh, ow);
					float result = 0;
					size_t maxIdx = 0;
					if (poolingType == PoolingType::MaxPooling)
					{
						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t inIdx = inputSize.getIndex(oc, inStartY + ph, inStartX + pw);
								if (result > prevRawData[inIdx])
								{
									result = prevRawData[inIdx];
									maxIdx = ph*poolingKernelSize.width + pw;
								}
							}
						}
						//FIXME : using int type!!
						maxIdxRawData[outIdx] = maxIdx;
					}
					else if (poolingType == PoolingType::MeanPooling)
					{
						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t inIdx = inputSize.getIndex(oc, inStartY + ph, inStartX + pw);
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
void EasyCNN::PoolingLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const ParamSize nextDiffSize = nextDiffBucket->getSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	easyAssert(maxIdxBucket->getSize()._3DSize() == nextDataSize._3DSize(),"idx size must equals with next data.");

	//update prevDiff data
	const float* maxIdxRawData = maxIdxBucket->getData().get();
	const ParamSize prevDiffSize(1, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	std::shared_ptr<ParamBucket> prevDiffBucket(std::make_shared<ParamBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff 
	//none
	//pass next layer's diff to previous layer
	for (size_t pn = 0; pn < prevDataSize.number; pn++)
	{
		for (size_t nc = 0; nc < nextDataSize.channels; nc++)
		{
			for (size_t nh = 0; nh < nextDataSize.height; nh++)
			{
				for (size_t nw = 0; nw < nextDataSize.width; nw++)
				{
					const size_t inStartX = nw*widthStep;
					const size_t inStartY = nh*heightStep;
					const size_t nextDataIdx = nextDataSize.getIndex(nc, nh, nw);
					if (poolingType == PoolingType::MaxPooling)
					{
						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t prevDiffIdx = prevDataSize.getIndex(nc, inStartY + ph, inStartX + pw);
								if (ph*poolingKernelSize.width + pw == maxIdxRawData[nextDataIdx])
								{
									prevDiff[prevDiffIdx] += nextDiff[nextDataIdx];
								}
							}
						}
					}else if (poolingType == PoolingType::MeanPooling)
					{
						const float meanDiff = nextDiff[nextDataIdx] / (float)(poolingKernelSize._2DSize());
						for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
						{
							for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
							{
								const size_t prevDiffIdx = prevDataSize.getIndex(nc, inStartY + ph, inStartX + pw);
								prevDiff[prevDiffIdx] += meanDiff;
							}
						}
					}
				}
			}
		}
	}

	//update this layer's param
	//nop

	nextDiffBucket = prevDiffBucket;
}