#include "EasyCNN/ConvolutionLayer.h"

EasyCNN::ConvolutionLayer::ConvolutionLayer()
{

}
EasyCNN::ConvolutionLayer::~ConvolutionLayer()
{

}
void EasyCNN::ConvolutionLayer::setParamaters(const BucketSize _kernelSize, const int _step, const bool _enabledBias)
{
	easyAssert(_kernelSize.channels >= 1 && _kernelSize.width > 0 && _kernelSize.height > 0 && _step > 0, "kernel size or step is invalidate.");
	kernelSize = _kernelSize;
	step = _step;
	enabledBias = _enabledBias;
}
DEFINE_LAYER_TYPE(EasyCNN::ConvolutionLayer, "ConvolutionLayer");
std::string EasyCNN::ConvolutionLayer::getLayerType() const
{
	return layerType;
}
void EasyCNN::ConvolutionLayer::solveInnerParams()
{
	const BucketSize inputSize = getInputBucketSize();
	BucketSize outputSize;
	outputSize.channels = kernelSize.channels;
	//FIXME : with step edge
	outputSize.width = inputSize.width / step;
	outputSize.height = inputSize.height / step;
	setOutpuBuckerSize(outputSize);
	easyAssert(inputSize.channels >= 1 && inputSize.width > 0 && inputSize.height > 0, "input size is invalidate.");
	easyAssert(outputSize.channels >= 1 && outputSize.width > 0 && outputSize.height > 0, "output size is invalidate.");
	easyAssert(kernelSize.channels >= 1 && kernelSize.width > 0 && kernelSize.height > 0 && step > 0 , "kernel size or step is invalidate.");
	kernelData.reset(new DataBucket(kernelSize));
	if (enabledBias)
	{
		biasData.reset(new DataBucket(BucketSize(kernelSize.channels, 1, 1)));
	}
}
void EasyCNN::ConvolutionLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");
	easyAssert(kernelSize.channels == outputSize.channels, "kernel channel must be equals with nextDataBucket");

	const data_type* prevRawData = prevDataBucket->getData().get();
	const data_type* kernelRawData = kernelData->getData().get();
	const data_type* biasRawData = enabledBias ? biasData->getData().get() : nullptr;
	data_type* nextRawData = nextDataBucket->getData().get();
	for (int oc = 0; oc < kernelSize.channels;oc++)
	{
		for (int oh = 0; oh < outputSize.height;oh++)
		{
			for (int ow = 0; ow < outputSize.width;ow++)
			{
				//FIXME : with step
				const int inStartX = ow*step;
				const int inStartY = oh*step;
				const int outIdx = oc*outputSize.height*outputSize.width + oh*outputSize.width + ow;
				data_type sum = 0;
				for (int ic = 0; ic < inputSize.channels; ic++)
				{
					for (int kh = 0; kh < kernelSize.height;kh++)
					{
						for (int kw = 0; kw < kernelSize.width;kw++)
						{
							const int inIdx = ic*inputSize.height*inputSize.width + (inStartY+kh)*inputSize.width + (inStartX+kw);
							const int kernelIdx = oc*kernelSize.height*kernelSize.width + kh*kernelSize.width + kw;
							sum += prevRawData[inIdx] * kernelRawData[kernelIdx];
						}
					}
				}
				sum /= inputSize.channels;
				if (enabledBias)
				{
					sum += biasRawData[oc];
				}
				nextRawData[outIdx] = sum;
			}
		}
	}
}
void EasyCNN::ConvolutionLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}