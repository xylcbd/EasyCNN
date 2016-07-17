#include "EasyCNN/ConvolutionLayer.h"

EasyCNN::ConvolutionLayer::ConvolutionLayer()
{

}
EasyCNN::ConvolutionLayer::~ConvolutionLayer()
{

}
void EasyCNN::ConvolutionLayer::setParamaters(const BucketSize _kernelSize, const int _widthStep, const int _heightStep, const bool _enabledBias)
{
	easyAssert(_kernelSize.number > 0 && _kernelSize.channels > 0 && 
		_kernelSize.width > 0 && _kernelSize.height > 0 && _widthStep > 0 && _heightStep > 0,
		"kernel size or step is invalidate.");
	kernelSize = _kernelSize;
	widthStep = _widthStep;
	heightStep = _heightStep;
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
	kernelSize.channels = inputSize.channels;
	easyAssert(inputSize.number > 0 && inputSize.channels > 0 && inputSize.width > 0 && inputSize.height > 0, "input size is invalidate.");
	easyAssert(kernelSize.number > 0 && kernelSize.channels > 0 && kernelSize.width > 0 && kernelSize.height > 0 && widthStep > 0 && heightStep > 0,
		"kernel size or step is invalidate.");
	BucketSize outputSize;
	outputSize.number = inputSize.number;
	outputSize.channels = kernelSize.number;
	outputSize.width = (inputSize.width-kernelSize.width) / widthStep + 1;
	outputSize.height = (inputSize.height-kernelSize.height) / heightStep + 1;
	setOutpuBuckerSize(outputSize);
	easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width > 0 && outputSize.height > 0, "output size is invalidate.");
	kernelData.reset(new DataBucket(kernelSize));
	if (enabledBias)
	{
		biasData.reset(new DataBucket(BucketSize(kernelSize.number, 1, 1, 1)));
	}
}
void EasyCNN::ConvolutionLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const BucketSize inputSize = getInputBucketSize();
	const BucketSize outputSize = getOutputBucketSize();
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");
	easyAssert(inputSize.number == outputSize.number && kernelSize.number == outputSize.channels && kernelSize.channels == inputSize.channels, 
		"kernel number/channel must be equals with nextDataBucket");

	const data_type* biasRawData = enabledBias ? biasData->getData().get() : nullptr;
	data_type* nextRawData = nextDataBucket->getData().get();
	for (int on = 0; on < outputSize.number; on++)
	{
		const data_type* prevRawData = prevDataBucket->getData().get() + on*inputSize.channels*inputSize.height*inputSize.width;
		const data_type* kernelRawData = kernelData->getData().get() + on*outputSize.channels*outputSize.height*outputSize.width;
		for (int oc = 0; oc < outputSize.channels; oc++)
		{
			for (int oh = 0; oh < outputSize.height; oh++)
			{
				for (int ow = 0; ow < outputSize.width; ow++)
				{
					const int inStartX = ow*widthStep;
					const int inStartY = oh*heightStep;
					const int outIdx = oc*outputSize.height*outputSize.width +
						oh*outputSize.width +
						ow;
					data_type sum = 0;
					for (int ic = 0; ic < inputSize.channels; ic++)
					{
						for (int kh = 0; kh < kernelSize.height; kh++)
						{
							for (int kw = 0; kw < kernelSize.width; kw++)
							{
								const int inIdx = ic*inputSize.height*inputSize.width +
									(inStartY + kh)*inputSize.width +
									(inStartX + kw);
								const int kernelIdx = ic*kernelSize.height*kernelSize.width + kh*kernelSize.width + kw;
								sum += prevRawData[inIdx] * kernelRawData[kernelIdx];
							}
						}
					}
					if (enabledBias)
					{
						sum += biasRawData[oc];
					}
					nextRawData[outIdx] = sum;
				}
			}
		}
	}
}
void EasyCNN::ConvolutionLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket)
{

}