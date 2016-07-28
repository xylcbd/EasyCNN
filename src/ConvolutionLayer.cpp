#include "EasyCNN/ConvolutionLayer.h"
#include "EasyCNN/CommonTools.h"

EasyCNN::ConvolutionLayer::ConvolutionLayer()
{

}
EasyCNN::ConvolutionLayer::~ConvolutionLayer()
{

}
void EasyCNN::ConvolutionLayer::setParamaters(const ParamSize _kernelSize, const int _widthStep, const int _heightStep, const bool _enabledBias)
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
	const DataSize inputSize = getInputBucketSize();
	kernelSize.channels = inputSize.channels;
	easyAssert(inputSize.number > 0 && inputSize.channels > 0 && inputSize.width > 0 && inputSize.height > 0, "input size is invalidate.");
	easyAssert(kernelSize.number > 0 && kernelSize.channels > 0 && kernelSize.width > 0 && kernelSize.height > 0 && widthStep > 0 && heightStep > 0,
		"kernel size or step is invalidate.");
	DataSize outputSize;
	outputSize.number = inputSize.number;
	outputSize.channels = kernelSize.number;
	outputSize.width = (inputSize.width-kernelSize.width) / widthStep + 1;
	outputSize.height = (inputSize.height-kernelSize.height) / heightStep + 1;
	setOutpuBuckerSize(outputSize);
	easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width > 0 && outputSize.height > 0, "output size is invalidate.");
	kernelData.reset(new ParamBucket(kernelSize));
	normal_distribution_init(kernelData->getData().get(), kernelData->getSize().totalSize(),0.0f,01.f);

	if (enabledBias)
	{
		biasData.reset(new ParamBucket(ParamSize(kernelSize.number, 1, 1, 1)));
		const_distribution_init(biasData->getData().get(), biasData->getSize().totalSize(), 0.0f);
	}
}
void EasyCNN::ConvolutionLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");
	easyAssert(inputSize.number == outputSize.number && kernelSize.number == outputSize.channels && kernelSize.channels == inputSize.channels, 
		"kernel number/channel must be equals with nextDataBucket");

	const float* kernelRawData = kernelData->getData().get();
	const float* biasRawData = biasData->getData().get();
	for (int on = 0; on < outputSize.number; on++)
	{	
		const float* prevRawData = prevDataBucket->getData().get() + on*inputSize.channels*inputSize.height*inputSize.width;		
		for (int oc = 0; oc < outputSize.channels; oc++)
		{			
			float* nextRawData = nextDataBucket->getData().get() + on*outputSize.channels*outputSize.height*outputSize.width + 
				oc*outputSize.height*outputSize.width;
			for (int oh = 0; oh < outputSize.height; oh++)
			{
				for (int ow = 0; ow < outputSize.width; ow++)
				{
					const int inStartX = ow*widthStep;
					const int inStartY = oh*heightStep;
					const int outIdx = oh*outputSize.width + ow;
					float sum = 0;
					for (int ic = 0; ic < inputSize.channels; ic++)
					{
						for (int kh = 0; kh < kernelSize.height; kh++)
						{
							for (int kw = 0; kw < kernelSize.width; kw++)
							{
								const int inIdx = inputSize.getIndex(ic,inStartY+kh,inStartX+kw);
								const int kernelIdx = kernelSize.getIndex(ic, kh, kw);
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
void EasyCNN::ConvolutionLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket)
{

}