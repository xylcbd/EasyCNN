#include "EasyCNN/ConvolutionLayer.h"
#include "EasyCNN/CommonTools.h"

EasyCNN::ConvolutionLayer::ConvolutionLayer()
{

}
EasyCNN::ConvolutionLayer::~ConvolutionLayer()
{

}
void EasyCNN::ConvolutionLayer::setParamaters(const ParamSize _kernelSize, const size_t _widthStep, const size_t _heightStep, const bool _enabledBias)
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
	normal_distribution_init(kernelData->getData().get(), kernelData->getSize()._4DSize(),0.0f,01.f);

	if (enabledBias)
	{
		biasData.reset(new ParamBucket(ParamSize(kernelSize.number, 1, 1, 1)));
		const_distribution_init(biasData->getData().get(), biasData->getSize()._4DSize(), 0.0f);
	}
}
void EasyCNN::ConvolutionLayer::forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket)
{
	const DataSize inputSize = getInputBucketSize();
	const DataSize outputSize = getOutputBucketSize();
	easyAssert(nextDataBucket->getSize() == outputSize, "outputSize must be equals with nextDataBucket's size.");
	easyAssert(inputSize.number == outputSize.number && kernelSize.number == outputSize.channels && kernelSize.channels == inputSize.channels, 
		"kernel number/channel must be equals with nextDataBucket");

	const float* prevRawData = prevDataBucket->getData().get();
	const float* kernelRawData = kernelData->getData().get();
	const float* biasRawData = biasData->getData().get();
	float* nextRawData = nextDataBucket->getData().get();
	for (size_t on = 0; on < outputSize.number; on++)
	{			
		for (size_t oc = 0; oc < outputSize.channels; oc++)
		{		
			for (size_t oh = 0; oh < outputSize.height; oh++)
			{
				for (size_t ow = 0; ow < outputSize.width; ow++)
				{
					const size_t inStartX = ow*widthStep;
					const size_t inStartY = oh*heightStep;
					const size_t outIdx = outputSize.getIndex(on, oc, oh, ow);
					float sum = 0;
					for (size_t kc = 0; kc < kernelSize.channels; kc++)
					{
						for (size_t kh = 0; kh < kernelSize.height; kh++)
						{
							for (size_t kw = 0; kw < kernelSize.width; kw++)
							{
								const size_t inIdx = inputSize.getIndex(on,kc, inStartY + kh, inStartX + kw);
								const size_t kernelIdx = kernelSize.getIndex(oc,kc, kh, kw);
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
void EasyCNN::ConvolutionLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();
	const ParamSize nextDiffSize = nextDiffBucket->getSize();
	const ParamSize biasSize = biasData->getSize();
	const float* prevData = prevDataBucket->getData().get();
	const float* nextData = nextDataBucket->getData().get();
	const float* nextDiff = nextDiffBucket->getData().get();
	float *kernel = kernelData->getData().get();
	float *bias = biasData->getData().get();

	//////////////////////////////////////////////////////////////////////////
	//update prevDiff data
	const ParamSize prevDiffSize(1, prevDataSize.channels, prevDataSize.height, prevDataSize.width);
	std::shared_ptr<ParamBucket> prevDiffBucket(std::make_shared<ParamBucket>(prevDiffSize));
	prevDiffBucket->fillData(0.0f);
	float* prevDiff = prevDiffBucket->getData().get();
	//calculate current inner diff
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
					const size_t nextDataIdx = nextDataSize.getIndex(pn,nc, nh, nw);
					const size_t nextDiffIdx = nextDataSize.getIndex(nc, nh, nw);
					for (size_t kc = 0; kc < kernelSize.channels; kc++)
					{
						for (size_t kh = 0; kh < kernelSize.height; kh++)
						{
							for (size_t kw = 0; kw < kernelSize.width;kw++)
							{
								const size_t prevDiffIdx = prevDiffSize.getIndex(0, kc, inStartY + kh, inStartX + kw);
								prevDiff[prevDiffIdx] += kernel[kernelSize.getIndex(nc,kc, kh, kw)] * nextDiff[nextDiffIdx];
							}
						}
					}
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//update this layer's param
	const ParamSize kernelDiffSize(kernelSize);
	std::shared_ptr<ParamBucket> kernelDiffBucket(std::make_shared<ParamBucket>(kernelDiffSize));
	kernelDiffBucket->fillData(0.0f);
	float* kernelDiff = kernelDiffBucket->getData().get();
	//update kernel
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
					const size_t nextDataIdx = nextDataSize.getIndex(pn, nc, nh, nw);
					const size_t nextDiffIdx = nextDataSize.getIndex(nc, nh, nw);
					for (size_t kc = 0; kc < kernelSize.channels; kc++)
					{
						for (size_t kh = 0; kh < kernelSize.height; kh++)
						{
							for (size_t kw = 0; kw < kernelSize.width; kw++)
							{
								const size_t kernelDiffIdx = kernelDiffSize.getIndex(nc, kc, kh, kw);
								const size_t prevDataIdx = prevDataSize.getIndex(pn, kc, inStartY + kh, inStartX + kw);
								kernelDiff[kernelDiffIdx] += prevData[prevDataIdx] * nextDiff[nextDiffIdx];
							}
						}
					}
				}
			}
		}
	}
	//apply change
	for (size_t kernelIdx = 0; kernelIdx < kernelSize._4DSize();kernelIdx++)
	{
		kernel[kernelIdx] -= getLearningRate()*kernelDiff[kernelIdx];
	}

	//////////////////////////////////////////////////////////////////////////
	//update bias
	const ParamSize biasDiffSize(biasSize);
	std::shared_ptr<ParamBucket> biasDiffBucket(std::make_shared<ParamBucket>(biasDiffSize));
	biasDiffBucket->fillData(0.0f);
	float* biasDiff = biasDiffBucket->getData().get();
	for (size_t pn = 0; pn < prevDataSize.number; pn++)
	{
		for (size_t nc = 0; nc < nextDiffSize.channels; nc++)
		{
			const size_t biasDiffIdx = nc;
			for (size_t nh = 0; nh < nextDiffSize.height; nh++)
			{
				for (size_t nw = 0; nw < nextDiffSize.width; nw++)
				{
					const size_t nextDiffIdx = nextDiffSize.getIndex(0,nc,nh,nw);
					biasDiff[biasDiffIdx] += 1.0f*nextDiff[nextDiffIdx];
				}
			}
		}
	}
	//apply change
	for (size_t biasIdx = 0; biasIdx < biasSize._4DSize(); biasIdx++)
	{
		bias[biasIdx] -= getLearningRate()*bias[biasIdx];
	}

	//////////////////////////////////////////////////////////////////////////
	nextDiffBucket = prevDiffBucket;
}