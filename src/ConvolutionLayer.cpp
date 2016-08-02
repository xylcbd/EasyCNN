#include "EasyCNN/ConvolutionLayer.h"
#include "EasyCNN/CommonTools.h"

#if WITH_OPENCV_DEBUG
#include "opencv2/opencv.hpp"
#endif

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
	const DataSize prevDataSize = prevDataBucket->getSize();
	const DataSize nextDataSize = nextDataBucket->getSize();

	const float* prevRawData = prevDataBucket->getData().get();
	const float* kernelRawData = kernelData->getData().get();
	const float* biasRawData = biasData->getData().get();
	float* nextRawData = nextDataBucket->getData().get();
	for (size_t nn = 0; nn < nextDataSize.number; nn++)
	{			
		for (size_t nc = 0; nc < nextDataSize.channels; nc++)
		{		
			for (size_t nh = 0; nh < nextDataSize.height; nh++)
			{
				for (size_t nw = 0; nw < nextDataSize.width; nw++)
				{
					const size_t inStartX = nw*widthStep;
					const size_t inStartY = nh*heightStep;					
					float sum = 0;
					for (size_t kc = 0; kc < kernelSize.channels; kc++)
					{
						for (size_t kh = 0; kh < kernelSize.height; kh++)
						{
							for (size_t kw = 0; kw < kernelSize.width; kw++)
							{
								const size_t prevDataIdx = prevDataSize.getIndex(nn, kc, inStartY + kh, inStartX + kw);
								const size_t kernelIdx = kernelSize.getIndex(nc,kc, kh, kw);
								sum += prevRawData[prevDataIdx] * kernelRawData[kernelIdx];
							}
						}
					}
					if (enabledBias)
					{
						const size_t biasIdx = nc;
						sum += biasRawData[biasIdx];
					}
					const size_t nextDataIdx = nextDataSize.getIndex(nn, nc, nh, nw);
					nextRawData[nextDataIdx] = sum;
				}
			}
		}
	}

#if WITH_OPENCV_DEBUG
	//input image
	for (int pn = 0; pn < prevDataSize.number; pn++)
	{
		for (int pc = 0; pc < prevDataSize.channels; pc++)
		{
			const float* imageData = prevRawData + pn*prevDataSize._3DSize() + pc*prevDataSize._2DSize();
			const size_t imageWidth = prevDataSize.width;
			const size_t imageStride = imageWidth;
			const size_t imageHeight = prevDataSize.height;
			const size_t imageChannel = 1;
			cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
			image.empty();
		}		
	}
	//kernel image
	for (int kn = 0; kn < kernelSize.number; kn++)
	{
		for (int kc = 0; kc < kernelSize.channels; kc++)
		{
			const float* imageData = kernelRawData + kn*kernelSize._3DSize() + kc*kernelSize._2DSize();
			const size_t imageWidth = kernelSize.width;
			const size_t imageStride = imageWidth;
			const size_t imageHeight = kernelSize.height;
			const size_t imageChannel = 1;
			cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
			image.empty();
		}
	}
	//output image
	for (int nn = 0; nn < nextDataSize.number; nn++)
	{
		for (int nc = 0; nc < nextDataSize.channels; nc++)
		{
			const float* imageData = nextRawData + nn*nextDataSize._3DSize() + nc*nextDataSize._2DSize();
			const size_t imageWidth = nextDataSize.width;
			const size_t imageStride = imageWidth;
			const size_t imageHeight = nextDataSize.height;
			const size_t imageChannel = 1;
			cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
			image.empty();
		}
	}
#endif //WITH_OPENCV_DEBUG
}
void EasyCNN::ConvolutionLayer::backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket)
{
	easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
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
					const size_t nextDiffIdx = nextDataSize.getIndex(nc, nh, nw);
					const size_t kn = nc;
					for (size_t kc = 0; kc < kernelSize.channels; kc++)
					{
						for (size_t kh = 0; kh < kernelSize.height; kh++)
						{
							for (size_t kw = 0; kw < kernelSize.width;kw++)
							{
								const size_t prevDiffIdx = prevDiffSize.getIndex(0, kc, inStartY + kh, inStartX + kw);
								const size_t kernelIdx = kernelSize.getIndex(kn, kc, kh, kw);
								prevDiff[prevDiffIdx] += kernel[kernelIdx] * nextDiff[nextDiffIdx] / nextDataSize.number;
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
					const size_t nextDiffIdx = nextDataSize.getIndex(nc, nh, nw);
					const size_t kn = nc;
					for (size_t kc = 0; kc < kernelSize.channels; kc++)
					{
						for (size_t kh = 0; kh < kernelSize.height; kh++)
						{
							for (size_t kw = 0; kw < kernelSize.width; kw++)
							{
								const size_t kernelDiffIdx = kernelDiffSize.getIndex(kn, kc, kh, kw);
								const size_t prevDataIdx = prevDataSize.getIndex(pn, kc, inStartY + kh, inStartX + kw);
								kernelDiff[kernelDiffIdx] += prevData[prevDataIdx] * nextDiff[nextDiffIdx] / nextDataSize.number;
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
					biasDiff[biasDiffIdx] += 1.0f*nextDiff[nextDiffIdx] / nextDataSize.number;
				}
			}
		}
	}
	//apply change
	for (size_t biasIdx = 0; biasIdx < biasSize._4DSize(); biasIdx++)
	{
		bias[biasIdx] -= getLearningRate()*biasDiff[biasIdx];
	}

	//////////////////////////////////////////////////////////////////////////
	nextDiffBucket = prevDiffBucket;
}