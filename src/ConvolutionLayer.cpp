#include <sstream>
#include "EasyCNN/ConvolutionLayer.h"
#include "EasyCNN/CommonTools.h"
#include "EasyCNN/MathFunctions.h"
#include "EasyCNN/ThreadPool.h"

#if WITH_OPENCV_DEBUG
#include "opencv2/opencv.hpp"
#endif

namespace EasyCNN
{
	ConvolutionLayer::ConvolutionLayer()
	{

	}
	ConvolutionLayer::~ConvolutionLayer()
	{

	}
	void ConvolutionLayer::setParamaters(const ParamSize _kernelSize, const size_t _widthStep, const size_t _heightStep, const bool _enabledBias, const PaddingType _padddingType)
	{
		easyAssert(_kernelSize.number > 0 && _kernelSize.channels > 0 &&
			_kernelSize.width > 0 && _kernelSize.height > 0 && _widthStep > 0 && _heightStep > 0,
			"kernel size or step is invalidate.");
		kernelSize = _kernelSize;
		widthStep = _widthStep;
		heightStep = _heightStep;
		enabledBias = _enabledBias;
		padddingType = _padddingType;
	}
	std::string ConvolutionLayer::serializeToString() const
	{
		const std::string spliter = " ";
		std::stringstream ss;
		//layer desc
		ss << getLayerType() << spliter
			<< kernelSize.number << spliter << kernelSize.channels << spliter << kernelSize.width << spliter << kernelSize.height << spliter
			<< widthStep << spliter << heightStep << spliter << enabledBias << spliter << padddingType << spliter;
		//weight
		const auto kernelData = kernel->getData().get();
		for (size_t i = 0; i < kernelSize.totalSize(); i++)
		{
			ss << kernelData[i] << spliter;
		}
		//bias
		if (enabledBias)
		{
			const auto biasData = bias->getData().get();
			const auto biasSize = bias->getSize();
			for (size_t i = 0; i < biasSize.totalSize(); i++)
			{
				ss << biasData[i] << spliter;
			}
		}
		return ss.str();
	}
	void ConvolutionLayer::serializeFromString(const std::string content)
	{
		std::stringstream ss(content);
		//layer desc
		std::string _layerType;
		int _padddingType = 0;
		ss >> _layerType
			>> kernelSize.number >> kernelSize.channels >> kernelSize.width >> kernelSize.height
			>> widthStep >> heightStep >> enabledBias >> _padddingType;
		padddingType = (PaddingType)_padddingType;
		easyAssert(_layerType == layerType, "layer type is invalidate.");
		solveInnerParams();
		//weight
		auto kernelData = kernel->getData().get();
		for (size_t i = 0; i < kernelSize.totalSize(); i++)
		{
			ss >> kernelData[i];
		}
		//bias
		if (enabledBias)
		{
			const auto biasData = bias->getData().get();
			const auto biasSize = bias->getSize();
			for (size_t i = 0; i < biasSize.totalSize(); i++)
			{
				ss >> biasData[i];
			}
		}
	}
	DEFINE_LAYER_TYPE(ConvolutionLayer, "ConvolutionLayer");
	std::string ConvolutionLayer::getLayerType() const
	{
		return layerType;
	}
	void ConvolutionLayer::solveInnerParams()
	{
		const DataSize inputSize = getInputBucketSize();
		kernelSize.channels = inputSize.channels;
		easyAssert(inputSize.number > 0 && inputSize.channels > 0 && inputSize.width > 0 && inputSize.height > 0, "input size is invalidate.");
		easyAssert(kernelSize.number > 0 && kernelSize.channels > 0 && kernelSize.width > 0 && kernelSize.height > 0 && widthStep > 0 && heightStep > 0,
			"kernel size or step is invalidate.");
		DataSize outputSize;
		outputSize.number = inputSize.number;
		outputSize.channels = kernelSize.number;
		if (padddingType == VALID)
		{
			outputSize.width = (inputSize.width - kernelSize.width) / widthStep + 1;
			outputSize.height = (inputSize.height - kernelSize.height) / heightStep + 1;
		}
		else if (padddingType == SAME)
		{
			outputSize.width = inputSize.width;
			outputSize.height = inputSize.height;
		}
		setOutpuBuckerSize(outputSize);
		easyAssert(outputSize.number > 0 && outputSize.channels > 0 && outputSize.width > 0 && outputSize.height > 0, "output size is invalidate.");
		if (kernel.get() == nullptr)
		{
			kernel.reset(new ParamBucket(kernelSize));
			normal_distribution_init(kernel->getData().get(), kernel->getSize().totalSize(), 0.0f, 0.1f);
			/*
			const size_t fan_in = inputSize._2DSize();
			const size_t fan_out = outputSize._2DSize();
			xavier_init(kernel->getData().get(), kernel->getSize().totalSize(), fan_in, fan_out);
			*/
		}
		if (kernelGradient.get() == nullptr)
		{
			kernelGradient.reset(new ParamBucket(kernel->getSize()));
			const_distribution_init(kernelGradient->getData().get(), kernelGradient->getSize().totalSize(), 0.0f);
		}
		if (enabledBias)
		{
			if (bias.get() == nullptr)
			{
				bias.reset(new ParamBucket(ParamSize(kernelSize.number, 1, 1, 1)));
				const_distribution_init(bias->getData().get(), bias->getSize().totalSize(), 0.0f);
			}
			if (biasGradient.get() == nullptr)
			{
				biasGradient.reset(new ParamBucket(bias->getSize()));
				const_distribution_init(biasGradient->getData().get(), biasGradient->getSize().totalSize(), 0.0f);
			}
		}
		//parmas
		params.clear();
		params.push_back(kernel);
		params.push_back(bias);
		//diffs
		gradients.clear();
		gradients.push_back(kernelGradient);
		gradients.push_back(biasGradient);
	}
	void ConvolutionLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();

		const float* prevData = prev->getData().get();
		const float* kernelData = kernel->getData().get();
		const float* biasData = bias->getData().get();
		float* nextData = next->getData().get();

		auto worker = [&](const size_t start, const size_t stop){
			convolution2d(prevData + start*prevSize._3DSize(), kernelData, biasData, nextData + start*nextSize._3DSize(),
				stop-start, prevSize.channels, prevSize.width, prevSize.height, 
				kernelSize.number,kernelSize.width, kernelSize.height, widthStep, heightStep,
				nextSize.width, nextSize.height,(int)padddingType);
		};
		dispatch_worker(worker, prevSize.number);

#if WITH_OPENCV_DEBUG
		//input image
		for (size_t pn = 0; pn < prevSize.number; pn++)
		{
			for (size_t pc = 0; pc < prevSize.channels; pc++)
			{
				const float* imageData = prevData + pn*prevSize._3DSize() + pc*prevSize._2DSize();
				const size_t imageWidth = prevSize.width;
				const size_t imageStride = imageWidth;
				const size_t imageHeight = prevSize.height;
				const size_t imageChannel = 1;
				cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
				std::stringstream ss;
				ss << "convolution input, number: " << pn << ",channel: " << pc;
				const std::string title = ss.str();
// 				cv::imshow(title, image);
			}		
		}
		//kernel image
		for (size_t kn = 0; kn < kernelSize.number; kn++)
		{
			for (size_t kc = 0; kc < kernelSize.channels; kc++)
			{
				const float* imageData = kernelData + kn*kernelSize._3DSize() + kc*kernelSize._2DSize();
				const size_t imageWidth = kernelSize.width;
				const size_t imageStride = imageWidth;
				const size_t imageHeight = kernelSize.height;
				const size_t imageChannel = 1;
				cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
 				std::stringstream ss;
				ss << "convolution kernel, number: " << kn << ",channel: " << kc;
				const std::string title = ss.str();
// 				cv::imshow(title, image);
			}
		}
		//output image
		for (size_t nn = 0; nn < nextSize.number; nn++)
		{
			for (size_t nc = 0; nc < nextSize.channels; nc++)
			{
				const float* imageData = nextData + nn*nextSize._3DSize() + nc*nextSize._2DSize();
				const size_t imageWidth = nextSize.width;
				const size_t imageStride = imageWidth;
				const size_t imageHeight = nextSize.height;
				const size_t imageChannel = 1;
				cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
				std::stringstream ss;
				ss << "convolution output, number: " << nn << ",channel: " << nc;
				const std::string title = ss.str();
// 				cv::imshow(title, image);
			}
		}
		cv::waitKey(0);
		cv::destroyAllWindows();
#endif //WITH_OPENCV_DEBUG
	}
	void ConvolutionLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const ParamSize biasSize = bias->getSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		float *kernelData = kernel->getData().get();
		float *biasData = bias->getData().get();
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//////////////////////////////////////////////////////////////////////////
		//update prevDiff
		prevDiff->fillData(0.0f);
		//calculate current inner diff
		auto worker = [&](const size_t start, const size_t stop){
			for (size_t nn = start; nn < stop; nn++)
			{
				for (size_t nc = 0; nc < nextSize.channels; nc++)
				{
					for (size_t nh = 0; nh < nextSize.height; nh++)
					{
						for (size_t nw = 0; nw < nextSize.width; nw++)
						{
							const size_t inStartX = nw*widthStep;
							const size_t inStartY = nh*heightStep;
							const size_t nextDiffIdx = nextSize.getIndex(nn, nc, nh, nw);
							const size_t kn = nc;
							for (size_t kc = 0; kc < kernelSize.channels; kc++)
							{
								for (size_t kh = 0; kh < kernelSize.height; kh++)
								{
									for (size_t kw = 0; kw < kernelSize.width; kw++)
									{
										const size_t inY = inStartY + kh;
										const size_t inX = inStartX + kw;
										if (inY >= 0 && inY < inputSize.height && inX >= 0 && inX < inputSize.width)
										{
											const size_t prevDiffIdx = prevDiffSize.getIndex(nn, kc, inY, inX);
											const size_t kernelIdx = kernelSize.getIndex(kn, kc, kh, kw);
											prevDiffData[prevDiffIdx] += kernelData[kernelIdx] * nextDiffData[nextDiffIdx];
										}
									}
								}
							}
						}
					}
				}
			}
		};
		dispatch_worker(worker, prevSize.number);

		//////////////////////////////////////////////////////////////////////////
		//update this layer's param
		const ParamSize kernelGradientSize(kernelSize);
		kernelGradient->fillData(0.0f);
		float* kernelGradientData = kernelGradient->getData().get();
		//update kernel gradient
		for (size_t nn = 0; nn < nextSize.number; nn++)
		{
			for (size_t nc = 0; nc < nextSize.channels; nc++)
			{
				for (size_t nh = 0; nh < nextSize.height; nh++)
				{
					for (size_t nw = 0; nw < nextSize.width; nw++)
					{
						const size_t inStartX = nw*widthStep;
						const size_t inStartY = nh*heightStep;
						const size_t nextDiffIdx = nextSize.getIndex(nn, nc, nh, nw);
						const size_t kn = nc;
						for (size_t kc = 0; kc < kernelSize.channels; kc++)
						{
							for (size_t kh = 0; kh < kernelSize.height; kh++)
							{
								for (size_t kw = 0; kw < kernelSize.width; kw++)
								{
									const size_t inY = inStartY + kh;
									const size_t inX = inStartX + kw;
									if (inY >= 0 && inY < inputSize.height && inX >= 0 && inX < inputSize.width)
									{
										const size_t kernelGradientIdx = kernelGradientSize.getIndex(kn, kc, kh, kw);
										const size_t prevIdx = prevSize.getIndex(nn, kc, inY, inX);
										kernelGradientData[kernelGradientIdx] += prevData[prevIdx] * nextDiffData[nextDiffIdx];
									}
								}
							}
						}
					}
				}
			}
		}
		//div by batch size
		div_inplace(kernelGradientData, (float)nextSize.number, kernelSize.totalSize());		

		//////////////////////////////////////////////////////////////////////////
		//update bias gradient
		biasGradient->fillData(0.0f);
		float* biasGradientData = biasGradient->getData().get();
		for (size_t nn = 0; nn < nextDiffSize.number; nn++)
		{
			for (size_t nc = 0; nc < nextDiffSize.channels; nc++)
			{
				const size_t biasGradientIdx = nc;
				for (size_t nh = 0; nh < nextDiffSize.height; nh++)
				{
					for (size_t nw = 0; nw < nextDiffSize.width; nw++)
					{
						const size_t nextDiffIdx = nextDiffSize.getIndex(nn, nc, nh, nw);
						biasGradientData[biasGradientIdx] += 1.0f*nextDiffData[nextDiffIdx];
					}
				}
			}
		}
		//div by batch size
		div_inplace(biasGradientData, (float)nextSize.number, biasSize.totalSize());
	}
}//namespace