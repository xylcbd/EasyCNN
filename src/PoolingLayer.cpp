#include <algorithm>
#include <sstream>
#include "EasyCNN/PoolingLayer.h"

#if WITH_OPENCV_DEBUG
#include "opencv2/opencv.hpp"
#endif

namespace EasyCNN
{
	PoolingLayer::PoolingLayer()
	{

	}
	PoolingLayer::~PoolingLayer()
	{

	}
	void PoolingLayer::setParamaters(const PoolingType _poolingType, const ParamSize _poolingKernelSize, const size_t _widthStep, const size_t _heightStep, const PaddingType _paddingType)
	{
		easyAssert(_poolingKernelSize.number == 1 && _poolingKernelSize.channels > 0 && _poolingKernelSize.width > 1 && _poolingKernelSize.height > 1 && _widthStep > 0 && _heightStep > 0,
			"parameters invalidate.");
		poolingKernelSize = _poolingKernelSize;
		poolingType = _poolingType;
		widthStep = _widthStep;
		heightStep = _heightStep;
		paddingType = _paddingType;
	}
	std::string PoolingLayer::serializeToString() const
	{
		const std::string spliter = " ";
		std::stringstream ss;
		//layer desc
		ss << getLayerType() << spliter
			<< poolingType << spliter
			<< poolingKernelSize.number << spliter
			<< poolingKernelSize.channels << spliter
			<< poolingKernelSize.width << spliter
			<< poolingKernelSize.height << spliter
			<< widthStep << spliter
			<< heightStep << spliter;

		return ss.str();
	}
	void PoolingLayer::serializeFromString(const std::string content)
	{
		std::stringstream ss(content);
		//layer desc
		std::string _layerType;
		int _poolingType = 0;
		int _paddingType = 0;
		ss >> _layerType
			>> _poolingType
			>> poolingKernelSize.number
			>> poolingKernelSize.channels
			>> poolingKernelSize.width
			>> poolingKernelSize.height
			>> widthStep
			>> heightStep
			>> _paddingType;
		poolingType = (PoolingType)_poolingType;
		paddingType = (PaddingType)_paddingType;
		easyAssert(_layerType == getLayerType(), "layer type is invalidate.");
		easyAssert((poolingType == MaxPooling || poolingType == MeanPooling), "pooling type is invalidate.");
		solveInnerParams();
	}
	DEFINE_LAYER_TYPE(PoolingLayer, "PoolingLayer");
	std::string PoolingLayer::getLayerType() const
	{
		return layerType;
	}
	void PoolingLayer::solveInnerParams()
	{
		easyAssert(poolingKernelSize.number > 0 && poolingKernelSize.channels > 0 && poolingKernelSize.width > 1 && poolingKernelSize.height > 1, "poolingKernelSize parameters invalidate.");
		const DataSize inputSize = getInputBucketSize();
		poolingKernelSize.number = 1;
		poolingKernelSize.channels = inputSize.channels;
		easyAssert(inputSize.number && poolingKernelSize.number && inputSize.channels == poolingKernelSize.channels &&
			inputSize.width >= poolingKernelSize.width && inputSize.height >= poolingKernelSize.height,
			"poolingKernelSize parameters invalidate.");
		DataSize outputSize;
		outputSize.number = inputSize.number;
		outputSize.channels = inputSize.channels;
		if (paddingType == VALID)
		{
			outputSize.width = (inputSize.width - poolingKernelSize.width) / widthStep + 1;
			outputSize.height = (inputSize.height - poolingKernelSize.height) / heightStep + 1;
		}
		else if (paddingType == SAME)
		{
			outputSize.width = (size_t)std::ceil((float)inputSize.width / (float)widthStep);
			outputSize.height = (size_t)std::ceil((float)inputSize.height / (float)heightStep);
		}
		setOutpuBuckerSize(outputSize);

		if (getPhase() == Phase::Train && poolingType == PoolingType::MaxPooling)
		{
			maxIdxes.reset(new ParamBucket(ParamSize(outputSize.number, outputSize.channels, outputSize.height, outputSize.width)));
		}
	}
	void PoolingLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevDataSize = prev->getSize();
		const DataSize nextDataSize = next->getSize();

		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();
		float* maxIdxesData = nullptr;
		if (getPhase() == Phase::Train && poolingType == PoolingType::MaxPooling)
		{
			auto newSize = maxIdxes->getSize();
			if (newSize.number != prevDataSize.number)
			{
				newSize.number = prevDataSize.number;
				maxIdxes.reset(new ParamBucket(newSize));
			}
			maxIdxesData = maxIdxes->getData().get();
		}
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
						const size_t nextDataIdx = nextDataSize.getIndex(nn, nc, nh, nw);
						float result = 0;
						size_t maxIdx = 0;
						if (poolingType == PoolingType::MaxPooling)
						{
							for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
							{
								for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
								{
									const size_t inY = inStartY + ph;
									const size_t inX = inStartX + pw;
									if (inY >= 0 && inY<inputSize.height && inX >= 0 && inX<inputSize.width)
									{
										const size_t prevDataIdx = prevDataSize.getIndex(nn, nc, inY, inX);
										if (result < prevData[prevDataIdx])
										{
											result = prevData[prevDataIdx];
											maxIdx = ph*poolingKernelSize.width + pw;
										}
									}									
								}
							}
							if (maxIdxes)
							{
								maxIdxesData[nextDataIdx] = (float)maxIdx;
							}
						}
						else if (poolingType == PoolingType::MeanPooling)
						{
							for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
							{
								for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
								{
									const size_t inY = inStartY + ph;
									const size_t inX = inStartX + pw;
									if (inY >= 0 && inY < inputSize.height && inX >= 0 && inX < inputSize.width)
									{
										const size_t prevDataIdx = prevDataSize.getIndex(nc, inY, inX);
										result += prevData[prevDataIdx];
									}
								}
							}
							result /= poolingKernelSize.width*poolingKernelSize.height;
						}
						nextData[nextDataIdx] = result;
					}//ow
				}//oh
			}//oc
		}//on

#if WITH_OPENCV_DEBUG
		//input image
		for (size_t pn = 0; pn < prevDataSize.number; pn++)
		{
			for (size_t pc = 0; pc < prevDataSize.channels; pc++)
			{
				const float* imageData = prevData + pn*prevDataSize._3DSize() + pc*prevDataSize._2DSize();
				const size_t imageWidth = prevDataSize.width;
				const size_t imageStride = imageWidth;
				const size_t imageHeight = prevDataSize.height;
				const size_t imageChannel = 1;
				cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
				std::stringstream ss;
				ss << "pooling input, number: " << pn << ",channel: " << pc;
				const std::string title = ss.str();
// 				cv::imshow(title, image);
			}
		}
		//output image
		for (size_t nn = 0; nn < nextDataSize.number; nn++)
		{
			for (size_t nc = 0; nc < nextDataSize.channels; nc++)
			{
				const float* imageData = nextData + nn*nextDataSize._3DSize() + nc*nextDataSize._2DSize();
				const size_t imageWidth = nextDataSize.width;
				const size_t imageStride = imageWidth;
				const size_t imageHeight = nextDataSize.height;
				const size_t imageChannel = 1;
				cv::Mat image((int)imageHeight, (int)imageWidth, CV_32FC1, (void*)imageData, imageStride*sizeof(imageData[0]));
				std::stringstream ss;
				ss << "pooling output, number: " << nn << ",channel: " << nc;
				const std::string title = ss.str();
// 				cv::imshow(title, image);
			}
		}
		cv::waitKey(0);
		cv::destroyAllWindows();
#endif //WITH_OPENCV_DEBUG
	}
	void PoolingLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//update prevDiff
		const float* maxIdxesData = nullptr;
		if (poolingType == PoolingType::MaxPooling)
		{
			easyAssert(maxIdxes->getSize()._3DSize() == nextSize._3DSize(), "idx size must equals with next data.");
			maxIdxesData = maxIdxes->getData().get();
		}
		prevDiff->fillData(0.0f);
		//calculate current inner diff 
		//none
		//pass next layer's diff to previous layer
		for (size_t nn = 0; nn < nextSize.number; nn++)
		{
			const float* nextDiffData = nextDiff->getData().get() + nn*nextDiffSize._3DSize();
			float* prevDiffData = prevDiff->getData().get() + nn*prevDiffSize._3DSize();

			for (size_t nc = 0; nc < nextSize.channels; nc++)
			{
				for (size_t nh = 0; nh < nextSize.height; nh++)
				{
					for (size_t nw = 0; nw < nextSize.width; nw++)
					{
						const size_t inStartX = nw*widthStep;
						const size_t inStartY = nh*heightStep;
						const size_t nextDataIdx = nextSize.getIndex(nc, nh, nw);
						if (poolingType == PoolingType::MaxPooling)
						{
							for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
							{
								for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
								{
									const size_t inY = inStartY + ph;
									const size_t inX = inStartX + pw;
									if (inY >= 0 && inY < inputSize.height && inX >= 0 && inX < inputSize.width)
									{
										const size_t prevDiffIdx = prevSize.getIndex(nc, inY, inX);
										if (ph*poolingKernelSize.width + pw == maxIdxesData[nextDataIdx])
										{
											prevDiffData[prevDiffIdx] += nextDiffData[nextDataIdx];
										}
									}
								}
							}
						}
						else if (poolingType == PoolingType::MeanPooling)
						{
							const float meanDiff = nextDiffData[nextDataIdx] / (float)(poolingKernelSize._2DSize());
							for (size_t ph = 0; ph < poolingKernelSize.height; ph++)
							{
								for (size_t pw = 0; pw < poolingKernelSize.width; pw++)
								{
									const size_t inY = inStartY + ph;
									const size_t inX = inStartX + pw;
									if (inY >= 0 && inY < inputSize.height && inX >= 0 && inX < inputSize.width)
									{
										const size_t prevDiffIdx = prevSize.getIndex(nc, inY, inX);
										prevDiffData[prevDiffIdx] += meanDiff;
									}
								}
							}
						}
					}
				}
			}
		}

		//update this layer's param
		//nop
	}
}//namespace