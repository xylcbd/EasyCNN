#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class ConvolutionLayer : public Layer
	{
	public:
		ConvolutionLayer();
		virtual ~ConvolutionLayer();
	public:
		void setParamaters(const ParamSize _kernelSize, const size_t _widthStep, const size_t _heightStep, const bool _enabledBias);
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void solveInnerParams();
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket);
	private:
		ParamSize kernelSize;
		size_t widthStep = 0;
		size_t heightStep = 0;
		std::shared_ptr<ParamBucket> kernelData;
		bool enabledBias = false;
		std::shared_ptr<ParamBucket> biasData;
	};
}