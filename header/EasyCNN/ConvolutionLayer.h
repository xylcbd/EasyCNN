#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class ConvolutionLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		ConvolutionLayer();
		virtual ~ConvolutionLayer();	
		void setParamaters(const ParamSize _kernelSize, const size_t _widthStep, const size_t _heightStep, const bool _enabledBias);
	protected:
		virtual std::string serializeToString() const override;
		virtual void serializeFromString(const std::string content) override;
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const override;
		virtual void solveInnerParams() override;
		virtual void forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next) override;
		virtual void backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
			std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff) override;
	private:
		ParamSize kernelSize;
		size_t widthStep = 0;
		size_t heightStep = 0;
		std::shared_ptr<ParamBucket> kernel;
		std::shared_ptr<ParamBucket> kernelGradient;
		bool enabledBias = false;
		std::shared_ptr<ParamBucket> bias;
		std::shared_ptr<ParamBucket> biasGradient;
	};
}