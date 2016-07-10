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
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
	private:
	};
}