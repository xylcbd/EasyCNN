#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class InputLayer : public Layer
	{
	public:
		InputLayer();
		virtual ~InputLayer();
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;	
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket);
	};
}