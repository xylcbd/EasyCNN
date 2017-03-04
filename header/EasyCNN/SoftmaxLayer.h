#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class SoftmaxLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		SoftmaxLayer();
		virtual ~SoftmaxLayer();		
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const override;
		virtual void forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next) override;
		virtual void backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
			std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff) override;
	};
}