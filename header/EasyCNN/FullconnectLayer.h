#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class FullconnectLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		FullconnectLayer();
		virtual ~FullconnectLayer();
	public:
		void setParamaters(const ParamSize _outMapSize,const bool _enabledBias);
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string serializeToString() const override;
		virtual void serializeFromString(const std::string content) override;		
		virtual std::string getLayerType() const override;
		virtual void solveInnerParams() override;
		virtual void forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next) override;
		virtual void backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
			std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff) override;
	private:
		ParamSize outMapSize;
		std::shared_ptr<ParamBucket> weight;
		std::shared_ptr<ParamBucket> weightGradient;
		bool enabledBias = false;
		std::shared_ptr<ParamBucket> bias;		
		std::shared_ptr<ParamBucket> biasGradient;
	};
}