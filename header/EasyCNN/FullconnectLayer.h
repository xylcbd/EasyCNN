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
		virtual std::string serializeToString() const override;
		virtual void serializeFromString(const std::string content) override;
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const override;
		virtual void solveInnerParams() override;
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket) override;
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<DataBucket>& nextDiffBucket) override;		
	private:
		ParamSize outMapSize;
		std::shared_ptr<ParamBucket> weightsData;
		std::shared_ptr<ParamBucket> weightsDiffData;
		bool enabledBias = false;
		std::shared_ptr<ParamBucket> biasData;		
		std::shared_ptr<ParamBucket> biasDiffData;
	};
}