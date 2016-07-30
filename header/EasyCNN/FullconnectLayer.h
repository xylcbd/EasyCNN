#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class FullconnectLayer : public Layer
	{
	public:
		FullconnectLayer();
		virtual ~FullconnectLayer();
	public:
		void setParamaters(const bool _enabledBias);
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void solveInnerParams();
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket);
	private:
		std::shared_ptr<ParamBucket> weightsData;
		bool enabledBias = false;
		std::shared_ptr<ParamBucket> biasData;
	};
}