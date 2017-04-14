#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class DropoutLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		DropoutLayer();
		DropoutLayer(const float _rate);
		virtual ~DropoutLayer();
	public:
		void setParamaters(const float _rate);
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
		float rate = 0.5f;
		std::shared_ptr<ParamBucket> mask;
	};
}