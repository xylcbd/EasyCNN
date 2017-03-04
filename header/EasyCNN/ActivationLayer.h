#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class ActivationLayer : public Layer
	{
	};

	class SigmodLayer : public ActivationLayer
	{
		FRIEND_WITH_NETWORK
	public:
		SigmodLayer();
		virtual ~SigmodLayer();
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const override;
		virtual void forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next) override;
		virtual void backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
			std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff) override;
	};

	class TanhLayer : public ActivationLayer
	{
		FRIEND_WITH_NETWORK
	public:
		TanhLayer();
		virtual ~TanhLayer();
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const override;
		virtual void forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next) override;
		virtual void backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
			std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff) override;
	};

	class ReluLayer : public ActivationLayer
	{
		FRIEND_WITH_NETWORK
	public:
		ReluLayer();
		virtual ~ReluLayer();
	protected:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const override;
		virtual void forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next) override;
		virtual void backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
			std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff) override;
	};
}