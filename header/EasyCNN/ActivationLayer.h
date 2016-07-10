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
	public:
		SigmodLayer();
		virtual ~SigmodLayer();
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
	};

	class TanhLayer : public ActivationLayer
	{
	public:
		TanhLayer();
		virtual ~TanhLayer();
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
	};

	class ReluLayer : public ActivationLayer
	{
	public:
		ReluLayer();
		virtual ~ReluLayer();
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
	};
}