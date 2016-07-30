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
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket,std::shared_ptr<ParamBucket>& nextDiffBucket);
	};

	class TanhLayer : public ActivationLayer
	{
	public:
		TanhLayer();
		virtual ~TanhLayer();
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket);
	};

	class ReluLayer : public ActivationLayer
	{
	public:
		ReluLayer();
		virtual ~ReluLayer();
	public:
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const;
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket);
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket);
	};
}