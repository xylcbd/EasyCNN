#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class PoolingLayer : public Layer
	{
		FRIEND_WITH_NETWORK
	public:
		enum PoolingType
		{
			MaxPooling=0,
			MeanPooling=1
		};
	public:
		PoolingLayer();
		virtual ~PoolingLayer();	
		void setParamaters(const PoolingType _poolingType, const ParamSize _poolingKernelSize, const size_t _widthStep, const size_t _heightStep);
	protected:
		virtual std::string serializeToString() const override;
		virtual void serializeFromString(const std::string content) override;
		DECLARE_LAYER_TYPE;
		virtual std::string getLayerType() const override;
		virtual void solveInnerParams() override;
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket) override;
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket, std::shared_ptr<ParamBucket>& nextDiffBucket) override;
	private:
		PoolingType poolingType = PoolingType::MaxPooling;
		//FIXME : using int type
		std::shared_ptr<ParamBucket> maxIdxesBucket;
		ParamSize poolingKernelSize;
		size_t widthStep = 0;
		size_t heightStep = 0;
	};
}