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
		enum PaddingType
		{
			VALID = 0,
			SAME = 1
		};
	public:
		PoolingLayer();
		virtual ~PoolingLayer();	
		void setParamaters(const PoolingType _poolingType, const ParamSize _poolingKernelSize, const size_t _widthStep, const size_t _heightStep,
			const PaddingType _paddingType);
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
		PoolingType poolingType = PoolingType::MaxPooling;
		std::shared_ptr<ParamBucket> maxIdxes;
		ParamSize poolingKernelSize;
		size_t widthStep = 0;
		size_t heightStep = 0;
		PaddingType paddingType = VALID;
	};
}