#pragma once

#include "EasyCNN/Configure.h"
#include "EasyCNN/DataBucket.h"
#include "EasyCNN/ParamBucket.h"

namespace EasyCNN
{
	class LossFunctor
	{
	public:
		virtual float getLoss(const std::shared_ptr<DataBucket> labelDataBucket,
			const std::shared_ptr<DataBucket> outputDataBucket) = 0;
		virtual void getDiff(const std::shared_ptr<DataBucket> labelDataBucket,
			const std::shared_ptr<DataBucket> outputDataBucket, std::shared_ptr<DataBucket>& diff) = 0;
	};

	class CrossEntropyFunctor : public LossFunctor
	{
	public:
		virtual float getLoss(const std::shared_ptr<DataBucket> labelDataBucket,
			const std::shared_ptr<DataBucket> outputDataBucket);
		virtual void getDiff(const std::shared_ptr<DataBucket> labelDataBucket,
			const std::shared_ptr<DataBucket> outputDataBucket, std::shared_ptr<DataBucket>& diff);
	};

	class MSEFunctor : public LossFunctor
	{
	public:
		virtual float getLoss(const std::shared_ptr<DataBucket> labelDataBucket,
			const std::shared_ptr<DataBucket> outputDataBucket);
		virtual void getDiff(const std::shared_ptr<DataBucket> labelDataBucket,
			const std::shared_ptr<DataBucket> outputDataBucket, std::shared_ptr<DataBucket>& diff);
	};
}