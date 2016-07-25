#pragma once
#include <memory>
#include <string>
#include "EasyCNN/Configure.h"
#include "EasyCNN/DataBucket.h"
#include "EasyCNN/ParamBucket.h"

#define DECLARE_LAYER_TYPE static const std::string layerType;
#define DEFINE_LAYER_TYPE(class_type,type_string) const std::string class_type::layerType = type_string; 

namespace EasyCNN
{
	class Layer
	{
	public:
		virtual std::string getLayerType() const = 0;
		//size
		void setInputBucketSize(const DataSize size);
		DataSize getInputBucketSize() const;		
		void setOutpuBuckerSize(const DataSize size);
		DataSize getOutputBucketSize() const;
		virtual void solveInnerParams();
		//data flow		
		virtual void forward(const std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket) = 0;
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, const std::shared_ptr<DataBucket> nextDataBucket) = 0;
	private:
		DataSize inputSize;
		DataSize outputSize;
	};
}