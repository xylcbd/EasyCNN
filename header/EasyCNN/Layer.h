#pragma once
#include <memory>
#include <string>
#include "EasyCNN/Configure.h"
#include "EasyCNN/DataBucket.h"

#define DECLARE_LAYER_TYPE static const std::string layerType;
#define DEFINE_LAYER_TYPE(class_type,type_string) const std::string class_type::layerType = type_string; 

namespace EasyCNN
{
	class Layer
	{
	public:
		virtual std::string getLayerType() const = 0;
		virtual void forward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket) = 0;
		virtual void backward(std::shared_ptr<DataBucket> prevDataBucket, std::shared_ptr<DataBucket> nextDataBucket) = 0;		
	};
}