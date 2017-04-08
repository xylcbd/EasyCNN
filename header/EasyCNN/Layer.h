#pragma once
#include <memory>
#include <string>
#include <vector>
#include "EasyCNN/Configure.h"
#include "EasyCNN/DataBucket.h"
#include "EasyCNN/ParamBucket.h"

#define DECLARE_LAYER_TYPE static const std::string layerType;
#define DEFINE_LAYER_TYPE(class_type,type_string) const std::string class_type::layerType = type_string; 
#define FRIEND_WITH_NETWORK friend class NetWork;

namespace EasyCNN
{
	enum class Phase
	{
		Train,
		Test
	};
	class Layer
	{
		FRIEND_WITH_NETWORK
	public:
		inline DataSize getInputBucketSize() const{ return inputSize; }
		inline DataSize getOutputBucketSize() const{ return outputSize; }
	protected:
		virtual std::string getLayerType() const = 0;
		virtual std::string serializeToString() const{ return getLayerType(); };
		virtual void serializeFromString(const std::string content){/*nop*/};
		//phase
		inline void setPhase(Phase phase) { this->phase = phase; }
		inline Phase getPhase() const{ return phase; }
		//learning rate
		inline void setLearningRate(const float learningRate){ this->learningRate = learningRate; }
		inline float getLearningRate() const{ return learningRate; }	
		//diff
		inline std::vector<std::shared_ptr<ParamBucket>> getDiffData() const { return gradients; }
		//params
		inline std::vector<std::shared_ptr<ParamBucket>> getParamData() const { return params; }
		//size
		inline void setInputBucketSize(const DataSize size){ inputSize = size; }		
		inline void setOutpuBuckerSize(const DataSize size){ outputSize = size; }		
		//solve params
		virtual void solveInnerParams(){ outputSize = inputSize; }
		//data flow		
		virtual void forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next) = 0;
		virtual void backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next, 
			std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff) = 0;
	protected:
		//subclass must add all gradient to gradients
		std::vector<std::shared_ptr<DataBucket>> gradients;
		//subclass must add all weight to params
		std::vector<std::shared_ptr<ParamBucket>> params;
		Phase phase = Phase::Train;
		DataSize inputSize;
		DataSize outputSize;
		float learningRate = 0.1f;
	};
}