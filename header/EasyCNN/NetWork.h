#pragma once
#include <memory>
#include <vector>
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"
#include "EasyCNN/LossFunction.h"
#include "EasyCNN/Optimizer.h"

namespace EasyCNN
{
	class NetWork
	{
	public:
		NetWork();
		virtual ~NetWork();
	public:
		//common
		//loss of batch
		float getLoss(const std::shared_ptr<DataBucket> labelDataBucket, const std::shared_ptr<DataBucket> outputDataBucket);
		//test only!
		bool loadModel(const std::string& modelFile);
		std::shared_ptr<DataBucket> testBatch(const std::shared_ptr<DataBucket> inputDataBucket);
		//train only!
		void setInputSize(const DataSize size);
		void setLossFunctor(std::shared_ptr<LossFunctor> lossFunctor);
		void setOptimizer(std::shared_ptr<Optimizer> optimizer);
		void setLearningRate(const float lr);
		void addayer(std::shared_ptr<Layer> layer);
		float trainBatch(const std::shared_ptr<DataBucket> inputDataBucket,
			const std::shared_ptr<DataBucket> labelDataBucket);
		bool saveModel(const std::string& modelFile);
	private:
		//common
		void setPhase(Phase phase);
		Phase getPhase() const;
		std::string encrypt(const std::string& content);
		std::string decrypt(const std::string& content);
	private:
		//common
		std::shared_ptr<DataBucket> forward(const std::shared_ptr<DataBucket> inputDataBucket);
		float backward(const std::shared_ptr<DataBucket> labelDataBucket);		
		std::shared_ptr<Layer> createLayerByType(const std::string layerType);
		std::string lookaheadLayerType(const std::string line);
	private:
		Phase phase = Phase::Train;
		std::vector<std::shared_ptr<Layer>> layers;
		std::vector<std::shared_ptr<DataBucket>> dataBuckets;
		std::vector<std::shared_ptr<DataBucket>> diffBuckets;
		std::shared_ptr<LossFunctor> lossFunctor;
		std::shared_ptr<Optimizer> optimizer;
	};
}