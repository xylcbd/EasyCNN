#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
//configure
#include "EasyCNN/Configure.h"
//layers
#include "EasyCNN/Layer.h"
#include "EasyCNN/ActivationLayer.h"
#include "EasyCNN/InputLayer.h"
#include "EasyCNN/ConvolutionLayer.h"
#include "EasyCNN/PoolingLayer.h"
#include "EasyCNN/FullconnectLayer.h"
#include "EasyCNN/SoftmaxLayer.h"
#include "EasyCNN/DropoutLayer.h"
#include "EasyCNN/BatchNormalizationLayer.h"
//network
#include "EasyCNN/NetWork.h"

namespace EasyCNN
{
	NetWork::NetWork()
	{
		logVerbose("NetWork constructed.");
	}
	NetWork::~NetWork()
	{
		logVerbose("NetWork destructed.");
	}

	//////////////////////////////////////////////////////////////////////////
	//common
	void NetWork::setPhase(Phase phase)
	{
		logVerbose("NetWork setPhase begin.");
		this->phase = phase;
		logVerbose("NetWork setPhase end.");
	}
	Phase NetWork::getPhase() const
	{
		return phase;
	}
	float NetWork::getLoss(const std::shared_ptr<DataBucket> labelDataBucket, const std::shared_ptr<DataBucket> outputDataBucket)
	{
		if (!lossFunctor)
		{
			return 0.0f;
		}
		return lossFunctor->getLoss(labelDataBucket, outputDataBucket);
	}
	std::shared_ptr<Layer> NetWork::createLayerByType(const std::string layerType)
	{
		if (layerType == InputLayer::layerType)
		{
			return std::make_shared<InputLayer>();
		}
		else if (layerType == ConvolutionLayer::layerType)
		{
			return std::make_shared<ConvolutionLayer>();
		}
		else if (layerType == PoolingLayer::layerType)
		{
			return std::make_shared<PoolingLayer>();
		}
		else if (layerType == FullconnectLayer::layerType)
		{
			return std::make_shared<FullconnectLayer>();
		}
		else if (layerType == SoftmaxLayer::layerType)
		{
			return std::make_shared<SoftmaxLayer>();
		}
		else if (layerType == SigmodLayer::layerType)
		{
			return std::make_shared<SigmodLayer>();
		}
		else if (layerType == TanhLayer::layerType)
		{
			return std::make_shared<TanhLayer>();
		}
		else if (layerType == ReluLayer::layerType)
		{
			return std::make_shared<ReluLayer>();
		}
		else if (layerType == DropoutLayer::layerType)
		{
			return std::make_shared<DropoutLayer>();
		}
		else if (layerType == BatchNormalizationLayer::layerType)
		{
			return std::make_shared<BatchNormalizationLayer>();
		}
		else
		{
			logVerbose("layer type : %s", layerType.c_str());
			easyAssert(false, "can't goto here.");
			return nullptr;
		}
	}
	std::string NetWork::lookaheadLayerType(const std::string line)
	{
		std::stringstream ss(line);
		std::string layerType = "unknown";
		ss >> layerType;
		return layerType;
	}
	std::shared_ptr<DataBucket> NetWork::forward(const std::shared_ptr<DataBucket> inputDataBucket)
	{
		logVerbose("NetWork forward begin.");
		easyAssert(layers.size() > 1, "layer count is less than 2.");
		easyAssert(layers[0]->getLayerType() == InputLayer::layerType, "first layer is not input layer.");
		easyAssert(dataBuckets.size() > 0, "data buckets is not ready.");
		//copy data from inputDataBucket
		//reshape data bucket
		const auto oldNumber = dataBuckets[0]->getSize().number;
		const auto newNumber = inputDataBucket->getSize().number;
		if (newNumber != oldNumber)
		{
			for (size_t i = 0; i < dataBuckets.size(); i++)
			{
				auto newSize = dataBuckets[i]->getSize();
				newSize.number = newNumber;
				dataBuckets[i].reset(new DataBucket(newSize));
			}
		}
		inputDataBucket->cloneTo(*dataBuckets[0]);

		for (size_t i = 0; i < layers.size(); i++)
		{
			logVerbose("NetWork layer[%d](%s) forward begin.", i, layers[i]->getLayerType().c_str());
			if (i < layers.size() - 1)
			{
				dataBuckets[i + 1]->fillData(0.0f);
			}
			layers[i]->forward(dataBuckets[i], dataBuckets[i + 1]);
			logVerbose("NetWork layer[%d](%s) forward end.", i, layers[i]->getLayerType().c_str());
		}

		logVerbose("NetWork forward end.");
		return dataBuckets[dataBuckets.size() - 1];
	}
	float NetWork::backward(const std::shared_ptr<DataBucket> labelDataBucket)
	{
		easyAssert(phase == Phase::Train, "phase must be train!");
		logVerbose("NetWork backward begin.");
		easyAssert(layers.size() > 1, "layer count is less than 2.");
		easyAssert(layers[0]->getLayerType() == InputLayer::layerType, "first layer is not input layer.");
		easyAssert(lossFunctor.get() != nullptr, "loss functor can't be empty!");

		const auto lastOutputData = dataBuckets[dataBuckets.size() - 1];
		easyAssert(lastOutputData->getSize() == labelDataBucket->getSize(), "last data bucket's size must be equals with label.");

		//get loss
		const float loss = getLoss(labelDataBucket, lastOutputData);

		//get diff
		if (diffBuckets.size() != layers.size()+1)
		{
			diffBuckets.push_back(std::make_shared<DataBucket>(labelDataBucket->getSize()));
		}
		for (size_t i = 0; i < dataBuckets.size(); i++)
		{
			if (diffBuckets[i]->getSize() != dataBuckets[i]->getSize())
			{
				diffBuckets[i].reset(new DataBucket(dataBuckets[i]->getSize()));
			}
		}
		if (diffBuckets[diffBuckets.size()-1]->getSize() != labelDataBucket->getSize())
		{
			diffBuckets[diffBuckets.size() - 1].reset(new DataBucket(labelDataBucket->getSize()));
		}

		lossFunctor->getDiff(labelDataBucket, lastOutputData, diffBuckets[diffBuckets.size() - 1]);		
		//other layer backward
		for (int i = (int)(layers.size()) - 1; i >= 0; i--)
		{
			logVerbose("NetWork layer[%d](%s) backward begin.", i, layers[i]->getLayerType().c_str());
			diffBuckets[i]->fillData(0.0f);
			layers[i]->backward(dataBuckets[i], dataBuckets[i + 1], diffBuckets[i], diffBuckets[i+1]);
			logVerbose("NetWork layer[%d](%s) backward end.", i, layers[i]->getLayerType().c_str());
		}


		//update parameters
		for (int i = (int)(layers.size()) - 1; i >= 0; i--)
		{
			logVerbose("NetWork layer[%d](%s) backward begin.", i, layers[i]->getLayerType().c_str());
			optimizer->update(layers[i]->getParamData(), layers[i]->getDiffData());
			logVerbose("NetWork layer[%d](%s) backward end.", i, layers[i]->getLayerType().c_str());
		}

		logVerbose("NetWork backward end.");

		return loss;
	}

	//////////////////////////////////////////////////////////////////////////
	//test only!
	bool NetWork::loadModel(const std::string& modelFile)
	{
		std::ifstream ifs(modelFile);
		if (!ifs.is_open())
		{
			return false;
		}
		//network param
		std::string line;
		std::getline(ifs, line);
		line = decrypt(line);		
		//get Input
		{
			const std::string layerType = lookaheadLayerType(line);
			easyAssert(layerType == InputLayer::layerType, "The first layer must be InputLayer!");
			std::shared_ptr<Layer> layer = createLayerByType(layerType);
			easyAssert(layer.get() != nullptr, "layer can't be null.");
			layer->serializeFromString(line);
			setInputSize(layer->getInputBucketSize());
			addayer(layer);
		}
		//layers' param
		while (!ifs.eof())
		{
			std::getline(ifs, line);
			line = decrypt(line);		
			if (line.size() <= 2)
			{
				continue;
			}
			const std::string layerType = lookaheadLayerType(line);
			std::shared_ptr<Layer> layer = createLayerByType(layerType);
			easyAssert(layer.get() != nullptr, "layer can't be null.");		
			//init input size
			const std::shared_ptr<DataBucket> prev = dataBuckets[dataBuckets.size() - 1];
			easyAssert(prev.get() != nullptr, "previous bucket is null.");
			const DataSize inputSize = prev->getSize();
			layer->setInputBucketSize(inputSize);
			layer->serializeFromString(line);
			addayer(layer);
		}
		setPhase(Phase::Test);
		return true;
	}
	//train phase may use this
	std::shared_ptr<DataBucket> NetWork::testBatch(const std::shared_ptr<DataBucket> inputDataBucket)
	{
		setPhase(Phase::Test);
		return forward(inputDataBucket);
	}

	//////////////////////////////////////////////////////////////////////////
	//train only!
	void NetWork::setInputSize(const DataSize size)
	{
		logVerbose("NetWork setInputSize begin.");
		easyAssert(size.number > 0 && size.channels > 0 && size.width > 0 && size.height > 0, "parameter invalidate.");
		easyAssert(dataBuckets.empty(), "dataBuckets must be empty now!");		
		dataBuckets.push_back(std::make_shared<DataBucket>(size));
		diffBuckets.push_back(std::make_shared<DataBucket>(size));
		logVerbose("NetWork setInputSize end.");
	}
	void NetWork::setLossFunctor(std::shared_ptr<LossFunctor> lossFunctor)
	{
		logVerbose("NetWork setInputSize begin.");
		this->lossFunctor = lossFunctor;
		logVerbose("NetWork setInputSize end.");
	}
	void NetWork::setOptimizer(std::shared_ptr<Optimizer> optimizer)
	{
		logVerbose("NetWork setOptimizer begin.");
		this->optimizer = optimizer;
		logVerbose("NetWork setOptimizer end.");
	}
	void NetWork::setLearningRate(const float lr)
	{
		this->optimizer->setLearningRate(lr);
	}
	void NetWork::addayer(std::shared_ptr<Layer> layer)
	{
		const auto layer_type = layer->getLayerType();
		logVerbose("NetWork addayer begin , type : %s", layer_type.c_str());
		layers.push_back(layer);

		easyAssert(dataBuckets.size() >= 1, "bucket count is less than 1.");
		const std::shared_ptr<DataBucket> prev = dataBuckets[dataBuckets.size() - 1];
		easyAssert(prev.get() != nullptr, "previous bucket is null.");
		const DataSize inputSize = prev->getSize();
		layer->setPhase(phase);
		layer->setInputBucketSize(inputSize);
		layer->solveInnerParams();
		const DataSize outputSize = layer->getOutputBucketSize();
		//dataBucket setting params
		dataBuckets.push_back(std::make_shared<DataBucket>(outputSize));
		diffBuckets.push_back(std::make_shared<DataBucket>(outputSize));
		logVerbose("NetWork addayer end. add data bucket done.");
	}
	float NetWork::trainBatch(const std::shared_ptr<DataBucket> inputDataBucket,
		const std::shared_ptr<DataBucket> labelDataBucket)
	{
		setPhase(Phase::Train);
		logVerbose("NetWork trainBatch begin.");
		forward(inputDataBucket);
		const float loss = backward(labelDataBucket);
		logVerbose("NetWork trainBatch end.");
		return loss;
	}
	bool NetWork::saveModel(const std::string& modelFile)
	{
		std::ofstream ofs(modelFile);
		if (!ofs.is_open())
		{
			return false;
		}
		//layers' param
		for (const auto& layer : layers)
		{
			ofs << encrypt(layer->serializeToString()) + "\n";
		}
		return true;
	}

	//toy crypt only now! you can custom it.
	std::string NetWork::encrypt(const std::string& content)
	{
#if WITH_ENCRYPT_MODEL
		std::string message = content;
		for (size_t i = 0; i < message.size(); i++)
		{
			message[i] -= 15;
		}
		return message;
#else
		return content;
#endif
	}
	std::string NetWork::decrypt(const std::string& content)
	{
#if WITH_ENCRYPT_MODEL		
		std::string message = content;
		for (size_t i = 0; i < message.size(); i++)
		{
			message[i] += 15;
		}
		return message;
#else
		return content;
#endif 
	}
}//namespace
