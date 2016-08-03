#include <algorithm>
#include <fstream>
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
//network
#include "EasyCNN/NetWork.h"


EasyCNN::NetWork::NetWork()
{
	logVerbose("NetWork constructed.");
}
EasyCNN::NetWork::~NetWork()
{
	logVerbose("NetWork destructed.");
}

//////////////////////////////////////////////////////////////////////////
//common
void EasyCNN::NetWork::setPhase(Phase phase)
{
	logVerbose("NetWork setPhase begin.");
	this->phase = phase;
	logVerbose("NetWork setPhase end.");
}
EasyCNN::Phase EasyCNN::NetWork::getPhase() const
{
	return phase;
}
std::string EasyCNN::NetWork::serializeToString() const
{
	const std::string spliter = " ";
	std::stringstream ss;
	const auto inputSize = dataBuckets[0]->getSize();
	ss << inputSize.channels << spliter << inputSize.width << spliter << inputSize.height << spliter;
	for (const auto& layer : layers)
	{
		ss << layer->getLayerType() << spliter;
	}
	return ss.str();
}
std::shared_ptr<EasyCNN::Layer> EasyCNN::NetWork::createLayerByType(const std::string layerType)
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
	else
	{
		easyAssert(false,"can't goto here.");
		return nullptr;
	}
}
std::vector<std::shared_ptr<EasyCNN::Layer>> EasyCNN::NetWork::serializeFromString(const std::string content)
{
	int number = 1;
	int channels = 0;
	int width = 0;
	int height = 0;
	std::stringstream ss(content);
	ss >> channels >> width >> height;
	setInputSize(DataSize(number, channels, width, height));
	std::vector<std::shared_ptr<EasyCNN::Layer>> tmpLayers;	
	while (!ss.eof())
	{
		std::string layerType;
		ss >> layerType;
		if (layerType.empty())
		{
			continue;
		}
		std::shared_ptr<EasyCNN::Layer> layer = createLayerByType(layerType);
		easyAssert(layer.get() != nullptr,"layer can't be null.");
		tmpLayers.push_back(layer);
	}
	return tmpLayers;
}
std::shared_ptr<EasyCNN::DataBucket> EasyCNN::NetWork::forward(const std::shared_ptr<DataBucket> inputDataBucket)
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
		layers[i]->forward(dataBuckets[i], dataBuckets[i + 1]);
		logVerbose("NetWork layer[%d](%s) forward end.", i, layers[i]->getLayerType().c_str());
	}

	logVerbose("NetWork forward end.");
	return dataBuckets[dataBuckets.size() - 1];
}
float EasyCNN::NetWork::backward(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket, const float learningRate)
{
	logVerbose("NetWork backward begin.");
	easyAssert(layers.size() > 1, "layer count is less than 2.");
	easyAssert(layers[0]->getLayerType() == InputLayer::layerType, "first layer is not input layer.");
	easyAssert(lossFunctor.get() != nullptr, "loss functor can't be empty!");

	const auto lastOutputData = dataBuckets[dataBuckets.size() - 1];
	easyAssert(lastOutputData->getSize() == labelDataBucket->getSize(), "last data bucket's size must be equals with label.");

	//get loss
	const float loss = lossFunctor->getLoss(labelDataBucket, lastOutputData);

	//get diff
	std::shared_ptr<ParamBucket> nextDiffBucket = lossFunctor->getDiff(labelDataBucket, lastOutputData);

	//other layer backward
	for (int i = (int)(layers.size()) - 1; i >= 0; i--)
	{
		logVerbose("NetWork layer[%d](%s) backward begin.", i, layers[i]->getLayerType().c_str());
		layers[i]->setLearningRate(learningRate);
		layers[i]->backward(dataBuckets[i], dataBuckets[i + 1], nextDiffBucket);
		logVerbose("NetWork layer[%d](%s) backward end.", i, layers[i]->getLayerType().c_str());
	}
	logVerbose("NetWork backward end.");

	return loss;
}

//////////////////////////////////////////////////////////////////////////
//test only!
bool EasyCNN::NetWork::loadModel(const std::string& modelFile)
{
	std::ifstream ifs(modelFile);
	if (!ifs.is_open())
	{
		return false;
	}
	//network param
	std::string line;
	std::getline(ifs, line);
	std::vector<std::shared_ptr<EasyCNN::Layer>> tmpLayers = this->serializeFromString(line);
	//layers' param
	for (auto& layer : tmpLayers)
	{
		std::getline(ifs, line);
		//init input size
		const std::shared_ptr<DataBucket> prevDataBucket = dataBuckets[dataBuckets.size() - 1];
		easyAssert(prevDataBucket.get() != nullptr, "previous bucket is null.");
		const DataSize inputSize = prevDataBucket->getSize();
		layer->setInputBucketSize(inputSize);
		layer->serializeFromString(line);
		addayer(layer);
	}
	setPhase(Phase::Test);
	return false;
}
//train phase may use this
std::shared_ptr<EasyCNN::DataBucket> EasyCNN::NetWork::testBatch(const std::shared_ptr<DataBucket> inputDataBucket)
{
	return forward(inputDataBucket);
}

//////////////////////////////////////////////////////////////////////////
//train only!
void EasyCNN::NetWork::setInputSize(const DataSize size)
{
	logVerbose("NetWork setInputSize begin.");
	easyAssert(size.number > 0 && size.channels > 0 && size.width > 0 && size.height > 0, "parameter invalidate.");
	easyAssert(dataBuckets.empty(), "dataBuckets must be empty now!");
	std::shared_ptr<DataBucket> dataBucket = std::make_shared<DataBucket>(size);
	dataBuckets.push_back(dataBucket);
	logVerbose("NetWork setInputSize end.");
}
void EasyCNN::NetWork::setLossFunctor(std::shared_ptr<LossFunctor> lossFunctor)
{
	logVerbose("NetWork setInputSize begin.");
	this->lossFunctor = lossFunctor;
	logVerbose("NetWork setInputSize end.");
}
void EasyCNN::NetWork::addayer(std::shared_ptr<Layer> layer)
{
	const auto layer_type = layer->getLayerType();
	logVerbose("NetWork addayer begin , type : %s", layer_type.c_str());
	layers.push_back(layer);

	easyAssert(dataBuckets.size() >= 1, "bucket count is less than 1.");
	const std::shared_ptr<DataBucket> prevDataBucket = dataBuckets[dataBuckets.size() - 1];
	easyAssert(prevDataBucket.get() != nullptr, "previous bucket is null.");
	const DataSize inputSize = prevDataBucket->getSize();
	layer->setPhase(phase);
	layer->setInputBucketSize(inputSize);
	layer->solveInnerParams();
	const DataSize outputSize = layer->getOutputBucketSize();
	std::shared_ptr<DataBucket> dataBucket = std::make_shared<DataBucket>(outputSize);
	//dataBucket setting params
	dataBuckets.push_back(dataBucket);
	logVerbose("NetWork addayer end. add data bucket done.");
}
float EasyCNN::NetWork::trainBatch(const std::shared_ptr<DataBucket> inputDataBucket,
	const std::shared_ptr<DataBucket> labelDataBucket, float learningRate)
{
	easyAssert(phase == Phase::Train, "phase must be train!");
	logVerbose("NetWork trainBatch begin.");
	forward(inputDataBucket);
	const float loss = backward(labelDataBucket, learningRate);
	logVerbose("NetWork trainBatch end.");
	return loss;
}
bool EasyCNN::NetWork::saveModel(const std::string& modelFile)
{
	std::ofstream ofs(modelFile);
	if (!ofs.is_open())
	{
		return false;
	}
	//network param
	ofs << this->serializeToString() << std::endl;
	//layers' param
	for (const auto& layer : layers)
	{		
		ofs << layer->serializeToString() << std::endl;
	}
	return true;
}