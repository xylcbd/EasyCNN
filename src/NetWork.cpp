#include <cassert>
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
	EASYCNN_LOG_VERBOSE("NetWork constructed.");
}
EasyCNN::NetWork::~NetWork()
{
	EASYCNN_LOG_VERBOSE("NetWork destructed.");
}
void EasyCNN::NetWork::addayer(std::shared_ptr<Layer> layer)
{
	const auto layer_type = layer->getLayerType();
	EASYCNN_LOG_VERBOSE("NetWork addayer : %s",layer_type.c_str());
	layers.push_back(layer);
}
void EasyCNN::NetWork::backward()
{
	EASYCNN_LOG_VERBOSE("NetWork backward begin.");
	assert(layers.size() > 1);
	assert(layers[0]->getLayerType() == InputLayer::layerType);

	for (size_t i = 0; i < layers.size();i++)
	{
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) backward begin.",i,layers[i]->getLayerType().c_str());
		layers[i]->forward(dataBuckets[i], dataBuckets[i+1]);
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) backward end.", i, layers[i]->getLayerType().c_str());
	}

	EASYCNN_LOG_VERBOSE("NetWork backward end.");
}
void EasyCNN::NetWork::forward(std::shared_ptr<DataBucket> inputDataBucket)
{
	EASYCNN_LOG_VERBOSE("NetWork forward end.");
	assert(layers.size() > 1);
	assert(layers[0]->getLayerType() == InputLayer::layerType);
	dataBuckets.clear();
	dataBuckets.push_back(inputDataBucket);

	for (size_t i = 0; i < layers.size(); i++)
	{
		std::shared_ptr<DataBucket> dataBucket = std::make_shared<DataBucket>();
		//dataBucket setting params
		dataBuckets.push_back(dataBucket);

		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) forward begin.", i, layers[i]->getLayerType().c_str());
		layers[i]->forward(dataBuckets[i], dataBuckets[i+1]);
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) forward end.", i, layers[i]->getLayerType().c_str());
	}
	EASYCNN_LOG_VERBOSE("NetWork forward end.");
}