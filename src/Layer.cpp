#include "EasyCNN/Layer.h"

void EasyCNN::Layer::setInputBucketSize(const EasyCNN::BucketSize size)
{
	inputSize = size;
}
EasyCNN::BucketSize EasyCNN::Layer::getInputBucketSize() const
{
	return inputSize;
}
void EasyCNN::Layer::solveInnerParams()
{
	outputSize = inputSize;
}
void EasyCNN::Layer::setOutpuBuckerSize(const BucketSize size)
{
	outputSize = size;
}
EasyCNN::BucketSize EasyCNN::Layer::getOutputBucketSize() const
{
	return outputSize;
}