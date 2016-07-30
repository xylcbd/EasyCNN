#include "EasyCNN/ParamBucket.h"

EasyCNN::ParamBucket::ParamBucket(const ParamSize _size)
	:size(_size),
	data(new float[size._4DSize()])
{
}
EasyCNN::ParamBucket::~ParamBucket()
{

}
void EasyCNN::ParamBucket::cloneTo(ParamBucket& target)
{
	target.size = this->size;
	const size_t dataSize = sizeof(float)*this->size._4DSize();
	memcpy(target.data.get(), this->data.get(), dataSize);
}
void EasyCNN::ParamBucket::fillData(const float item)
{
	std::fill(data.get(), data.get() + getSize()._4DSize(), item);
}
std::shared_ptr<float> EasyCNN::ParamBucket::getData() const
{
	return data;
}
EasyCNN::ParamSize EasyCNN::ParamBucket::getSize() const
{
	return size;
}