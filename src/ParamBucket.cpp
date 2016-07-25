#include "EasyCNN/ParamBucket.h"

EasyCNN::ParamBucket::ParamBucket(const ParamSize _size)
	:size(_size),
	data(new float[size.number*size.channels*size.width*size.height])
{
}
EasyCNN::ParamBucket::~ParamBucket()
{

}
void EasyCNN::ParamBucket::cloneTo(ParamBucket& target)
{
	target.size = this->size;
	const int dataSize = sizeof(float)*this->size.totalSize();
	memcpy(target.data.get(), this->data.get(), dataSize);
}
std::shared_ptr<float> EasyCNN::ParamBucket::getData() const
{
	return data;
}
EasyCNN::ParamSize EasyCNN::ParamBucket::getSize() const
{
	return size;
}