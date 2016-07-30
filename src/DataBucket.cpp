#include <algorithm>
#include "EasyCNN/DataBucket.h"

EasyCNN::DataBucket::DataBucket(const DataSize _size)
	:size(_size),
	data(new float[size._4DSize()])
{
}
EasyCNN::DataBucket::~DataBucket()
{

}
void EasyCNN::DataBucket::fillData(const float item)
{
	std::fill(data.get(), data.get() + getSize()._4DSize(), item);
}
void EasyCNN::DataBucket::cloneTo(DataBucket& target)
{
	target.size = this->size;
	const size_t dataSize = sizeof(float)*this->size._4DSize();
	memcpy(target.data.get(), this->data.get(), dataSize);
}
std::shared_ptr<float> EasyCNN::DataBucket::getData() const
{
	return data;
}
EasyCNN::DataSize EasyCNN::DataBucket::getSize() const
{
	return size;
}