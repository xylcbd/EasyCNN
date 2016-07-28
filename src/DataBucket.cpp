#include <algorithm>
#include "EasyCNN/DataBucket.h"

EasyCNN::DataBucket::DataBucket(const DataSize _size)
	:size(_size),
	data(new float[size.number*size.channels*size.width*size.height])
{
}
EasyCNN::DataBucket::~DataBucket()
{

}
void EasyCNN::DataBucket::fillData(float item)
{
	std::fill(data.get(), data.get() + getSize().totalSize(), item);
}
void EasyCNN::DataBucket::cloneTo(DataBucket& target)
{
	target.size = this->size;
	const int dataSize = sizeof(float)*this->size.totalSize();
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