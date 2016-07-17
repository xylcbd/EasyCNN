#include "EasyCNN/DataBucket.h"

EasyCNN::DataBucket::DataBucket(const BucketSize _size)
	:size(_size),
	data(new data_type[size.number*size.channels*size.width*size.height])
{
}
EasyCNN::DataBucket::~DataBucket()
{

}
void EasyCNN::DataBucket::cloneTo(DataBucket& target)
{
	target.size = this->size;
	const int dataSize = sizeof(data_type)*this->size.totalSize();
	memcpy(target.data.get(), this->data.get(), dataSize);
}
std::shared_ptr<EasyCNN::data_type> EasyCNN::DataBucket::getData() const
{
	return data;
}
EasyCNN::BucketSize EasyCNN::DataBucket::getSize() const
{
	return size;
}