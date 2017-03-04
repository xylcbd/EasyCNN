#include <algorithm>
#include "EasyCNN/DataBucket.h"

namespace EasyCNN
{
	DataBucket::DataBucket(const DataSize _size)
		:size(_size),
		data(new float[size.totalSize()])
	{

	}
	DataBucket::~DataBucket()
	{

	}
	void DataBucket::fillData(const float item)
	{
		std::fill(data.get(), data.get() + getSize().totalSize(), item);
	}
	void DataBucket::cloneTo(DataBucket& target)
	{
		target.size = this->size;
		const size_t dataSize = sizeof(float)*this->size.totalSize();
		memcpy(target.data.get(), this->data.get(), dataSize);
	}
	std::shared_ptr<float> DataBucket::getData() const
	{
		return data;
	}
	DataSize DataBucket::getSize() const
	{
		return size;
	}
}//namespace