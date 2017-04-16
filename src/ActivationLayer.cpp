#include <algorithm>
#include "EasyCNN/ActivationLayer.h"
#include "EasyCNN/MathFunctions.h"
#include "EasyCNN/ThreadPool.h"

namespace EasyCNN
{
	SigmodLayer::SigmodLayer()
	{

	}
	SigmodLayer::~SigmodLayer()
	{

	}
	DEFINE_LAYER_TYPE(SigmodLayer, "SigmodLayer");
	std::string SigmodLayer::getLayerType() const
	{
		return layerType;
	}
	void SigmodLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();

		auto worker = [&](const size_t start,const size_t stop){
			const size_t offset = start*prevSize._3DSize();
			const size_t total_size = (stop - start)*prevSize._3DSize();
			sigmoid(prevData + offset, nextData + offset, total_size);
		};
		dispatch_worker(worker, prevSize.number);
	}
	void SigmodLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//update prevDiff
		prevDiff->fillData(0.0f);

		auto worker = [&](const size_t start, const size_t stop){
			const size_t offset = start*prevSize._3DSize();
			const size_t total_size = (stop - start)*prevSize._3DSize();
			//calculate current inner diff
			df_sigmoid(nextData + offset, prevDiffData + offset, total_size);
			//multiply next diff
			mul_inplace(prevDiffData + offset, nextDiffData + offset, total_size);
		};
		dispatch_worker(worker, prevSize.number);

		//update this layer's param
		//Tanh layer : nop
	}

	TanhLayer::TanhLayer()
	{

	}
	TanhLayer::~TanhLayer()
	{

	}
	DEFINE_LAYER_TYPE(TanhLayer, "TanhLayer");
	std::string TanhLayer::getLayerType() const
	{
		return layerType;
	}


	void TanhLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();

		auto worker = [&](const size_t start, const size_t stop){
			const size_t offset = start*prevSize._3DSize();
			const size_t total_size = (stop - start)*prevSize._3DSize();
			tanh(prevData + offset, nextData + offset, total_size);
		};
		dispatch_worker(worker, prevSize.number);
	}
	void TanhLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
			const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//update prevDiff
		prevDiff->fillData(0.0f);		

		auto worker = [&](const size_t start, const size_t stop){
			const size_t offset = start*prevSize._3DSize();
			const size_t total_size = (stop - start)*prevSize._3DSize();
			//calculate current inner diff
			df_tanh(nextData + offset, prevDiffData + offset, total_size);
			//multiply next diff
			mul_inplace(prevDiffData + offset, nextDiffData + offset, total_size);
		};
		dispatch_worker(worker, prevSize.number);

		//update this layer's param
		//Tanh layer : nop
	}

	ReluLayer::ReluLayer()
	{

	}
	ReluLayer::~ReluLayer()
	{

	}
	DEFINE_LAYER_TYPE(ReluLayer, "ReluLayer");
	std::string ReluLayer::getLayerType() const
	{
		return layerType;
	}


	void ReluLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();

		auto worker = [&](const size_t start, const size_t stop){
			const size_t offset = start*prevSize._3DSize();
			const size_t total_size = (stop - start)*prevSize._3DSize();
			relu(prevData + offset, nextData + offset, total_size);
		};
		dispatch_worker(worker, prevSize.number);
	}
	void ReluLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//update prevDiff		
		prevDiff->fillData(0.0f);	

		auto worker = [&](const size_t start, const size_t stop){
			const size_t offset = start*prevSize._3DSize();
			const size_t total_size = (stop - start)*prevSize._3DSize();
			//calculate current inner diff
			df_relu(nextData + offset, prevDiffData + offset, total_size);
			//multiply next diff
			mul_inplace(prevDiffData + offset, nextDiffData + offset, total_size);
		};
		dispatch_worker(worker, prevSize.number);

		//update this layer's param
		//RELU layer : nop
	}
}//namespace