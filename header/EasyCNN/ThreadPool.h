/*
*  Modified for EasyCNN(https://github.com/xylcbd/EasyCNN) by (https://github.com/xylcbd) based on ThreadPool(https://github.com/progschj/ThreadPool)
*  License: WTFPL
*/
#pragma once
#include <functional>
#include "EasyCNN/Configure.h"

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace EasyCNN
{
	//TODO: using wrapper to hidden information of ThreadPool
	class ThreadPool {			
	public:	
		static ThreadPool& instance();
		size_t size() const;
		void resize(const size_t new_size);
		template<class F, class... Args>
		auto enqueue(F&& f, Args&&... args)
			->std::future<typename std::result_of<F(Args...)>::type>;	
	private:
		ThreadPool(const size_t threads);
		virtual ~ThreadPool();
		void shutdown();
		void startup(const size_t threads);
	private:
		// need to keep track of threads so we can join them
		std::vector< std::thread > workers;
		// the task queue
		std::queue< std::function<void()> > tasks;

		// synchronization
		std::mutex queue_mutex;
		std::condition_variable condition;
		bool stop = true;
	};
	// add new work item to the pool
	template<class F, class... Args>
	auto ThreadPool::enqueue(F&& f, Args&&... args)
		-> std::future<typename std::result_of<F(Args...)>::type>
	{
		using return_type = typename std::result_of<F(Args...)>::type;

		auto task = std::make_shared< std::packaged_task<return_type()> >(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
			);

		std::future<return_type> res = task->get_future();
		{
			std::unique_lock<std::mutex> lock(queue_mutex);

			// don't allow enqueueing after stopping the pool
			if (stop)
				throw std::runtime_error("enqueue on stopped ThreadPool");

			tasks.emplace([task](){ (*task)(); });
		}
		condition.notify_one();
		return res;
	}

	//////////////////////////////////////////////////////////////////////////
	//APIs
	//get thread number
	size_t get_thread_num();
	//set thread number, and returned support thread number
	size_t set_thread_num(const size_t num);
	//dispatcher tasks of layer
	void dispatch_worker(std::function<void(const size_t, const size_t)> func, const size_t number);
};