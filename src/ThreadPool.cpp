#include <algorithm>
#include "EasyCNN/ThreadPool.h"
#include "EasyCNN/EasyAssert.h"

namespace EasyCNN
{
	ThreadPool& ThreadPool::instance()
	{
		static ThreadPool inst(2);
		return inst;
	}
	// the constructor just launches some amount of workers
	ThreadPool::ThreadPool(const size_t threads)
	{
		startup(threads);
	}
	// the destructor joins all threads
	ThreadPool::~ThreadPool()
	{
		shutdown();
	}
	void ThreadPool::shutdown()
	{
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			stop = true;
		}
		condition.notify_all();
		for (std::thread &worker : workers)
		{
			if (worker.joinable())
			{
				worker.join();
			}
		}
		workers.clear();
		tasks = decltype(tasks)();
	}
	void ThreadPool::startup(const size_t threads)
	{
		stop = false;
		for (size_t i = 0; i < threads; ++i)
			workers.emplace_back(
			[this]
		{
			for (;;)
			{
				std::function<void()> task;

				{
					std::unique_lock<std::mutex> lock(this->queue_mutex);
					this->condition.wait(lock,
						[this]{ return this->stop || !this->tasks.empty(); });
					if (this->stop && this->tasks.empty())
						return;
					task = std::move(this->tasks.front());
					this->tasks.pop();
				}

				task();
			}
		}
		);
	}
	inline size_t ThreadPool::size() const
	{
		return workers.size();
	}
	void ThreadPool::resize(const size_t new_size)
	{
		//stop thread pool
		shutdown();
		//start thread pool
		startup(new_size);
	}

	//////////////////////////////////////////////////////////////////////////
	//APIs
	size_t get_thread_num()
	{
		return ThreadPool::instance().size();
	}
	size_t set_thread_num(const size_t num)
	{
		easyAssert(num > 0, "number of thread must be larger than 0");
		if (num != get_thread_num())
		{
			ThreadPool::instance().resize(num);
		}
		return get_thread_num();
	}
	void dispatch_worker(std::function<void(const size_t, const size_t)> func, const size_t number)
	{
		if (number <= 0)
		{
			return;
		}
		const size_t threads_of_pool = ThreadPool::instance().size();
		if (threads_of_pool <= 1 || number <= 1)
		{
			func(0, number);
		}
		else
		{
			easyAssert(threads_of_pool > 1, "thread must be larger than 1");
			// 1/4 2/4 /4/4 5/4 => all ok!
			const size_t payload_per_thread = number / threads_of_pool;
			const size_t remainder_payload = number - payload_per_thread*threads_of_pool;
			const size_t remainder_proc_last_idx = remainder_payload;

			size_t start = 0;
			std::vector<std::future<void>> futures;
			for (size_t i = 0; i < threads_of_pool; i++)
			{
				size_t stop = start + payload_per_thread;
				if (i < remainder_proc_last_idx)
				{
					stop = stop + 1;
				}
				futures.push_back(ThreadPool::instance().enqueue(func, start, stop));
				start = stop;
				if (stop >= number)
				{
					break;
				}
			}
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
	}
}//namespace