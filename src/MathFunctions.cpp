#include <cmath>
#include <random>
#include "EasyCNN/MathFunctions.h"

namespace EasyCNN
{
	void normal_distribution_init(float* data, const size_t size, const float mean_value, const float standard_deviation)
	{
		std::random_device rd;
		std::mt19937 engine(rd());
		std::normal_distribution<float> dist(mean_value, standard_deviation);
		for (size_t i = 0; i < size; i++)
		{
			data[i] = dist(engine);
		}
	}
	void const_distribution_init(float* data, const size_t size, const float const_value)
	{
		for (size_t i = 0; i < size; i++)
		{
			data[i] = const_value;
		}
	}

	void mul(const float* a, const float* b, float* c, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			c[i] = a[i]*b[i];
		}
	}
	void mul_inplace(float* a_inplace, const float* b, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			a_inplace[i] *= b[i];
		}
	}
	//f(x)=1/(1+e^(-x))
	static inline float sigmoid(const float x)
	{
		float result = 0;
		result = 1.0f / (1.0f + std::exp(-1.0f*x));
		return result;
	}
	void sigmoid(const float* x, float* y, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			y[i] = sigmoid(x[i]);
		}
	}
	//f'(x) = x(1-x)
	static inline float df_sigmoid(const float x)
	{
		return x*(1.0f - x);
	}
	void df_sigmoid(const float* x, float* y, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			y[i] = df_sigmoid(x[i]);
		}
	}

	//f(x)=(e^x-e^(-x))/(e^x+e^(-x))
	static inline float tanh(const float x)
	{
		float result = 0;
		const float ex = std::exp(x);
		const float efx = std::exp(-x);
		result = (ex - efx) / (ex + efx);
		return result;
	}
	void tanh(const float* x, float* y, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			y[i] = tanh(x[i]);
		}
	}
	//f'(x)=1-x^(1/2)
	static inline float df_tanh(const float x)
	{
		return 1.0f - std::sqrt(x);
	}
	void df_tanh(const float* x, float* y, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			y[i] = df_tanh(x[i]);
		}
	}

	//f(x)=max(x,0)
	static inline float relu(const float x)
	{
#define MAX_OP(a,b) (a)>(b)?(a):(b)
		float result = MAX_OP(x, 0.0f);
#undef MAX_OP
		return result;
	}
	void relu(const float* x, float* y, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			y[i] = relu(x[i]);
		}
	}
	//f'(x)=0(x<=0),1(x>0)
	static inline float df_relu(const float x)
	{
		//note : too small df is not suitable.
		return x <= 0.0f ? 0.01f : 1.0f;
	}
	void df_relu(const float* x, float* y, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			y[i] = df_relu(x[i]);
		}
	}
}//namespace