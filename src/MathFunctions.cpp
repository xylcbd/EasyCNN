#include <cmath>
#include <random>
#include "EasyCNN/MathFunctions.h"
#include "EasyCNN/DataBucket.h"

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
	void uniform_distribution_init(float* data, const size_t size, const float low_value, const float high_deviation)
	{
		std::random_device rd;
		std::mt19937 engine(rd());
		std::uniform_real_distribution<float> dist(low_value, high_deviation);
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
	void xavier_init(float* data, const size_t size, const size_t fan_in, const size_t fan_out)
	{
		const float weight_base = std::sqrt(6.0f / float(fan_in + fan_out));
		uniform_distribution_init(data, size, -weight_base, weight_base);
	}


	float moving_average(float avg, const int acc_number, float value)
	{
		avg -= avg / acc_number;
		avg += value / acc_number;
		return avg;
	}

	void mul(const float* a, const float* b, float* c, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			c[i] = a[i]*b[i];
		}
	}
	void mul_inplace(float* a, const float* b, const size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			a[i] *= b[i];
		}
	}

	//a /= b
	void div_inplace(float* a, const float b, const size_t len)
	{
		for (size_t i = 0; i < len;i++)
		{
			a[i] /= b;
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

	//
	void fullconnect(const float* input, const float* weight, const float* bias, float* output,
		const size_t n, const size_t is, const size_t os)
	{
		if (bias)
		{
			for (size_t k = 0; k < n; k++)
			{
				const float* n_input = input + k*is;
				float* n_output = output + k*os;
				for (size_t i = 0; i < os; i++)
				{
					float sum = 0.0f;
					for (size_t j = 0; j < is; j++)
					{
						sum += n_input[j] * weight[i*is + j];
					}
					sum += bias[i];
					n_output[i] = sum;
				}
			}
		}
		else
		{
			for (size_t k = 0; k < n; k++)
			{
				const float* n_input = input + k*is;
				float* n_output = output + k*os;
				for (size_t i = 0; i < os; i++)
				{
					float sum = 0.0f;
					for (size_t j = 0; j < is; j++)
					{
						sum += n_input[j] * weight[i*is + j];
					}
					n_output[i] = sum;
				}
			}
		}
	}
	
	static void convolution2d_validate(const float* input, const float* kernel, const float* bias, float* output,
		const size_t in, const size_t ic, const size_t iw, const size_t ih,
		const size_t kn, const size_t kw, const size_t kh, const size_t kws, const size_t khs,
		const size_t ow, const size_t oh)
	{
		const DataSize inputSize(in, ic, iw, ih);
		const DataSize kernelSize(kn, ic, kw, kh);
		const DataSize outputSize(in, kn, ow, oh);
		for (size_t nn = 0; nn < in; nn++)
		{
			for (size_t nc = 0; nc < outputSize.channels; nc++)
			{
				for (size_t nh = 0; nh < outputSize.height; nh++)
				{
					for (size_t nw = 0; nw < outputSize.width; nw++)
					{
						const size_t inStartX = nw*kws;
						const size_t inStartY = nh*khs;
						float sum = 0;
						for (size_t kc = 0; kc < kernelSize.channels; kc++)
						{
							for (size_t kh = 0; kh < kernelSize.height; kh++)
							{
								for (size_t kw = 0; kw < kernelSize.width; kw++)
								{
									const size_t prevIdx = inputSize.getIndex(nn, kc, inStartY + kh, inStartX + kw);
									const size_t kernelIdx = kernelSize.getIndex(nc, kc, kh, kw);
									sum += input[prevIdx] * kernel[kernelIdx];
								}
							}
						}
						if (bias)
						{
							const size_t biasIdx = nc;
							sum += bias[biasIdx];
						}
						const size_t nextIdx = outputSize.getIndex(nn, nc, nh, nw);
						output[nextIdx] = sum;
					}
				}
			}
		}
	}
	static void convolution2d_same(const float* input, const float* kernel, const float* bias, float* output,
		const size_t in, const size_t ic, const size_t iw, const size_t ih,
		const size_t kn, const size_t kw, const size_t kh, const size_t kws, const size_t khs,
		const size_t ow, const size_t oh)
	{
		const DataSize inputSize(in, ic, iw, ih);
		const DataSize kernelSize(kn, ic, kw, kh);
		const DataSize outputSize(in, kn, ow, oh);
		for (size_t nn = 0; nn < in; nn++)
		{
			for (size_t nc = 0; nc < outputSize.channels; nc++)
			{
				for (size_t nh = 0; nh < outputSize.height; nh++)
				{
					for (size_t nw = 0; nw < outputSize.width; nw++)
					{
						const int inStartX = nw - kw/2;
						const int inStartY = nh - kh / 2;
						float sum = 0;
						for (size_t kc = 0; kc < kernelSize.channels; kc++)
						{
							for (size_t kh = 0; kh < kernelSize.height; kh++)
							{
								for (size_t kw = 0; kw < kernelSize.width; kw++)
								{
									const int inY = inStartY + kh;
									const int inX = inStartX + kw;
									if (inY >= 0 && inY<(int)inputSize.height && inX >= 0 && inX<(int)inputSize.width)
									{
										const size_t prevIdx = inputSize.getIndex(nn, kc, inY, inX);
										const size_t kernelIdx = kernelSize.getIndex(nc, kc, kh, kw);
										sum += input[prevIdx] * kernel[kernelIdx];
									}									
								}
							}
						}
						if (bias)
						{
							const size_t biasIdx = nc;
							sum += bias[biasIdx];
						}
						const size_t nextIdx = outputSize.getIndex(nn, nc, nh, nw);
						output[nextIdx] = sum;
					}
				}
			}
		}
	}
	void convolution2d(const float* input, const float* kernel, const float* bias, float* output,
		const size_t in, const size_t ic, const size_t iw, const size_t ih,
		const size_t kn, const size_t kw, const size_t kh, const size_t kws, const size_t khs,
		const size_t ow, const size_t oh,
		const int mode)
	{
		if (mode == 0)
		{
			convolution2d_validate(input, kernel, bias, output, in, ic, iw, ih, kn, kw, kh, kws, khs, ow, oh);
		}else if (mode == 1)
		{
			convolution2d_same(input, kernel, bias, output, in, ic, iw, ih, kn, kw, kh, kws, khs, ow, oh);
		}
	}
}//namespace