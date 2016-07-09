#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class ActivationLayer : public Layer
	{
	public:
		ActivationLayer();
		virtual ~ActivationLayer();
	private:
	};

	class SigmodLayer : public ActivationLayer
	{
	public:
		SigmodLayer();
		virtual ~SigmodLayer();
	};

	class TanhLayer : public ActivationLayer
	{
	public:
		TanhLayer();
		virtual ~TanhLayer();
	};

	class ReluLayer : public ActivationLayer
	{
	public:
		ReluLayer();
		virtual ~ReluLayer();
	};
}