#pragma once
#include <string>
#include "EasyCNN/Configure.h"

namespace EasyCNN
{
	void setAssertFatalCallback(void (*cb)(void* userData,const std::string& errorStr),void* userData);
	void easyAssert(const bool condition, const char* fmt, ...);
}