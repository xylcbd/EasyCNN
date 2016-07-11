#include "EasyCNN/EasyAssert.h"
#include "EasyCNN/EasyLogger.h"

static void* assertFatalUserData = nullptr;
static void (*assertFatalCB)(void* userData,const std::string& errorStr) = nullptr;

void setAssertFatalCallback(void(*cb)(void* userData, const std::string& errorStr), void* userData)
{
	assertFatalCB = cb;
	assertFatalUserData = userData;
}
void EasyCNN::easyAssert(const bool condition, const char* fmt, ...)
{
	if (!condition)
	{
		va_list args;
		va_start(args, fmt);
		const std::string errorStr = EASYCNN_LOG_FORMATSTRING(fmt, args);
		EASYCNN_LOG_FATAL(fmt, args);
		va_end(args);
		if (assertFatalCB)
		{
			assertFatalCB(assertFatalUserData, errorStr);
		}
		else
		{
			exit(0);
		}
	}
}
