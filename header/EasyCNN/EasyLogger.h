#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <ctime>
#include <sstream>
#include <cstdarg>
#include <functional>
#include "EasyCNN/Configure.h"

namespace EasyCNN
{
	enum LogLevel
	{		
		EASYCNN_LOG_LEVEL_VERBOSE,
		EASYCNN_LOG_LEVEL_CRITICAL,
		EASYCNN_LOG_LEVEL_FATAL,
		EASYCNN_LOG_LEVEL_NONE
	};
	void setLogLevel(const LogLevel level);
	LogLevel getLogLevel();
	//default : console
	void setLogRedirect(std::function<void(const LogLevel,const std::string)> logCb);
	//
	void logVerbose(const char* fmt, ...);
	void logCritical(const char* fmt, ...);
	void logFatal(const char* fmt, ...);
}