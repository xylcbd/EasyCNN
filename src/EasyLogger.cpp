#include <iostream>
#include <iomanip>
#include "EasyCNN/EasyLogger.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif //__ANDROID__ 

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif


namespace EasyCNN
{
	static std::string formatString(const char* fmt, va_list args)
	{
		std::string content;
		const int size = vsnprintf(NULL, 0, fmt, args);
		if (size > 0) {
			content.resize(size);
			vsprintf(const_cast<char*>(content.data()), fmt, args);
		}
		return content;
	}
	static std::string level2str(const LogLevel level)
	{
		switch (level)
		{
		case LogLevel::EASYCNN_LOG_LEVEL_VERBOSE:
			return "verbose";
		case LogLevel::EASYCNN_LOG_LEVEL_CRITICAL:
			return "critical";
		case LogLevel::EASYCNN_LOG_LEVEL_FATAL:
			return "fatal";
		default:
			break;
		}
		return "unknown";
	}
	static std::string buildInnerContent(const LogLevel level, const std::string& content)
	{
		std::stringstream ss;
		const auto t = time(nullptr);
		const auto local = localtime(&t);
		ss << "[" <<
			std::setw(4) << std::setfill('0') << std::setiosflags(std::ios::fixed) << local->tm_year + 1990 << "/" <<
			std::setw(2) << std::setfill('0') << std::setiosflags(std::ios::fixed) << local->tm_mon + 1 << "/" <<
			std::setw(2) << std::setfill('0') << std::setiosflags(std::ios::fixed) << local->tm_mday << " " <<
			std::setw(2) << std::setfill('0') << std::setiosflags(std::ios::fixed) << local->tm_hour << ":" << 
			std::setw(2) << std::setfill('0') << std::setiosflags(std::ios::fixed) << local->tm_min << ":" << 
			std::setw(2) << std::setfill('0') << std::setiosflags(std::ios::fixed) << local->tm_sec << "]";
		ss << "[" << level2str(level) << "] " << content << std::endl;
		return ss.str();
	}
	static void defaultLogRoute(const LogLevel level, const std::string& content)
	{
		const std::string innerContent = buildInnerContent(level, content);
#ifdef __ANDROID__
		__android_log_print(ANDROID_LOG_INFO, "digit", "log : %s", innerContent.c_str());
#else
		std::cout << innerContent;
		std::cout.flush();
#endif //__ANDROID__
	}

	//////////////////////////////////////////////////////////////////////////
	static LogLevel globalLogLevel = LogLevel::EASYCNN_LOG_LEVEL_VERBOSE;
	static std::function<void(const LogLevel, const std::string)> globalLogCb = defaultLogRoute;
	//log level setting
	void setLogLevel(const LogLevel level)
	{
		globalLogLevel = level;
	}
	LogLevel getLogLevel()
	{
		return globalLogLevel;
	}
	//log route setting
	void setLogRedirect(std::function<void(const LogLevel, const std::string)> logCb)
	{
		globalLogCb = logCb;
	}
	//log function
	void logVerbose(const char* fmt, ...)
	{
		const LogLevel level = LogLevel::EASYCNN_LOG_LEVEL_VERBOSE;
		if (globalLogLevel > level)
		{
			return;
		}
		va_list args;
		va_start(args, fmt);
		globalLogCb(level, formatString(fmt, args));
		va_end(args);
	}
	void logCritical(const char* fmt, ...)
	{
		const LogLevel level = LogLevel::EASYCNN_LOG_LEVEL_CRITICAL;
		if (globalLogLevel > level)
		{
			return;
		}
		va_list args;
		va_start(args, fmt);
		globalLogCb(level, formatString(fmt, args));
		va_end(args);
	}
	void logFatal(const char* fmt, ...)
	{
		const LogLevel level = LogLevel::EASYCNN_LOG_LEVEL_FATAL;
		if (globalLogLevel > level)
		{
			return;
		}
		va_list args;
		va_start(args, fmt);
		globalLogCb(level, formatString(fmt, args));
		va_end(args);
	}
}