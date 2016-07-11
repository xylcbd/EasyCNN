#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <ctime>
#include <sstream>
#include <cstdarg>
#include "EasyCNN/Configure.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

namespace EasyCNN
{
	namespace Anonymous
	{
		class Logger
		{
		public:
			enum class LogLevel
			{
				Verbose,
				Critical,
				Fatal
			};
			static std::string formatString(const char* fmt, ...) {
				std::string s;
				va_list ap;
				va_start(ap, fmt);
				int size = vsnprintf(NULL, 0, fmt, ap);
				va_end(ap);
				if (size > 0) {
					s.resize(size);
					va_start(ap, fmt);
					// Writes the trailing '\0' as well, but we don't care.
					vsprintf(const_cast<char*>(s.data()), fmt, ap);
					va_end(ap);
				}
				return s;
			}
			static void logging(const LogLevel level, const std::string& content)
			{
				Logger::instance().inner_logging(level,content);
			}
		private:
			Logger(bool _consoleOnly = true)
				:consoleOnly(_consoleOnly)
			{
				if (!consoleOnly){
					ofs.open("easycnn_log.txt");
				}
			}
			~Logger()
			{
				if (consoleOnly)
				{
					std::cout.flush();
				}
				else
				{
					ofs.flush();
				}
			}
			static Logger& instance()
			{
				static Logger inst;
				return inst;
			}
			std::string level2str(const LogLevel level)
			{
				switch (level)
				{
				case LogLevel::Verbose:
					return "verbose";
				case LogLevel::Critical:
					return "critical";
				case LogLevel::Fatal:
					return "fatal";
				default:
					break;
				}
				return "unknown";
			}
			std::string buildInnerContent(const LogLevel level, const std::string& content)
			{
				std::stringstream ss;
				const auto t = time(nullptr);
				const auto local = localtime(&t);
				ss << "[" << local->tm_hour << ":" << local->tm_min << ":" << local->tm_sec << "]";
				ss << "[" << level2str(level) << "] " << content << std::endl;
				return ss.str();
			}
			void inner_logging(const LogLevel level, const std::string& content)
			{
				const std::string innerContent = buildInnerContent(level, content);
				if (consoleOnly)
				{
					std::cout << innerContent;
					std::cout.flush();
				}
				else
				{
					ofs << innerContent;
					ofs.flush();
				}
			}
		private:
			std::ofstream ofs;
			bool consoleOnly = true;
		};
	}
}

#define EASYCNN_LOG_FORMATSTRING(fmt,...) EasyCNN::Anonymous::Logger::formatString(fmt,__VA_ARGS__)
#define EASYCNN_LOG_VERBOSE(fmt,...) EasyCNN::Anonymous::Logger::logging(EasyCNN::Anonymous::Logger::LogLevel::Verbose,EASYCNN_LOG_FORMATSTRING(fmt, __VA_ARGS__));
#define EASYCNN_LOG_CRITICAL(fmt,...) EasyCNN::Anonymous::Logger::logging(EasyCNN::Anonymous::Logger::LogLevel::Critical,EASYCNN_LOG_FORMATSTRING(fmt, __VA_ARGS__));
#define EASYCNN_LOG_FATAL(fmt,...) EasyCNN::Anonymous::Logger::logging(EasyCNN::Anonymous::Logger::LogLevel::Fatal,EASYCNN_LOG_FORMATSTRING(fmt, __VA_ARGS__));