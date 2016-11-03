LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := easycnn

LOCAL_SRC_FILES := \
	$(LOCAL_PATH)/../../src/ActivationLayer.cpp \
	$(LOCAL_PATH)/../../src/ConvolutionLayer.cpp \
	$(LOCAL_PATH)/../../src/DataBucket.cpp \
	$(LOCAL_PATH)/../../src/EasyAssert.cpp \
	$(LOCAL_PATH)/../../src/EasyLogger.cpp \
	$(LOCAL_PATH)/../../src/FullconnectLayer.cpp \
	$(LOCAL_PATH)/../../src/InputLayer.cpp \
	$(LOCAL_PATH)/../../src/LossFunction.cpp \
	$(LOCAL_PATH)/../../src/NetWork.cpp \
	$(LOCAL_PATH)/../../src/ParamBucket.cpp \
	$(LOCAL_PATH)/../../src/PoolingLayer.cpp \
	$(LOCAL_PATH)/../../src/SoftmaxLayer.cpp
	
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../header
LOCAL_CFLAGS :=  -D__ARM_NEON -D__cpusplus -O3 -mfloat-abi=softfp -mfpu=neon -march=armv7-a -mtune=cortex-a8 -fopenmp -std=c++11 -ffunction-sections -fdata-sections -fvisibility=hidden

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)    
LOCAL_ARM_NEON := true   
endif

include $(BUILD_STATIC_LIBRARY)