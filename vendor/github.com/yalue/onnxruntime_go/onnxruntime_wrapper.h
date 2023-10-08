#ifndef ONNXRUNTIME_WRAPPER_H
#define ONNXRUNTIME_WRAPPER_H

// We want to always use the unix-like onnxruntime C APIs, even on Windows, so
// we need to undefine _WIN32 before including onnxruntime_c_api.h. However,
// this requires a careful song-and-dance.

// First, include these common headers, as they get transitively included by
// onnxruntime_c_api.h. We need to include them ourselves, first, so that the
// preprocessor will skip them while _WIN32 is undefined.
#include <stdio.h>
#include <stdlib.h>

// Next, we actually include the header.
#undef _WIN32
#include "onnxruntime_c_api.h"

// ... However, mingw will complain if _WIN32 is *not* defined! So redefine it.
#define _WIN32

#ifdef __cplusplus
extern "C" {
#endif

// Used for the OrtSessionOptionsAppendExecutionProvider_CoreML function
// pointer on supported systems. Must match the signature in
// coreml_provider_factory.h provided along with the onnxruntime releases for
// Apple platforms.
typedef OrtStatus* (*AppendCoreMLProviderFn)(OrtSessionOptions*, uint32_t);

// Takes a pointer to the api_base struct in order to obtain the OrtApi
// pointer. Intended to be called from Go. Returns nonzero on error.
int SetAPIFromBase(OrtApiBase *api_base);

// OrtSessionOptionsAppendExecutionProvider_CoreML is exported directly from
// the Apple .dylib, so we call this function on Apple platforms to set the
// function pointer to the correct address. On other platforms, the function
// pointer should remain NULL.
void SetCoreMLProviderFunctionPointer(void *ptr);

// Wraps ort_api->ReleaseStatus(status)
void ReleaseOrtStatus(OrtStatus *status);

// Wraps calling ort_api->CreateEnv. Returns a non-NULL status on error.
OrtStatus *CreateOrtEnv(char *name, OrtEnv **env);

// Wraps ort_api->DisableTelemetryEvents. Returns a non-NULL status on error.
OrtStatus *DisableTelemetry(OrtEnv *env);

// Wraps ort_api->EnableTelemetryEvents. Returns a non-NULL status on error.
OrtStatus *EnableTelemetry(OrtEnv *env);

// Wraps ort_api->ReleaseEnv
void ReleaseOrtEnv(OrtEnv *env);

// Wraps ort_api->CreateCpuMemoryInfo with some basic, default settings.
OrtStatus *CreateOrtMemoryInfo(OrtMemoryInfo **mem_info);

// Wraps ort_api->ReleaseMemoryInfo
void ReleaseOrtMemoryInfo(OrtMemoryInfo *info);

// Returns the message associated with the given ORT status.
const char *GetErrorMessage(OrtStatus *status);

// Wraps ort_api->CreateSessionOptions
OrtStatus *CreateSessionOptions(OrtSessionOptions **o);

// Wraps ort_api->ReleaseSessionOptions
void ReleaseSessionOptions(OrtSessionOptions *o);

// Wraps ort_api->SetIntraOpNumThreads
OrtStatus *SetIntraOpNumThreads(OrtSessionOptions *o, int n);

// Wraps ort_api->SetInterOpNumThreads
OrtStatus *SetInterOpNumThreads(OrtSessionOptions *o, int n);

// Wraps ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2
OrtStatus *AppendExecutionProviderCUDAV2(OrtSessionOptions *o,
  OrtCUDAProviderOptionsV2 *cuda_options);

// Wraps ort_api->CreateCUDAProviderOptions
OrtStatus *CreateCUDAProviderOptions(OrtCUDAProviderOptionsV2 **o);

// Wraps ort_api->ReleaseCUDAProviderOptions
void ReleaseCUDAProviderOptions(OrtCUDAProviderOptionsV2 *o);

// Wraps ort_api->UpdateCUDAProviderOptions
OrtStatus *UpdateCUDAProviderOptions(OrtCUDAProviderOptionsV2 *o,
  const char **keys, const char **values, int num_keys);

// Wraps ort_api->CreateTensorRTProviderOptions
OrtStatus *CreateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 **o);

// Wraps ort_api->ReleaseTensorRTProviderOptions
void ReleaseTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *o);

// Wraps ort_api->UpdateTensorRTProviderOptions
OrtStatus *UpdateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *o,
  const char **keys, const char **values, int num_keys);

// Wraps ort_api->SessionOptionsAppendExecutionProvider_TensorRT_V2
OrtStatus *AppendExecutionProviderTensorRTV2(OrtSessionOptions *o,
  OrtTensorRTProviderOptionsV2 *tensor_rt_options);

// Wraps OrtSessionOptionsAppendExecutionProvider_CoreML, exported from the
// dylib on Apple devices. Safely returns a non-NULL status on other platforms.
OrtStatus *AppendExecutionProviderCoreML(OrtSessionOptions *o,
  uint32_t flags);

// Creates an ORT session using the given model. The given options pointer may
// be NULL; if it is, then we'll use default options.
OrtStatus *CreateSession(void *model_data, size_t model_data_length,
  OrtEnv *env, OrtSession **out, OrtSessionOptions *options);

// Runs an ORT session with the given input and output tensors, along with
// their names. In our use case, outputs must NOT be NULL.
OrtStatus *RunOrtSession(OrtSession *session,
  OrtValue **inputs, char **input_names, int input_count,
  OrtValue **outputs, char **output_names, int output_count);

// Wraps ort_api->ReleaseSession
void ReleaseOrtSession(OrtSession *session);

// Used to free OrtValue instances, such as tensors.
void ReleaseOrtValue(OrtValue *value);

// Creates an OrtValue tensor with the given shape, and backed by the user-
// supplied data buffer.
OrtStatus *CreateOrtTensorWithShape(void *data, size_t data_size,
  int64_t *shape, int64_t shape_size, OrtMemoryInfo *mem_info,
  ONNXTensorElementDataType dtype, OrtValue **out);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // ONNXRUNTIME_WRAPPER_H
