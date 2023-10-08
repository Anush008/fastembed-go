//go:build windows

package onnxruntime_go

// This file includes the Windows-specific code for loading the onnxruntime
// library and setting up the environment.

import (
	"fmt"
	"syscall"
	"unsafe"
)

// #include "onnxruntime_wrapper.h"
import "C"

// This will contain the handle to the onnxruntime dll if it has been loaded
// successfully.
var libraryHandle syscall.Handle

func platformCleanup() error {
	e := syscall.FreeLibrary(libraryHandle)
	libraryHandle = 0
	return e
}

func platformInitializeEnvironment() error {
	if onnxSharedLibraryPath == "" {
		onnxSharedLibraryPath = "onnxruntime.dll"
	}
	handle, e := syscall.LoadLibrary(onnxSharedLibraryPath)
	if e != nil {
		return fmt.Errorf("Error loading ONNX shared library \"%s\": %w",
			onnxSharedLibraryPath, e)
	}
	getApiBaseProc, e := syscall.GetProcAddress(handle, "OrtGetApiBase")
	if e != nil {
		syscall.FreeLibrary(handle)
		return fmt.Errorf("Error finding OrtGetApiBase function in %s: %w",
			onnxSharedLibraryPath, e)
	}
	ortApiBase, _, e := syscall.SyscallN(uintptr(getApiBaseProc), 0)
	if ortApiBase == 0 {
		syscall.FreeLibrary(handle)
		if e != nil {
			return fmt.Errorf("Error calling OrtGetApiBase: %w", e)
		} else {
			return fmt.Errorf("Error calling OrtGetApiBase")
		}
	}
	tmp := C.SetAPIFromBase((*C.OrtApiBase)(unsafe.Pointer(ortApiBase)))
	if tmp != 0 {
		syscall.FreeLibrary(handle)
		return fmt.Errorf("Error setting ORT API base: %d", tmp)
	}
	libraryHandle = handle
	return nil
}
