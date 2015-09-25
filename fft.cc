#include <iostream>
#include <cstring>

#include "fft.hh"

#define CHECK(MSG)                              \
    if (err != CL_SUCCESS) {                    \
      std::cerr << __FILE__ << ":" << __LINE__  \
          << " Unexpected result for " << MSG   \
          << " (" << err << ")" << std::endl;   \
      return false;                             \
    }
    
FftCallback*    callback;

void CL_CALLBACK event_callback(cl_event event, cl_int status, void* user_data) {
        
    if (CL_COMPLETE != status) {
        return;
    }
    
    FftJob* job = (FftJob*) user_data;   
    callback->fft_complete(job);    
}

Fft::Fft(size_t   fft_size, 
               Device   device, 
               long     count, 
               int      parallel, 
               TestData test_data, 
               double   mean, 
               double   std) 
 :  _fft_size       (fft_size),
    _device_type    (device),
    _count          (count),
    _parallel       (parallel),
    _test_data      (test_data),
    _mean           (mean),
    _std            (std)
 {
 }

Fft::~Fft() {
   release();
}

bool Fft::init() {
   if (select_platform() 
       && setup_cl()
       && setup_clFft() 
       && setup_forward()
       && setup_backward())
        return true;
    return false;
}

void Fft::release() {
   
   if (NULL == _context)
      return;
   
    // Release clFFT library. 
    clfftTeardown();
    
    // Release OpenCL working objects. 
    clReleaseCommandQueue(_queue);
    _queue = NULL;
    
    clReleaseContext(_context);
    _context = NULL;
}

void Fft::register_callback(FftCallback* cb) {
    callback = callback;
}


bool Fft::forward(FftJob* job) {
    
    cl_int err = 0;
   
    cl_event write = 0;
    cl_event read = 0;
    cl_event transform = 0;

    // get buffer
    
/*
    // Enqueue write tab array into _local_buffers[0]
    err = clEnqueueWriteBuffer(_queue, buffer->data(), CL_FALSE, 0, 
                                buffer->size(), job->data(), 0, NULL, &write);
    CHECK("clEnqueueWriteBuffer");

    // Enqueue the FFT
    err = clfftEnqueueTransform(_forward, CLFFT_FORWARD, 1, &_queue, 1, &write, &transform,
                                 buffer->data_addr(), NULL, buffer->temp());
    CHECK("clEnqueueTransform");

    // Copy result to input array
    err = clEnqueueReadBuffer(_queue, buffer->data(), CL_FALSE, 0,
                               buffer->size(), job->data(), 1, &transform, &read);
    CHECK("clEnqueueReadBuffer");

    clSetEventCallback(read, CL_COMPLETE, &event_callback, buffer);
*/
    return true;
}

bool Fft::backward(FftJob* job) {
    cl_int err = 0;
   
    cl_event write = 0;
    cl_event read = 0;
    cl_event transform = 0;

    // get buffer
    
/*
    // Enqueue write tab array into _local_buffers[0]
    err = clEnqueueWriteBuffer(_queue, buffer->data(), CL_FALSE, 0, 
                                buffer->size(), job->data(), 0, NULL, &write);
    CHECK("clEnqueueWriteBuffer");

    // Enqueue the FFT
    err = clfftEnqueueTransform(_backward, CLFFT_BACKWARD, 1, &_queue, 1, &write, &transform,
                                 buffer->data_addr(), NULL, buffer->temp());
    CHECK("clEnqueueTransform");

    // Copy result to input array
    err = clEnqueueReadBuffer(_queue, buffer->data(), CL_FALSE, 0,
                               buffer->size(), job->data(), 1, &transform, &read);
    CHECK("clEnqueueReadBuffer");

    clSetEventCallback(read, CL_COMPLETE, &event_callback, buffer);
*/
    return true;

}

bool Fft::select_platform() {
    cl_int          err = 0;
    cl_uint         platform_count = 0;
    cl_platform_id  platform[5];
    cl_device_type  type;
    
    switch (_device_type) {
    default:
    case GPU:
        type = CL_DEVICE_TYPE_GPU;
        break;
    case CPU:
        type = CL_DEVICE_TYPE_CPU;
        break;
    }

    // get list of platforms
    err = clGetPlatformIDs(0, NULL, &platform_count);
    CHECK("clGetPlatformIds - platform count");
    
    err = clGetPlatformIDs(5, platform, NULL);
    CHECK("clGetPlatformIds - list of platforms");
    
    // find a platform supporting our device type
    for (uint i = 0; i < platform_count; ++i) {
        err = clGetDeviceIDs(platform[i], type, 1, &_device, NULL);
        if (err == CL_SUCCESS) {
            _platform = platform[i];
            return true;
        }
    }
    
    return false;
}

bool Fft::setup_cl() {
    cl_int err = 0;

    // Setup context
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) _platform, 0};
    _context = clCreateContext(props, 1, &_device, NULL, NULL, &err);
    CHECK("clCreateContext");

    // Setup queues
    _queue = clCreateCommandQueue(_context, _device, 0 /* IN-ORDER */, &err);
    CHECK("clCreateCommandQueue CPU");

    return true;
}

bool Fft::setup_clFft() {
    cl_int err = 0;

    // Setup clFFT. 
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    CHECK("clfftInitSetupData");
    err = clfftSetup(&fftSetup);
    CHECK("clfftSetup");
    
    return true;    
}

bool Fft::setup_forward() {
    cl_int err = 0;
    
    // Size of FFT 
    size_t clLengths = _fft_size;
    clfftDim dim = CLFFT_1D;
    
    // Create a default plan for a complex FFT 
    err = clfftCreateDefaultPlan(&_forward, _context, dim, &clLengths);
    CHECK("clfftCreateDefaultPlan");

    // Set plan parameters
    err = clfftSetPlanPrecision(_forward, CLFFT_SINGLE);
    CHECK("clfftSetPlanPrecision");
    err = clfftSetLayout(_forward, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    CHECK("clfftSetLayout");
    err = clfftSetResultLocation(_forward, CLFFT_INPLACE);
    CHECK("clfftSetResultLocation");

    // Bake the plan
    err = clfftBakePlan(_forward, 1, &_queue, NULL, NULL);
    CHECK("clfftBakePlan");

    return true;
}

bool Fft::setup_backward() {
    cl_int err = 0;

    // Size of FFT
    size_t clLengths = _fft_size;
    clfftDim dim = CLFFT_1D;
    
    // Create a default plan for a complex FFT 
    err = clfftCreateDefaultPlan(&_backward, _context, dim, &clLengths);
    CHECK("clfftCreateDefaultPlan");

    // Set plan parameters
    err = clfftSetPlanPrecision(_backward, CLFFT_SINGLE);
    CHECK("clfftSetPlanPrecision");
    err = clfftSetLayout(_backward, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
    CHECK("clfftSetLayout");
    err = clfftSetResultLocation(_backward, CLFFT_INPLACE);
    CHECK("clfftSetResultLocation");

    // Bake the plan
    err = clfftBakePlan(_backward, 1, &_queue, NULL, NULL);
    CHECK("clfftBakePlan");

    return true;
}
