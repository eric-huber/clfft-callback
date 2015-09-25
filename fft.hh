#ifndef __fft_hh
#define __fft_hh

#include <clFFT.h>
#include <vector>

#include "fftcallback.hh"

class Fft {
    
public:
    enum Device    {GPU, CPU};
	enum TestData  {PERIODIC, RANDOM};

public:
    Fft(size_t       fft_size, 
        Device       device_type, 
        long         count, 
        int          parallel, 
        TestData     test_data, 
        double       mean, 
        double       std);

	virtual ~Fft();

    bool init();
    void release();

    // we can only accept one callback at the moment
    void register_callback(FftCallback* callback);

    bool forward(FftJob* job);
    bool backward(FftJob* job);

private:
    bool select_platform();
    bool setup_cl();
    bool setup_clFft();
    bool setup_forward();
    bool setup_backward();
    bool setup_buffers();

private:
    size_t                  _fft_size;
    Device                  _device_type; 
    long                    _count;
    int                     _parallel;
    TestData                _test_data; 
    double                  _mean;
    double                  _std;
    
    cl_platform_id          _platform;
    cl_device_id            _device;
    cl_context              _context;
    cl_command_queue        _queue;
    clfftPlanHandle         _forward;
    clfftPlanHandle         _backward;
};

#endif // __fft_hh