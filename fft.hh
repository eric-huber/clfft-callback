#ifndef __fft_hh
#define __fft_hh

#include <clFFT.h>
#include <vector>

#include "fftcallback.hh"
#include "fftbuffer.hh"

class Fft {
public:
    enum Device    {GPU, CPU};

public:
    Fft(size_t       fft_size, 
        Device       device_type,
        bool         use_out_of_order,
        int          queue_count,
        int          parallel);

	virtual ~Fft();
	
	size_t       get_size()     { return _fft_size; }
	Device       get_device()   { return _device_type; }
	int          get_parallel() { return _parallel; }

    bool         init();
    void         release();

    // we can only accept one callback at the moment
    void         register_callback(FftCallback* callback) { _callback = callback; }
    FftCallback* get_callback()                           { return _callback; }

    FftBuffer*   get_buffer();

    bool         forward(FftBuffer* job);
    bool         backward(FftBuffer* job);

private:
    bool select_platform();
    bool setup_cl();
    bool setup_clFft();
    bool setup_forward();
    bool setup_backward();
    bool setup_buffers();
    
    size_t  buffer_size();

private:
    size_t                         _fft_size;
    Device                         _device_type;
    bool                           _use_out_of_order;
    int                            _queue_count;
    int                            _parallel;
    
    cl_platform_id                 _platform;
    cl_device_id                   _device;
    cl_context                     _context;
    std::vector<cl_command_queue>  _queues;
    std::vector<clfftPlanHandle>   _forward;
    std::vector<clfftPlanHandle>   _backward;
    std::vector<FftBuffer*>        _buffers;
    
    FftCallback*                   _callback;
};

#endif // __fft_hh