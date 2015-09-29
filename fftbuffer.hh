#ifndef __fftbuffer_hh
#define __fftbuffer_hh

#include <atomic>
#include <chrono>
#include <clFFT.h>
#include <string>

typedef std::chrono::high_resolution_clock::time_point time_pt;

class Fft;

class FftBuffer {
public: 
    enum TestData  {PERIODIC, RANDOM};
    enum Contains  {DATA, FFT, IFFT, DONE};

public:
    FftBuffer(Fft* fft, size_t size, cl_mem local);
    ~FftBuffer();
    
public:    
    void        populate(TestData data_type, double mean, double std);
    void        scale(double factor);

    void        copy(FftBuffer& other);
    
    double      rms(FftBuffer& inverse);    
    double      signal_to_quant_error(FftBuffer& inverse);

    void        write(std::string file);
    void        write_hermitian(std::string file);

    void        release();
  
    cl_float*   data()              { return _data; }

    cl_float    at(int index)       { return _data[index]; }
    cl_float    at_hr(int index)    { return _data[2 * index]; }
    cl_float    at_hi(int index)    { return _data[2 * index + 1]; }
    
    int         size()              { return _size; }
    int         size_h()            { return _size / 2; }

    cl_mem      local()             { return _local; }
    cl_mem*     local_addr()        { return &_local;}
    
    Fft*        fft()               { return _fft; }
    void        queue(int q)        { _queue = q; }
    int         queue()             { return _queue; }    
    
    void        start_timer();
    void        end_timer();
    long        total_time()                    { return _duration.count(); }
    long        transforms()                    { return _transforms; }
    double      ave_time();
    
    void        contains(Contains contains)     { _contains = contains; }
    Contains    contains()                      { return _contains; }
    
    void        set_complete(bool is_complete)  { _is_complete = is_complete; }
    bool        is_complete()                   { return _is_complete; }
    
private:
    void        randomize(double mean, double std);
    void        periodic();
    
    double      signal_energy();
    double      quant_error_energy(FftBuffer& inverse);  

private:
    Fft*        _fft;
    int         _queue;
    size_t      _size;
    cl_mem      _local;
    cl_float*   _data;
    
    time_pt                     _start;
    std::chrono::nanoseconds    _duration;
    long                        _transforms;
    
    Contains                    _contains;
    std::atomic_bool            _is_complete;
};

#endif // __fftbuffer_hh