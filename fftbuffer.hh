#ifndef __fftbuffer_hh
#define __fftbuffer_hh

#include <clFFT.h>
#include <string>

class Fft;

class FftBuffer {
public: 
    enum TestData  {PERIODIC, RANDOM};

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

private:
    void        randomize(double mean, double std);
    void        periodic();
    
    double      signal_energy();
    double      quant_error_energy(FftBuffer& inverse);  

private:
    Fft*        _fft;
    size_t      _size;
    cl_mem      _local;
    cl_float*   _data;
};

#endif // __fftbuffer_hh