#include <clFFT.h>
#include <string>

#include "fft.hh"

class FftJob {
  
public:
    FftJob(size_t fft_size);
    ~FftJob();
    
public:    
    void        populate(Fft::TestData data_type, double mean, double std);
    void        scale(double factor);

    void        copy(FftJob& other);
    
    double      rms(FftJob& inverse);    
    double      signal_to_quant_error(FftJob& inverse);

    void        write(std::string file);
    void        write_hermitian(std::string file);

    void        release();
  
    cl_float*   data()              { return _data; }

    cl_float    at(int index)       { return _data[index]; }
    cl_float    at_hr(int index)    { return _data[2 * index]; }
    cl_float    at_hi(int index)    { return _data[2 * index + 1]; }
    
    int         size()              { return _size; }
    int         size_h()            { return _size / 2; }

private:
    void        randomize(double mean, double std);
    void        periodic();
    
    double      signal_energy();
    double      quant_error_energy(FftJob& inverse);  

private:
    size_t      _size;
    cl_float*   _data;
};