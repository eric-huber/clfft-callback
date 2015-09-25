#ifndef __ffttest_hh
#define __ffttest_hh

#include "fft.hh"

class FftTest : public FftCallback {
public:
    FftTest();

public:
    void init(size_t          fft_size, 
              Fft::Device     device_type, 
              long            count, 
               int            parallel, 
              Fft::TestData   test_data, 
              double          mean, 
              double          std);
    void test();
    void release();     

public:
    virtual void fft_complete(FftJob* job);
    
private:
    Fft*    _fft;
};

#endif // __ffttest_hh