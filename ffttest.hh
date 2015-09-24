#ifndef __ffttest_hh
#define __ffttest_hh

#include "fft.hh"

class FftTest : public FftCallback {
public:
    FftTest(size_t          fft_size, 
            Fft::Device     device_type, 
            long            count, 
            int             parallel, 
            Fft::TestData   test_data, 
            double          mean, 
            double          std);

public:
    void init();
    void test();
    void release();     

public:
    virtual void fft_complete();
    
private:
    Fft     _fft;
    bool    _is_init;
};

#endif // __ffttest_hh