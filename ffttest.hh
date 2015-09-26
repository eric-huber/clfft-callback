#ifndef __ffttest_hh
#define __ffttest_hh

#include "fft.hh"

class FftTest : public FftCallback {
public:
    FftTest();

public:
    bool init(size_t                fft_size, 
              Fft::Device           device_type, 
              long                  count, 
              int                   parallel, 
              FftBuffer::TestData   test_data, 
              double                mean, 
              double                std);
    void test();
    void release();     

public:
    virtual void fft_complete(FftBuffer* job);
    
private:
    Fft*                _fft;
    
    FftBuffer::TestData _test_data; 
    double              _mean; 
    double              _std;
};

#endif // __ffttest_hh