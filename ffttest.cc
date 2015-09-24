#include "ffttest.hh"

FftTest::FftTest(size_t         fft_size, 
                 Fft::Device    device, 
                 long           count, 
                 int            parallel, 
                 Fft::TestData  test_data, 
                 double         mean, 
                 double         std)
 : _fft(fft_size, device, count, parallel, test_data, mean, std),
   _is_init(false)
{	
}

void FftTest::init() {
    _is_init = _fft.init();
}

void FftTest::test() {
    if (!_is_init)
        return;
        
    _fft.register_callback(this);
}

void FftTest::release() {
    _fft.release();
}

void FftTest::fft_complete() {
    
}