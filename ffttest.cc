#include "ffttest.hh"

FftTest::FftTest() {	
}

void FftTest::init(size_t         fft_size, 
                   Fft::Device    device, 
                   long           count, 
                   int            parallel, 
                   Fft::TestData  test_data, 
                   double         mean, 
                   double         std)
{    
    _fft = new Fft(fft_size, device, count, parallel, test_data, mean, std);
}

void FftTest::test() {
    
    _fft->register_callback(this);    
}

void FftTest::release() {
    _fft->release();
}

void FftTest::fft_complete(FftJob* job) {
    
}