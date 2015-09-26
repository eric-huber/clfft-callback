#include "ffttest.hh"

#include <iostream>

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

FftTest::FftTest() {	
}

bool FftTest::init(size_t               fft_size, 
                   Fft::Device          device, 
                   long                 count, 
                   int                  parallel, 
                   FftBuffer::TestData  test_data, 
                   double               mean, 
                   double               std) {

    _test_data = test_data;
    _mean = mean;
    _std = std;
    
    _fft = new Fft(fft_size, device, count, parallel);
    return _fft->init();
}

void FftTest::test() {
    
    _fft->register_callback(this);
    
    FftBuffer* buffer = _fft->get_buffer();
    buffer->populate(_test_data, _mean, _std);
    buffer->write(_data_file_name);
    
    _fft->forward(buffer);
}

void FftTest::release() {
    _fft->release();
}

void FftTest::fft_complete(FftBuffer* buffer) {
	
	buffer->write_hermitian(_fft_file_name);
}