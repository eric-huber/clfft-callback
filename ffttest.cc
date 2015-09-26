#include "ffttest.hh"

#include <iostream>

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

FftTest::FftTest()
 : _save_data(false),
    _total(0)
{	
}

bool ready = false;

bool FftTest::init(size_t               fft_size, 
                   Fft::Device          device, 
                   long                 count, 
                   int                  parallel, 
                   FftBuffer::TestData  test_data, 
                   double               mean, 
                   double               std) {

    _count = count;
    _test_data = test_data;
    _mean = mean;
    _std = std;
    ready = false;
    
    _fft = new Fft(fft_size, device, parallel);
    return _fft->init();
}

void FftTest::test() {

    std::unique_lock<std::mutex> lock(_mut);

    _fft->register_callback(this);

    time_pt start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < _fft->get_parallel(); ++i) {
        FftBuffer* buffer = _fft->get_buffer();
        buffer->populate(_test_data, _mean, _std);
    
        _fft->forward(buffer);
    }
    
    _cond.wait(lock, []{ return ready; });
    
    time_pt end = std::chrono::high_resolution_clock::now();
    
    print_results(start, end);    
}

void FftTest::release() {
    _fft->release();
}

void FftTest::done() {
    
    ready = true;
    _cond.notify_one();
}

void FftTest::print_results(time_pt start, time_pt finish) {
    
    std::chrono::nanoseconds duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);
    float count   = _count * 2;
    float ave_dur = duration.count() / count;

    std::cout << "Device:     " << ( Fft::GPU == _fft->get_device() ? "GPU" : "CPU") << std::endl;
    std::cout << "Parallel:   " << _fft->get_parallel() << std::endl;
    std::cout << "Iterations: " << _count << std::endl;
    std::cout << "Data size:  " << _fft->get_size() << std::endl;
    std::cout << "Data type:  " << (_test_data == FftBuffer::PERIODIC ? "Periodic" : "Random") << std::endl;
    if (FftBuffer::PERIODIC == _test_data) {
        std::cout << "Mean:       " << _mean << std::endl;
        std::cout << "Std Dev:    " << _std << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Time:       " << duration.count() << " ns" << std::endl;
    std::cout << "Average:    " << ave_dur << " ns (" << (ave_dur / 1000.0) << " μs)" << std::endl;
}

void FftTest::fft_complete(FftBuffer* buffer) {

    switch (buffer->contains()) {
    case FftBuffer::FFT:
        _fft->backward(buffer);
        break;
 
    case FftBuffer::IFFT:
        if (_save_data)
            buffer->write(_bak_file_name);
        
        ++_total;
        if (_total < _count) {
            buffer->populate(_test_data, _mean, _std);
            _fft->forward(buffer);
        } else {
            done();
        }
        break;
    }
}