#include "ffttest.hh"

#include <iostream>
#include <iomanip>

const char* _data_file_name = "fft-data.txt";
const char* _fft_file_name  = "fft-forward.txt";
const char* _bak_file_name  = "fft-backward.txt";

FftTest::FftTest()
 : _save_data(false),
    _total(0)
{	
}

std::atomic_bool ready;

bool FftTest::init(size_t               fft_size, 
                   Fft::Device          device,
                   bool                 use_out_of_order,
                   int                  queue_count,
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
    
    _fft = new Fft(fft_size, device, use_out_of_order, queue_count, parallel);
    return _fft->init();
}

void FftTest::test() {

    std::unique_lock<std::mutex> lock(_mut);

    _fft->register_callback(this);

    // start initial transforms    
    for (int i = 0; i < _fft->get_parallel(); ++i) {
        FftBuffer* buffer = _fft->get_buffer();
        buffer->populate(_test_data, _mean, _std);
        _buffers.push_back(buffer);
    
        _fft->forward(buffer);
    }
    
    while (_total < _count) {
        _cond.wait(lock, []{ return ready.load(); });
    
        FftBuffer* buffer = get_complete_buffer();
        if (NULL == buffer)
            continue;

        buffer->set_complete(false);
        switch (buffer->contains()) {
        case FftBuffer::FFT:
            _fft->backward(buffer);
            break;
     
        case FftBuffer::IFFT:
            if (_save_data)
                buffer->write(_bak_file_name);
            
            ++_total;
            if (_total < _count) {
                //buffer->populate(_test_data, _mean, _std);
                _fft->forward(buffer);
            }
            break;
        }
    }
    
    print_results();
}

void FftTest::release() {
    _fft->release();
}

FftBuffer* FftTest::get_complete_buffer() {
    for (auto buffer : _buffers) {
        if (buffer->is_complete())
            return buffer;
    }
    return NULL;
}

void FftTest::print_results() {
    
    double total_dur = 0;
    for (auto buffer : _buffers) {
        total_dur += buffer->ave_time();
    }
    double ave_dur = total_dur / (double) _buffers.size();
    
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
    for (int i = 0; i < _buffers.size(); ++i) {
        std::cout << "Buffer " << std::setw(2) << i;
        std::cout << " time: " << std::setw(12) << _buffers[i]->total_time();
        std::cout << " ns, count: " << std::setw(4) << _buffers[i]->transforms();
        std::cout << ", ave: " << _buffers[i]->ave_time();
        std::cout << " ns (" << (_buffers[i]->ave_time() / 1000.0) << " μs)";
        std::cout << std::endl;    
    }
    std::cout << std::endl;
    std::cout << "Time:       " << total_dur << " ns" << std::endl;
    std::cout << "Average:    " << ave_dur << " ns (" << (ave_dur / 1000.0) << " μs)" << std::endl;
}

void FftTest::fft_complete(FftBuffer* buffer) {

    // mark this data set as done
    buffer->set_complete(true);
    
    // wake the main thread
    ready = true;
    _cond.notify_one();
}