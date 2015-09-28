#ifndef __ffttest_hh
#define __ffttest_hh

#include <condition_variable>
#include <mutex>
#include <chrono>

#include "fft.hh"

class FftTest : public FftCallback {

public:
    typedef std::chrono::high_resolution_clock::time_point time_pt;

public:
    FftTest();

public:
    bool init(size_t                fft_size, 
              Fft::Device           device_type,
              bool                  use_out_of_order,
              int                   queue_count,
              long                  count, 
              int                   parallel, 
              FftBuffer::TestData   test_data, 
              double                mean, 
              double                std);
    void test();
    void release();
    
    void done();
    void print_results(time_pt start, time_pt finish);

public:
    virtual void fft_complete(FftBuffer* job);
    
private:
    Fft*                        _fft;
    
    FftBuffer::TestData         _test_data; 
    double                      _mean; 
    double                      _std;
    bool                        _save_data;
    
    int                         _count;
    int                         _total;
    
    std::mutex                  _mut;
    std::condition_variable     _cond;
};

#endif // __ffttest_hh