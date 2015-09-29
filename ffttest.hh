#ifndef __ffttest_hh
#define __ffttest_hh

#include <condition_variable>
#include <mutex>

#include "fft.hh"

class FftTest : public FftCallback {

public:
    FftTest();

public:
    bool        init(size_t                fft_size, 
                     Fft::Device           device_type,
                     bool                  use_out_of_order,
                     int                   queue_count,
                     long                  count, 
                     int                   parallel, 
                     FftBuffer::TestData   test_data, 
                     double                mean, 
                     double                std);
    
    void        test();
    void        release();
    
    FftBuffer*  get_complete_buffer();
    void        print_results();

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
    
    std::vector<FftBuffer*>     _buffers;
    
    std::mutex                  _mut;
    std::condition_variable     _cond;
};

#endif // __ffttest_hh