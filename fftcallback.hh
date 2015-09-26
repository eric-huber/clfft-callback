#ifndef __fftcallback_hh
#define __fftcalback_hh

class FftBuffer;

class FftCallback {
public:
    virtual void fft_complete(FftBuffer* job) = 0;
};

#endif //__fftcallback_hh