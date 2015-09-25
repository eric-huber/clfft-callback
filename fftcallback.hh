#ifndef __fftcallback_hh
#define __fftcalback_hh

class FftJob;

class FftCallback {
public:
    virtual void fft_complete(FftJob* job) = 0;
};

#endif //__fftcallback_hh