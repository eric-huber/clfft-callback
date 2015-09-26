#include "fftbuffer.hh"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <math.h>

FftBuffer::FftBuffer(Fft* fft, size_t size, cl_mem local) 
 : _fft(fft),
   _size(size),
   _local(local)
{
    _data  = new cl_float[_size];
}


FftBuffer::~FftBuffer() {
    release();
}

void FftBuffer::copy(FftBuffer& other) {
    for (int i = 0; i < _size; ++i) {
        _data[i] = other._data[i];
    }    
}

double FftBuffer::rms(FftBuffer& inverse) {
    
    double rms = 0;
    
    for (int i = 0; i < _size; ++i) {
        rms += pow(inverse.at(i) - at(i), 2);
    }
    rms /= size();
    rms = sqrt(rms);
    return rms;
}

double FftBuffer::signal_to_quant_error(FftBuffer& inverse) {
    
    return 10.0 * log10( signal_energy() / quant_error_energy(inverse) );
}

double FftBuffer::signal_energy() {
    double energy = 0;
    for (int i = 0; i < _size; ++i) {
        energy += pow(at(i), 2);
    }
    return energy;
}

double FftBuffer::quant_error_energy(FftBuffer& inverse) {
    
    double energy = 0;
    for (int i = 0; i < _size; ++i) {
        energy += pow(at(i) - inverse.at(i), 2);
    }
    return energy;
}

void FftBuffer::populate(TestData data_type, double mean, double std) {
    switch (data_type) {
    case PERIODIC:
    default:
        periodic();
        break;
    case RANDOM:
        randomize(mean, std);
        break;
    }
}

void FftBuffer::randomize(double mean, double std) {
    
    std::default_random_engine       generator(std::random_device{}());
    std::normal_distribution<double> distribution(mean, std);
    
    srand(time(NULL));

    for(int i = 0; i < _size; i++) {
        float number = distribution(generator);
        _data[i]  = number;
    }
}

void FftBuffer::periodic() {
    for (int i = 0; i < _size; ++i) {
        float t = i * .002;
        float amp = sin(M_PI * t);
        amp += sin(2 * M_PI * t);
        amp += sin(3 * M_PI * t); 
        _data[i] = amp;
    }
}

void FftBuffer::scale(double factor) {
    for (int i = 0; i < _size; ++i) {
        _data[i] *= factor;
    }
}

void FftBuffer::write(std::string filename) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 0; i < _size; ++i) {
        ofs << _data[i] << std::endl;
    }
    
    ofs.close();   
}

void FftBuffer::write_hermitian(std::string filename) {
    std::ofstream ofs;
    ofs.open(filename);
    
    for (int i = 1; i < _size / 2; ++i) {
        auto real = at_hr(i);
        auto imag = at_hi(i);
        auto amplitude = sqrt(pow(real, 2) + pow(imag, 2));
        auto phase = atan2(imag, real);
        ofs << amplitude << ", " << phase << std::endl;
    }
    
    ofs.close();   
}

void FftBuffer::release() {
    if (NULL != _data) {
        delete[] _data;
        _data = NULL;
    }
}