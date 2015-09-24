#include <boost/program_options.hpp>
#include <clFFT.h>

#include <thread>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>
 
#include "ffttest.hh"

namespace po = boost::program_options;

int main(int ac, char* av[]) {

    size_t              fft_size        = 8192;
    Fft::Device         device          = Fft::GPU;
    Fft::TestData       test_data       = Fft::RANDOM;
    int                 parallel        = 16;
    long                count           = 1000;
    double              mean            = 0.5;
    double              std             = 0.2;

    try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",         "Produce help message")
        ("cpu,c",          "Force CPU usage")
        
        ("size,s",         po::value<int>(), "Set the size of the buffer [8192]")
       
        ("periodic,p",     "Use a periodic data set")
        ("random,r",       "Use a gaussian distributed random data set")
        ("mean,m",         po::value<double>(), "Mean for random data")
        ("deviation,d",    po::value<double>(), "Standard deviation for random data")
        
        ("jobs,j",         po::value<int>(), "Jobs to perform in parallel")
        ("loops,l",        po::value<long>(), "Set the number of iterations to perform");        

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }
        
        if (vm.count("cpu")) {
            device = Fft::CPU;
        }
                
        if (vm.count("periodic")) {
        	test_data = Fft::PERIODIC;
        }
        
        if (vm.count("random")) {
        	test_data = Fft::RANDOM;
        }
        
        if (vm.count("mean")) {
            mean = vm["mean"].as<double>();
        }

        if (vm.count("deviation")) {
            std = vm["deviation"].as<double>();
        }
        
        if (vm.count("jobs")) {
            parallel = vm["jobs"].as<int>();
        }

        if (vm.count("loops")) {
            count = vm["loops"].as<long>();
        }

        if (vm.count("size")) {
            fft_size = vm["size"].as<int>();
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return 1;
    }

    // to nearest 16
    count = ((int) ceil(count / parallel)) * parallel;

    // test it
    FftTest test(fft_size, device, count, parallel, test_data, mean, std);
    test.init();
    test.test();
    test.release();
    
    return 0;
}
