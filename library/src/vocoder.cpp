
#include <string>
#include "cxxopts.hpp"

#include "wavernn.h"

using namespace std;

int main(int argc, char* argv[])
{

    cxxopts::Options options("vocoder", "WaveRNN based vocoder");
    options.add_options()
            ("w,weights", "File with network weights", cxxopts::value<string>())
            ("m,mel", "File with mel inputs", cxxopts::value<string>())
            ;
    auto result = options.parse(argc, argv);

    string weights_file = result["weights"].as<string>();
    string mel_file = result["mel"].as<string>();



    return 0;
}
