#include "potasms.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
int main(int argc,char**argv) {
    if(argc!=5) { std::cerr << "usage: potasms_native_bench WEIGHTS.bin SCALE BIAS SYNTHETIC_TEXTS.txt\n"; return 2; }
    auto start=Clock::now();
    std::ifstream weights(argv[1],std::ios::binary);
    std::vector<int8_t> w((std::istreambuf_iterator<char>(weights)),{});
    float scale=std::stof(argv[2]),bias=std::stof(argv[3]);
    if(w.empty() || w.size()>1048576) return 2;
    std::ifstream messages(argv[4]);std::vector<std::string> texts;
    for(std::string line;std::getline(messages,line);) texts.push_back(line);
    if(texts.empty()) return 2;
    double load_ms=milliseconds(start);
    auto infer=[&](const std::string &text) {
        pota_feature *f=nullptr;size_t n=0;float z=0;
        int result=pota_features(reinterpret_cast<const uint8_t*>(text.data()),text.size(),POTA_CASEFOLD_V2,
                                 static_cast<uint32_t>(w.size()),2,5,&f,&n);
        if(result==POTA_OK) result=pota_logit(f,n,w.data(),w.size(),scale,bias,nullptr,0,&z);
        pota_free(f);
        if(result!=POTA_OK) { std::cerr<<"inference rejected\n";std::exit(1); }
        return pota_sigmoid(z);
    };
    start=Clock::now();double checksum=infer(texts.front());double first=milliseconds(start);
    for(size_t i=0;i<100;++i) checksum+=infer(texts[i%texts.size()]);
    std::vector<double> duration;duration.reserve(5000);
    for(size_t i=0;i<5000;++i) {
        start=Clock::now();checksum+=infer(texts[i%texts.size()]);duration.push_back(milliseconds(start));
    }
    std::sort(duration.begin(),duration.end());
    long rss=0;
#if defined(__APPLE__) || defined(__linux__)
    rusage usage{};getrusage(RUSAGE_SELF,&usage);rss=usage.ru_maxrss;
#if defined(__linux__)
    rss*=1024;
#endif
#endif
    std::cout<<std::setprecision(9)<<"{\"runtime\":\""<<pota_version()<<"\",\"weights_bytes\":"<<w.size()
        <<",\"load_ms\":"<<load_ms<<",\"first_inference_ms\":"<<first
        <<",\"warm_iterations\":5000,\"median_ms\":"<<duration[2500]<<",\"p95_ms\":"<<duration[4750]
        <<",\"process_peak_rss_bytes\":"<<rss<<",\"checksum\":"<<checksum<<"}\n";
}
