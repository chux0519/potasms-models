#include "potasms.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

std::string normalized(std::string s,int mode) {
    uint8_t *p=nullptr;size_t n=0;
    CHECK(pota_normalize(reinterpret_cast<const uint8_t*>(s.data()),s.size(),mode,&p,&n)==POTA_OK);
    std::string out(reinterpret_cast<char*>(p),n);pota_free(p);return out;
}
int main() {
    {
        pota_feature signedFeatures[]={{0,.6f},{1,-.8f}};
        float head[]={1,2},delta[]={0,0},score=0;
        CHECK(pota_float_logit(signedFeatures,2,head,2,.1f,nullptr,0,&score)==POTA_OK);
        CHECK(std::abs(score+.9f)<1e-6);
        int8_t legacyHead[]={1,2};
        CHECK(pota_logit(signedFeatures,2,legacyHead,2,1,0,nullptr,0,&score)==POTA_INVALID);
        CHECK(pota_pa2_margin_sample(signedFeatures,2,delta,2,-.2f,1,1,.05f,2)==POTA_OK);
        CHECK(std::abs(delta[0]-.1f)<1e-6 && std::abs(delta[1]+.133333333f)<1e-6);
        CHECK(pota_float_logit(signedFeatures,2,nullptr,2,0,delta,0,&score)==POTA_OK);
        CHECK(score>0); // signed features must improve the supplied negative margin
        const float before[]={delta[0],delta[1]};
        signedFeatures[1].bucket=0;
        CHECK(pota_pa2_margin_sample(signedFeatures,2,delta,2,-.2f,1,1,.05f,2)==POTA_INVALID);
        CHECK(delta[0]==before[0] && delta[1]==before[1]);
        signedFeatures[1].bucket=1;
        CHECK(pota_pa2_margin_sample(signedFeatures,2,delta,2,1001,1,1,.05f,2)==POTA_INVALID);
        CHECK(delta[0]==before[0] && delta[1]==before[1]);
    }
    {
        pota_feature features[]={{1,.6f},{3,.8f}};
        int8_t weights[4]={0,0,0,0};float delta[4]={0,0,0,0};
        CHECK(pota_pa2_sample(features,2,weights,4,1,0,delta,0,0,1,1,.05f,2)==POTA_OK);
        CHECK(std::abs(delta[1]-.02f)<1e-7 && std::abs(delta[3]-(.05f/1.5f*.8f))<1e-7);
        CHECK(delta[0]==0 && delta[2]==0);
        auto a=delta[1],b=delta[3];
        CHECK(pota_pa2_sample(features,2,weights,4,1,0,delta,0,0,1,0,.05f,2)==POTA_INVALID);
        CHECK(delta[1]==a && delta[3]==b);
        features[1].bucket=1;
        CHECK(pota_pa2_sample(features,2,weights,4,1,0,delta,0,0,1,1,.05f,2)==POTA_INVALID);
        CHECK(delta[1]==a && delta[3]==b);
    }
    CHECK(normalized("ΩΣ ΟΣ Σ",POTA_LEGACY_LOWER)=="ως ος σ");
    CHECK(normalized("ΩΣ ΟΣ Σ Straße",POTA_CASEFOLD_V2)=="ωσ οσ σ strasse");
    CHECK(normalized("  ＷＩＮ\xe2\x80\x8b\n prize  ",POTA_CLEAN)=="WIN prize");
    CHECK(normalized("İ",POTA_LEGACY_LOWER)=="i\xcc\x87");
    uint8_t *p=nullptr;size_t n=0;
    CHECK(pota_normalize(reinterpret_cast<const uint8_t*>("\xff"),1,0,&p,&n)==POTA_INVALID);
    std::string long_text(16385,'a');
    CHECK(pota_normalize(reinterpret_cast<const uint8_t*>(long_text.data()),long_text.size(),0,&p,&n)==POTA_TOO_LARGE);
    pota_feature *f=nullptr;
    CHECK(pota_features(reinterpret_cast<const uint8_t*>("hello"),5,2,64,5,2,&f,&n)==POTA_INVALID);
    CHECK(pota_features(reinterpret_cast<const uint8_t*>("hello"),5,2,64,2,5,&f,&n)==POTA_OK);
    float norm=0;for(size_t i=0;i<n;++i){norm+=f[i].value*f[i].value;if(i)CHECK(f[i-1].bucket<f[i].bucket);}
    CHECK(std::abs(norm-1)<1e-6);
    std::vector<int8_t>w(64,0);std::vector<float>d(64,0);float db=0,z;
    CHECK(pota_sgd_sample(f,n,w.data(),64,1,0,d.data(),&db,1,.12f,.0005f,4)==POTA_OK);
    std::vector<float>aligned;for(size_t i=0;i<n;++i)aligned.push_back(d[f[i].bucket]);
    CHECK(pota_logit(f,n,w.data(),64,1,0,aligned.data(),db,&z)==POTA_OK && z>0);
    float before=0;for(size_t i=0;i<n;++i)before+=f[i].value*d[f[i].bucket];
    CHECK(pota_anchor_sample(f,n,d.data(),64,.5f)==POTA_OK);
    float after=0;for(size_t i=0;i<n;++i)after+=f[i].value*d[f[i].bucket];
    CHECK(std::abs(after-before*.5f)<1e-6 && db>0);
    auto saved=d;
    CHECK(pota_anchor_sample(f,n,d.data(),64,2)==POTA_INVALID && d==saved);
    auto value=f[0].value;f[0].value=10;
    CHECK(pota_anchor_sample(f,n,d.data(),64,.5f)==POTA_INVALID && d==saved);
    f[0].value=value;
    f[0].bucket=64;
    CHECK(pota_anchor_sample(f,n,d.data(),64,.5f)==POTA_INVALID && d==saved);
    CHECK(pota_logit(f,n,w.data(),64,1,0,nullptr,0,&z)==POTA_INVALID);
    pota_free(f);
    // Additional kind; strict exact behavior and the C struct/ABI are unchanged.
    std::string pattern="Pay now 4501";
    pota_rule rule{POTA_NORMALIZED_EXACT,0,1,1,0,
        reinterpret_cast<const uint8_t *>(pattern.data()),pattern.size()};
    CHECK(pota_rule_validate(&rule)==POTA_OK);
    auto matchRule=[&](const std::string &body) {
        int32_t matched=-2;
        CHECK(pota_rules_match(&rule,1,nullptr,0,
            reinterpret_cast<const uint8_t *>(body.data()),body.size(),&matched)==POTA_OK);
        return matched;
    };
    CHECK(matchRule("  PAY  now\n4501 ")==0);
    CHECK(matchRule("Pay now 4502")==-1);
    CHECK(matchRule("Do not pay now 4501")==-1);
    rule.kind=POTA_EXACT;
    CHECK(matchRule("Pay  now\n4501")==-1);

}
