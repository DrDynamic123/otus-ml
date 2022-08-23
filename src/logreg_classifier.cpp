#include "logreg_classifier.h"

#include <cassert>
#include <numeric>
#include <cmath>
#include <stdexcept>

//using LogregClassifier;

namespace {

template<typename T>
auto sigma(T x) {
    return 1/(1 + std::exp(-x));
}

}

LogregClassifier::LogregClassifier(const coef_t& coef)
    : m_coef{coef} 
{
    assert(!m_coef.empty());
}

float LogregClassifier::predict_proba(const features_t& feat) const {
    if (feat.size() + 1 != m_coef.size())
        throw std::runtime_error("feature vector size missmatch");
    auto p = std::inner_product(feat.begin(), feat.end(), ++m_coef.begin(), m_coef.front());
    return sigma(p);
}
