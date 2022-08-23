#include "fashion_classifier.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>


FashionClassifier::FashionClassifier(const std::string& fileCoef)
{
    std::ifstream istream{fileCoef};
    assert(istream.is_open());
    std::string line;
    
    while(std::getline(istream, line))
    {
        std::istringstream linestream{line};
        auto coef = read_vector(linestream);
        m_classifiers.push_back(coef);
    }
}

int FashionClassifier::fashionIsMyProfeshion(const LogregClassifier::features_t& feat) {
    int fashion = 0;
    float predMax = 0;
    float pred;
    for(int i = 0; i < static_cast<int>(m_classifiers.size()); ++i)
    {
        pred = m_classifiers[i].predict_proba(feat);
        if (predMax < pred)
        {
            predMax = pred;
            fashion = i;
        }
    }
    return fashion;
}
