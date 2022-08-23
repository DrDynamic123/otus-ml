#pragma once

#include "logreg_classifier.h"
#include "helpers.h"

#include <vector>
#include <map>

const std::map<int, std::string> goods{
    {0, "T-shirt/top"},
    {1, "Trouser"},
    {2, "Pullover"},
    {3, "Dress"},
    {4, "Coat"},
    {5, "Sandal"},
    {6, "Shirt"},
    {7, "Sneaker"},
    {8, "Bag"},
    {9, "Ankle boot"}
};

class FashionClassifier {
public:
    FashionClassifier(const std::string& fileCoef);

    int fashionIsMyProfeshion(const LogregClassifier::features_t& feat);

protected:
    std::vector<LogregClassifier> m_classifiers;
};


