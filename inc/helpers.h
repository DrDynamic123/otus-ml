#pragma once

#include <istream>
#include <vector>

#include "classifier.h"

bool read_features(std::istream& stream, BinaryClassifier::features_t& features);

std::vector<float> read_vector(std::istream&);
