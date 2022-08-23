#include <fstream>
#include <iostream>
#include <limits>

#include <gtest/gtest.h>

#include "logreg_classifier.h"
#include "fashion_classifier.h"
#include "helpers.h"

//using LogregClassifier;
using std::clog;

TEST(LogregClassifier, compare_to_python) {
    auto predictor = FashionClassifier{"train/logreg_coef.txt"};

    auto features = LogregClassifier::features_t{};

    int class_expected = 0;

    
    std::ifstream test_data{"train/test_data_logreg.txt"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        test_data >> class_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        auto pred = predictor.fashionIsMyProfeshion(features);

        EXPECT_EQ(class_expected, pred);
    }
}
