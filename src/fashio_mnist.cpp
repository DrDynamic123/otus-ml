#include "fashion_classifier.h"

#include <iostream>
#include <fstream>
#include <cassert>


/**
 * @brief main
 * 
 * @return int 
 */
int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <data file> <model file>" << std::endl;
            return EXIT_FAILURE;
        }

        std::string coefFile = argv[2];
        std::string featFile = argv[1];

        auto predictor = FashionClassifier{coefFile};

        auto features = LogregClassifier::features_t{};

        int class_expected = 0;

        std::ifstream test_data{featFile};
        assert(test_data.is_open());
        int guess = 0;
        int number = 0;
        for (;;) {
            test_data >> class_expected;
            if (!read_features(test_data, features)) {
                break;
            }
            auto pred = predictor.fashionIsMyProfeshion(features);

            ++number;
            if (class_expected == pred)
                ++guess;
        }
        std::cout << static_cast<float>(guess) / number << "\n";
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
