#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include "data.h"
#include <stddef.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

void Get_Input_Data(float Data[][INPUT_SIZE], bool Get_Train_Data)
{

    size_t i;
    std::string line;
    size_t j;
    std::string word;

    char buff[FILENAME_MAX];
    GetCurrentDir(buff, FILENAME_MAX );
    std::string current_working_dir(buff);

    if (Get_Train_Data)
        current_working_dir += "/data/fashion_mnist_train_vectors.csv";
    else
        current_working_dir += "/data/fashion_mnist_test_vectors.csv";

    std::fstream file(current_working_dir, std::ios::in);

    if (file.is_open())
    {
        std::cout << "Could open the file\n";
        i = 0;
        while (std::getline(file, line))
        {
            std::stringstream str(line);

            j = 0;
            while (std::getline(str, word, ','))
            {
                Data[i][j] = static_cast<float>(std::atoi(word.c_str())) / 256;
                j++;
                if (j >= INPUT_SIZE)
                    break;
            }
            i++;

        }
    }
    else {

        std::cout << "Could not open the file\n";
    }
}

void Get_Output_Data(float Output[][OUTPUT_SIZE], bool Get_Train_Data)
{

    size_t i;
    std::string line;
    std::string word;

    char buff[FILENAME_MAX];
    GetCurrentDir(buff, FILENAME_MAX );
    std::string current_working_dir(buff);

    if (Get_Train_Data)
        current_working_dir += "/data/fashion_mnist_train_labels.csv";
    else
        current_working_dir += "/data/fashion_mnist_test_labels.csv";

    std::fstream file(current_working_dir, std::ios::in);
    if (file.is_open()) {
        std::cout << "Could open the file\n";
        i = 0;
        while (std::getline(file, line)) {
            std::stringstream str(line);

            int number = std::stoi(line);
            for (int pos = 0; pos < OUTPUT_SIZE; pos++) {
                Output[i][pos] = 0.0f;
            }
            Output[i][number] = 1.0f;
            i++;

        }
    }
    else
        std::cout << "Could not open the file\n";
}
