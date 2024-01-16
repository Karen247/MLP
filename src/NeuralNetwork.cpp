//
// Created by karen on 12/4/23.
//
// NeuralNetwork_Project.cpp
// Karen Pailozian & Remi Monteil
//
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <fstream>
#include <ctime>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <cmath>
const float lambda = 1.050;
const float alpha = 1.673;
const int BatchSize = 16;
const float beta1 = 0.9;
const float beta2 = 0.999;
//Input Layer
const int Input_Layer = 784;

//Hidden layer
const int Hidden_Layer1 = 128;
const int Hidden_Layer2 = 128;

//Output Layer
const int Output_Layer = 10;

const int epoch = 60;
float learning_rate = 0.001;

const float momentum = 0.9;
const float epsilon = 1;
const float stabilization = 1e-8;

float Weights_h1[Hidden_Layer1][Input_Layer];
float Weights_h2[Hidden_Layer2][Hidden_Layer1];
float Weights_o[Output_Layer][Hidden_Layer2];

float Deltas_w_h1[Hidden_Layer1][Input_Layer];
float Deltas_w_h2[Hidden_Layer2][Hidden_Layer1];
float Deltas_w_o[Output_Layer][Hidden_Layer2];

float Bias_h1[Hidden_Layer1];
float Bias_h2[Hidden_Layer2];
float Bias_o[Output_Layer];

float Input[Input_Layer];
float Output_h1[Hidden_Layer1]; //input of hidden_layer2
float Output_h2[Hidden_Layer2]; //input of output
float Output_o[Output_Layer];

float Output_Expected[Output_Layer];

float Potential_h1[Hidden_Layer1];
float Potential_h2[Hidden_Layer2];
float Potential_o[Output_Layer];

float Deltas_h1[Hidden_Layer1];
float Deltas_h2[Hidden_Layer2];
float Deltas_o[Output_Layer];

const int Train_Data_size = 60000;
float Train_Data[Train_Data_size][Input_Layer];
float Train_Output[Train_Data_size][Output_Layer];
float Prediction_Train_Output[Train_Data_size];

const int Test_Data_size = 10000;
float Test_Data[Test_Data_size][Input_Layer];
float Test_Output[Test_Data_size][Output_Layer];
float Prediction_Test_Output[Test_Data_size];

std::vector<int> train_index;
std::vector<int> validation_index;
// WORKS OK
void Get_Input_Data(bool Get_Train_Data)
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
                if (Get_Train_Data) {
                    Train_Data[i][j] = static_cast<float>(std::atoi(word.c_str())) / 255;
                }

                else {
                    Test_Data[i][j] = static_cast<float>(std::atoi(word.c_str())) / 255;
                }
                j++;
                if (j >= Input_Layer)
                    break;
            }

            if (Get_Train_Data) {
                if (i == Train_Data_size) {
                    break;
                }
            }
            else {
                if (i == Test_Data_size)
                    break;
            }
            i++;

        }
    }
    else {

        std::cout << "Could not open the file\n";
    }
}
// WORKS OK
void Get_Output_Data(bool Get_Train_Data)
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

            if (Get_Train_Data) {
                int number = std::stoi(line);
                for (int pos = 0; pos < Output_Layer; pos++) {
                    Train_Output[i][pos] = 0.0f;
                }
                Train_Output[i][number] = 1.0f;
            }
            else {
                int number = std::stoi(line);
                for (int pos = 0; pos < Output_Layer; pos++) {
                    Test_Output[i][pos] = 0.0f;
                }
                Test_Output[i][number] = 1.0f;
            }
            if (Get_Train_Data) {
                if (i == Train_Data_size) {
                    break;
                }
            }
            else {
                if (i == Test_Data_size)
                    break;
            }
            i++;

        }
    }
    else
        std::cout << "Could not open the file\n";
}

// Works OK
void Init_Input(int index, bool Training) {
    for (int i = 0; i < Input_Layer; i++) {
        if (Training) {
            Input[i] = Train_Data[index][i];
        }
        else
            Input[i] = Test_Data[index][i];
    }
}

void SetPredictionValue(size_t index, bool train) {
    float max = 0.0;
    size_t max_index;

    for (size_t i = 0; i < Output_Layer; i++) {
        if (max < Output_o[i]) {
            max = Output_o[i];
            max_index = i;
        }
    }

    if (train)
        Prediction_Train_Output[index] = max_index;
    else
        Prediction_Test_Output[index] = max_index;
}

void CreatePredictionFile(bool train) {

    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    std::string current_working_dir(buff);

    if (train)
        current_working_dir += "/train_predictions.csv";
    else
        current_working_dir += "/test_predictions.csv";

    std::ofstream  predictionfile;
    predictionfile.open(current_working_dir);

    if (train){
        for (size_t i = 0; i < Train_Data_size; i++) {
            predictionfile << Prediction_Train_Output[i];
            if (i + 1 < Train_Data_size)
                predictionfile << "\n";
        }
    }
    else {
        for (size_t i = 0; i < Test_Data_size; i++) {
            predictionfile << Prediction_Test_Output[i];
            if (i + 1 < Test_Data_size)
                predictionfile << "\n";
        }
    }
}

// WORKS OK
void Init_Weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> DistRelu(0, (float) 2 / Input_Layer);
    std::uniform_real_distribution<> DistSelu(0, (float) 1 / Input_Layer);
    std::uniform_real_distribution<> DistSoft(0, (float) 2 / (Output_Layer + Hidden_Layer1));

    for (size_t i = 0; i < Hidden_Layer1; i++) {
        Bias_h1[i] = DistRelu(gen);
        for (size_t j = 0; j < Input_Layer; j++) {
            Weights_h1[i][j] = DistSoft(gen);
        }
    }

    for (size_t i = 0; i < Hidden_Layer2; i++) {
        Bias_h2[i] = DistRelu(gen);
        for (size_t j = 0; j < Hidden_Layer1; j++) {
            Weights_h2[i][j] = DistSoft(gen);
        }
    }

    for (size_t i = 0; i < Output_Layer; i++) {
        Bias_o[i] = DistSoft(gen);
        for (size_t j = 0; j < Hidden_Layer2; j++) {
            Weights_o[i][j] = DistSoft(gen);
        }
    }
}
// WORKS OK
void Init_Output_Expected(int index, bool Training) {
    for (int i = 0; i < Output_Layer; i++) {
        if (Training) {
            Output_Expected[i] = Train_Output[index][i];
        }
        else
            Output_Expected[i] = Test_Output[index][i];
    }
}


float Selu(float x) {
    if (x >= 0) return lambda * x;
    else return lambda * (alpha * (exp(x) - 1));
}

float Derivate_Selu(float x) {
    if (x >= 0) return lambda;
    else return lambda * (alpha * exp(x));
}

float Relu(float x) {
    return x > 0 ? x : 0;
}

float Derivate_Relu(float x) {
    return x > 0 ? 1 : 0;
}

float Sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float Derivate_Sigmoid(float x) {
    return x * (1 - x);
}

void Forward_Propagation() {
    //WORKS OK
    for (size_t i = 0; i < Hidden_Layer1; i++) {
        float tmp_potential = Bias_h1[i];
        for (size_t j = 0; j < Input_Layer; j++) {
            tmp_potential += Weights_h1[i][j] * Input[j];
        }
        Potential_h1[i] = tmp_potential;
        Output_h1[i] = Sigmoid(tmp_potential);
    }


    // WORKS OK
    for (size_t i = 0; i < Hidden_Layer2; i++) {
        float tmp_potential = Bias_h2[i];

        for (size_t j = 0; j < Hidden_Layer1; j++) {
            tmp_potential += Weights_h2[i][j] * Output_h1[j];
        }

        Potential_h2[i] = tmp_potential;
        Output_h2[i] = Sigmoid(tmp_potential);
    }

    // FOR final layer softmax is used
    float sum = 0.0;
    for (size_t i = 0; i < Output_Layer; i++) {
        float tmp_potential = Bias_o[i];

        for (size_t j = 0; j < Hidden_Layer2; j++) {
            tmp_potential += Weights_o[i][j] * Output_h2[j];
        }

        Potential_o[i] = tmp_potential;
        sum += exp(tmp_potential);
    }

    // WORKS OK
    for (size_t i = 0; i < Output_Layer; i++)
    {
        Output_o[i] = exp(Potential_o[i]) / sum;
    }
}

// WORKS OK
float cross_entropy() {
    float loss = 0.0;
    for (size_t i = 0; i < Output_Layer; i++) {
        loss += -1 * (Output_Expected[i] * std::log2(Output_o[i] + stabilization)); // add binary cross entropy
    }
    return loss;
}

void accuracy(float& acc) {
    float max = 0.0;
    size_t max_index;

    for (size_t i = 0; i < Output_Layer; i++) {
        if (max < Output_o[i]) {
            max = Output_o[i];
            max_index = i;
        }
    }

    if (Output_Expected[max_index] == 1)
        acc++;
}

// WORKS OK
void Backward_Propagation() {
    float total_error;

    // WORKS OK
    for (size_t i = 0; i < Output_Layer; i++) {
        Deltas_o[i] = Output_o[i] - Output_Expected[i];
    }

    // WORKS OK
    for (size_t i = 0; i < Hidden_Layer2; i++) {
        total_error = 0.0;

        for (size_t j = 0; j < Output_Layer; j++) {
            total_error += Deltas_o[j] * Weights_o[j][i];
        }

        Deltas_h2[i] = total_error * Derivate_Sigmoid(Output_h2[i]);
    }

    // WORKS OK
    for (size_t i = 0; i < Hidden_Layer1; i++) {
        total_error = 0.0;

        for (size_t j = 0; j < Hidden_Layer2; j++) {
            total_error += Deltas_h2[j] * Weights_h2[j][i];
        }
        Deltas_h1[i] = total_error * Derivate_Sigmoid(Output_h1[i]);
    }


    // Weight update
    //WORKS OK
    for (size_t i = 0; i < Hidden_Layer1; i++) {
        Bias_h1[i] += -learning_rate * Deltas_h1[i];

        for (size_t j = 0; j < Input_Layer; j++) {
            Deltas_w_h1[i][j] = -learning_rate * Deltas_h1[i] * Input[j] + Deltas_w_h1[i][j] * momentum;
            Weights_h1[i][j] += Deltas_w_h1[i][j];

        }
    }

    // WORKS OK
    for (size_t i = 0; i < Hidden_Layer2; i++) {
        Bias_h2[i] += -learning_rate * Deltas_h2[i];

        for (size_t j = 0; j < Hidden_Layer1; j++) {
            Deltas_w_h2[i][j] = -learning_rate * Deltas_h2[i] * Output_h1[j] + Deltas_w_h2[i][j] * momentum;
            Weights_h2[i][j] += Deltas_w_h2[i][j];
        }
    }

    // WORKS OK
    for (size_t i = 0; i < Output_Layer; i++) {
        Bias_o[i] += -learning_rate * Deltas_o[i];
        for (size_t j = 0; j < Hidden_Layer2; j++) {
            Deltas_w_o[i][j] = -learning_rate * Deltas_o[i] * Output_h2[j] + Deltas_w_o[i][j] * momentum;
            Weights_o[i][j] += Deltas_w_o[i][j];
        }
    }

}

int learning_v2(int tmp_index) {
    Init_Input(tmp_index, true);
    Init_Output_Expected(tmp_index, true);
    Forward_Propagation();
    Backward_Propagation();
    return 0;
}
void shuffle_data(int &seed) {
    srand(seed);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::unordered_set<int> set_random;
    std::uniform_int_distribution<int> dist{ 0, Train_Data_size - 1 };
    while (set_random.size() != Train_Data_size) {
        set_random.insert((int) rand() % (Train_Data_size));
    }

    std::unordered_set<int> set_random_validation;
    std::uniform_int_distribution<int> dist_validation{ 0, 6000 - 1 };
    while (set_random_validation.size() != 6000) {
        int val_index = (int) rand() % (Train_Data_size);
        auto it = set_random.find(val_index);
        if (it != set_random.end()) {
            set_random.erase(it);
        }
        set_random_validation.insert(val_index);
    }
    train_index.clear();
    validation_index.clear();
    train_index.insert(train_index.end(), set_random.begin(), set_random.end());
    validation_index.insert(validation_index.end(), set_random_validation.begin(), set_random_validation.end());
}

float validate() {
    float acc = 0.0;
    int data_size = 0;

    for (auto j : validation_index) {
        Init_Input(j, true);
        Init_Output_Expected(j, true);
        Forward_Propagation();
        accuracy(acc);
    }
    return acc;
}
int Train_v2() {

    int seed = 1;
    int i = 0;

    for (; i < epoch; i++) {
        shuffle_data(seed);
        seed+=1;
        int index = 0;
        int batch_num = 1;
        int count = 0;
        for (int j = 0; j < Train_Data_size * 0.9 - ((int)(Train_Data_size * 0.9) % BatchSize); j += BatchSize) {
            count = 0;
            int num_good_results = 0;
            float total_cross_entropy = 0.0;
            while (count < BatchSize) {
                printf("%d and %d and %d \n", index, train_index[index], batch_num);
                printf("entropy: %f\n", cross_entropy());
                total_cross_entropy += cross_entropy();

                learning_v2(train_index[index]);
                index++;
                count++;

                if (cross_entropy() < 0.001) {
                    num_good_results += 1;
                    if (num_good_results == 3) {
                        printf("stopping on entropy: %f\n", cross_entropy());
                        break;
                    }
                }
            }

            index = j;
            batch_num++;

            if (batch_num * BatchSize >= 0.9 * Train_Data_size) {
                break;
            }

        }
        float accuracy = validate();
        if (accuracy / (Train_Data_size * 0.1) > 0.94) {
                return i;
        }
    }
    return i;
}

void Testing(bool train) {
    float acc = 0.0;
    int data_size = 0;
    if (train)
        data_size = Train_Data_size;
    else
        data_size = Test_Data_size;
    for (int j = 0; j < data_size; j++) {
        Init_Input(j, train);
        Init_Output_Expected(j, train);
        Forward_Propagation();
        accuracy(acc);
        SetPredictionValue(j, train);

    }
    if (train)
        printf("Accuracy %% : %f  \n\n", acc * 100 / Train_Data_size);
    else
        printf("Accuracy %% : %f  \n\n", acc * 100 / Test_Data_size);
}
int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    printf("starting to get the data\n");
    Get_Input_Data(true);
    Get_Input_Data(false);
    Get_Output_Data(true);
    Get_Output_Data(false);
    Init_Weights();
    printf("starting to train\n");
    int epochs = Train_v2();
    printf("Epochs: %d\n", epochs);
    Testing(true);
    CreatePredictionFile(true);
    Testing(false);
    CreatePredictionFile(false);
    printf("Batch: %d\n Epoch:  %d\n LR: %f\n H1: %d\n H2: %d\n", BatchSize, epoch, learning_rate, Hidden_Layer1, Hidden_Layer2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(stop - start);

    std::cout << "Time taken by function: "
              << duration.count() << " Minutes" << std::endl;
}