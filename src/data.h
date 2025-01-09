#ifndef DATA
//Input Layer
const int INPUT_SIZE = 784;
const int OUTPUT_SIZE = 10;
const int Train_Data_size = 60000;
const int Test_Data_size = 10000;
void Get_Input_Data(float Data[][INPUT_SIZE], bool Get_Train_Data);
void Get_Output_Data(float Output[][OUTPUT_SIZE], bool Get_Train_Data);

#endif