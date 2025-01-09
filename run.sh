#!/bin/bash
## change this file to your needs

echo "Adding some modules"

# module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network
g++ -Wall -std=c++17 -Ofast src/NeuralNetwork.cpp src/activation.cpp src/data.cpp -o network

echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 ./network
