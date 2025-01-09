#include "activation.h"

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
