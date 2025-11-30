#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <utility> // Pour std::pair
#include <algorithm>
#include <map> // Nécessaire pour la ré-indexation

// Définition de types pour simplifier la lecture
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
size_t countNonzeroClusterJ(const Vector& SS, int j);
void updateClusterSizeJ(Vector& CS, const Vector& SS, int j);


