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
#include <random>
#include <stdexcept>
#include <limits>
// Déclaration du générateur (doit être défini dans prior.cpp ou un fichier d'utilitaires)
extern std::mt19937 generator_prior;
// Définition de types pour simplifier la lecture
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

int sample_discrete(const Vector& probabilities);
double sampleBeta(double alpha, double beta);