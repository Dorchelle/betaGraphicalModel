
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm> 
#include <iostream>
#include <map> 
#include "fonctions.h"
// Nécessaire pour la ré-indexation
// Définition de types pour simplifier la lecture
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
// Déclaration d'un générateur aléatoire (supposé défini dans prior.cpp ou un utilitaire)
extern std::mt19937 generator_prior;
PropAjoutRetrait extractNoeud(const Matrix& A_Omega);