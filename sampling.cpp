#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <vector>
#include <utility> // Pour std::pair
#include <algorithm>
#include <map> // Nécessaire pour la ré-indexation
#include <random>
#include <stdexcept>
#include <limits>
#include "sampling.h"
// Déclaration du générateur (doit être défini dans prior.cpp ou un fichier d'utilitaires)
extern std::mt19937 generator_prior;
//extern std::mt19937_64 rng;  // ou ton générateur global
// Définition de types pour simplifier la lecture
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
int sampleDiscrete(const Vector& probabilities) {
    
    // Vérifier si le vecteur de probabilités est vide
    if (probabilities.empty()) {
        throw std::invalid_argument("sampleDiscrete: Le vecteur de probabilités est vide.");
    }
    
    // Le générateur std::discrete_distribution fait la normalisation en interne, 
    // mais il est plus sûr de lui donner des probabilités positives.
    // Votre code Python effectue déjà une normalisation robuste avant l'appel.

    // Créer la distribution discrète (multinomiale)
    // Elle prend les poids (probabilités) et les utilise pour échantillonner un index.
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    // Effectuer le tirage
    // L'entier retourné est l'indice (0-basé) du cluster sélectionné.
    return dist(generator_prior);
}
// Sample the Beta distribution



double sampleBeta(double a, double b) {

     if (!(a > 0.0) || !(b > 0.0)) {
        std::cerr << "[beta_rvs] Paramètres invalides: a=" << a
                  << ", b=" << b << "\n";
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::gamma_distribution<double> g1(a, 1.0);
    std::gamma_distribution<double> g2(b, 1.0);

    const double eps = 1e-12;      // seuil de "trop petit"
    const int max_tries = 10;      // nombre max de tentatives

    for (int k = 0; k < max_tries; ++k) {
        std::gamma_distribution<double> dist_alpha(a, 1);
        std::gamma_distribution<double> dist_beta(b, 1);

        double x = dist_alpha(generator_prior);
        double y = dist_beta(generator_prior);
        double denom = x + y;

        if (denom > eps) {
            double val = x / denom;
            if (val >= 0.0001 && val <= 0.9999 && std::isfinite(val)) {
                return val;
            }
        }
    }

    // 2) Si on arrive ici : trop de problèmes numériques → fallback
    double mean = a / (a + b);   // moyenne théorique de la Beta(a,b)
   /* std::cerr << "[beta_rvs] denom trop petit ou instable, "
              << "retourne la moyenne théorique: " << mean << "\n"; */
    return mean;
   
}




double sampleGamma(double shape, double scale) {
    
    // Vérification des conditions : la forme et l'échelle doivent être > 0
    if (shape <= 0.0 || scale <= 0.0) {
        // En cas de paramètre invalide, retourne une valeur stable clampée.
        return 0.001; 
    }

    // Le constructeur C++ std::gamma_distribution utilise : (shape, rate)
    // Où rate = 1 / scale
    //double rate = 1.0 / scale;
    double rate =1/scale;
    
    // Créer la distribution Gamma
    std::gamma_distribution<double> gamma_dist(shape, rate);
    
    // Effectuer le tirage
    return gamma_dist(generator_prior);
}