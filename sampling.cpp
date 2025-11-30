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
// Définition de types pour simplifier la lecture
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
int sampleDiscrete(const Vector& probabilities) {
    
    // Vérifier si le vecteur de probabilités est vide
    if (probabilities.empty()) {
        throw std::invalid_argument("sample_discrete: Le vecteur de probabilités est vide.");
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
double sampleBeta(double alpha, double beta) {
    
    // Vérification de base pour les paramètres Gamma
    if (alpha <= 0.0 || beta <= 0.0) {
        // La distribution Bêta n'est pas bien définie ou dégénérée pour alpha/beta <= 0.
        return 0.001; // Retourne la valeur clampée pour la stabilité.
    }
    
    // std::gamma_distribution utilise (shape, rate). Ici, rate = 1.0.
    std::gamma_distribution<double> gamma_a(alpha, 1.0);
    std::gamma_distribution<double> gamma_b(beta, 1.0);
    
    // Tirer les deux variables Gamma indépendantes
    double val_a = gamma_a(generator_prior);
    double val_b = gamma_b(generator_prior);
    
    // Calculer le ratio: Gamma(alpha) / (Gamma(alpha) + Gamma(beta))
    if (std::abs(val_a + val_b) < 1e-12) { 
        // Cas où la somme est presque nulle (très peu probable si alpha et beta sont > 0)
        return 0.5; 
    }
    
    return val_a / (val_a + val_b);
}