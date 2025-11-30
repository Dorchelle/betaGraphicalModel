// prior.cpp
#include "prior.h"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "sampling.h"
// --- Générateur de Nombres Aléatoires (UNIQUE dans ce fichier) ---
// Définition du générateur statique pour le module prior

static std::random_device rd;
std::mt19937 generator_prior(rd()); 
extern double sampleBeta(double alpha, double beta); 
// --- 1. Priors Simples (Uniforme, Graphe) ---
double priorSigma(double a_sigma, double b_sigma) {
    if (a_sigma >= b_sigma) throw std::invalid_argument("priorSigma: a_sigma doit être < b_sigma.");
    std::uniform_real_distribution<double> uniform_dist(a_sigma, b_sigma);
    return uniform_dist(generator_prior);
}


Vector priorXI(double a_xi, double b_xi, int p) {
    Vector xi(p);
    std::uniform_real_distribution<double> uniform_dist(a_xi, b_xi);
    for (int i = 0; i < p; ++i) {
        xi[i] = uniform_dist(generator_prior);
    }
    return xi;
}

Matrix adjacencyMatrix(int p) {
    if (p <= 0) return Matrix();
    Matrix A(p, Vector(p, 0.0));
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    double edge_probability = 0.5; // Hypothèse G(p, 0.5)

    for (int i = 0; i < p; ++i) {
        for (int j = i + 1; j < p; ++j) {
            if (uniform_01(generator_prior) < edge_probability) {
                A[i][j] = 1.0;
                A[j][i] = 1.0;
            }
        }
    }
    return A;
}


// --- 2. Fonctions d'Échantillonnage de Priors (Gamma, Beta) ---
Vector priorGamma(const Matrix& theta, double b, const Vector& SS) {
    Vector Cs = calculateCs(SS);
    int Ncl = Cs.size();
    Vector alpha(Ncl);
    
    // Logique simplifiée : L'appel à meanSS dans ce contexte Python est ambigu.
    // L'échantillonnage Gamma utilise des paramètres dérivés du cluster j.
    for (int j = 0; j < Ncl; ++j) {
        // Placeholder pour les paramètres de forme/échelle
        double shape = 1.0;
        double rate = 1.0 / b;

        std::gamma_distribution<double> gamma_dist(shape, rate);
        alpha[j] = gamma_dist(generator_prior);
        
        if (alpha[j] < 0.0001) alpha[j] = 0.0001;
    }
    return alpha;
}


Vector priorBeta(const Matrix& lambda_1, double b, const Vector& SS) {
    Vector Cs = calculateCs(SS);
    int Ncl = Cs.size();
    Vector alpha(Ncl);
    
    for (int j = 0; j < Ncl; ++j) {
        // Placeholder pour les paramètres de forme
        double a_shape = 1.0; 
        double b_shape = b;

        alpha[j] = sampleBeta(a_shape, b_shape);
        
        if (alpha[j] < 0.0001) alpha[j] = 0.0001;
        if (alpha[j] > 0.9999) alpha[j] = 0.9999;
    }
    return alpha;
}


// ... priorGamma1 et priorBeta1 sont similaires et sont des placeholders ...
Vector priorGamma1(const Matrix& theta, double b, const Vector& SS) {
    return Vector(calculateCs(SS).size(), 0.0001);
}


Vector priorBeta1(const Matrix& lambda_1, double b, const Vector& SS) {
    return Vector(calculateCs(SS).size(), 0.5);
}


// --- 3. Échantillonneurs de Processus Stochastiques ---
// Les implémentations de priorNormal et priorLogNormal sont reprises de notre travail précédent.
// Elles utilisent le même générateur generator_prior.

// ... (Inclure les corps de priorNormal et priorLogNormal ici) ...

// --- 4. Fonction de Partition (Dirichlet Process) ---

PriorPartitionResult priorPartition(const Matrix& X, const Matrix& IDH, const Matrix& theta,
                                   Vector& SS_init, Vector& CS_init, Vector& mu_init, Vector& alpha_init,
                                   int m_neal, double b_mu, double b_alpha, double M) {
    // NOTE: C'EST LA FONCTION LA PLUS COMPLEXE ET LA PLUS LONGUE.
    // Elle nécessite une gestion minutieuse de l'indexation, des allocations/suppressions
    // dynamiques de vecteurs (np.append, np.delete), et des comparaisons de clusters.
    
    PriorPartitionResult result;
    // ... implémentation complète nécessaire ici ...
    
    // Placeholder pour le résultat
    result.Ncl = 1;
    result.SS = SS_init;
    result.CS_r = CS_init;
    result.mu_r = mu_init;
    result.alpha_r = alpha_init;
    
    return result;
}