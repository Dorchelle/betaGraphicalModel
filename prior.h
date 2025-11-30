// prior.h

#ifndef PRIOR_H
#define PRIOR_H

#include "fonctions.h" // Nécessaire pour Matrix, Vector, meanSS, etc.
#include <random>      // Pour les distributions aléatoires
#include <vector>
#include <cmath>
#include <algorithm>   // Pour std::max_element

// Déclaration du générateur (doit être défini dans prior.cpp ou un fichier d'utilitaires)
// Pour l'instant, nous supposons qu'il sera défini dans prior.cpp.

// --- 1. Priors Simples (Uniforme, Graphe) ---
double priorSigma(double a_sigma, double b_sigma);
Vector priorXI(double a_xi, double b_xi, int p);
Matrix adjacencyMatrix(int p); 
Vector calculateCs(const Vector& SS);
// --- 2. Fonctions d'Échantillonnage de Priors (Gamma, Beta) ---
// Note: Ces fonctions supposent que meanSS retourne un vecteur de taille Ncl
Vector priorGamma(const Matrix& theta, double b, const Vector& SS);
Vector priorBeta(const Matrix& lambda_1, double b, const Vector& SS);
Vector priorGamma1(const Matrix& theta, double b, const Vector& SS);
Vector priorBeta1(const Matrix& lambda_1, double b, const Vector& SS);

// --- 3. Échantillonneurs de Processus Stochastiques ---
Matrix priorNormal(const Matrix& B, double sigma, double xi, int T1);
Matrix priorLogNormal(const Matrix& B, double sigma, double xi, int T1);

// --- 4. Fonction de Partition (Dirichlet Process) ---
struct PriorPartitionResult {
    int Ncl;
    Vector CS_r;
    Vector SS;
    Vector mu_r;
    Vector alpha_r;
};

PriorPartitionResult priorPartition(const Matrix& X, const Matrix& IDH, const Matrix& theta,
                                   Vector& SS_init, Vector& CS_init, Vector& mu_init, Vector& alpha_init,
                                   int m_neal, double b_mu, double b_alpha, double M);
                                   
#endif // PRIOR_H