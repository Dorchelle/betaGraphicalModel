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
        double shape = 0.5*meanSS(theta,SS,j);
        double rate = 1.0 / b;

        std::gamma_distribution<double> gamma_dist(shape, rate);
        alpha[j] = gamma_dist(generator_prior);
        
        if (alpha[j] < 0.001) alpha[j] = 0.001;
    }
    return alpha;
}


Vector priorBeta(const Matrix& lambda_1, double b, const Vector& SS) {
    Vector Cs = calculateCs(SS);
    int Ncl = Cs.size();
    Vector alpha(Ncl);
    
    for (int j = 0; j < Ncl; ++j) {
        // Placeholder pour les paramètres de forme
        double a_shape = meanSS(lambda_1,SS,j); 
        double b_shape = b;
        alpha[j] = sampleBeta(a_shape, b_shape);
        if (alpha[j] < 0.001) alpha[j] = 0.001;
        if (alpha[j] > 0.99999) alpha[j] = 0.99999;
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

/**
 * Traduction de Prior_log_normal(B, sigma, xi, T1)
 * Échantillonne la matrice theta (variables latentes) sous une hypothèse Log-Normale.
 * @return La matrice theta (p x T1) échantillonnée (échelle normale).
 */
Matrix priorLogNormal(const Matrix& B, double sigma, const Vector& xi, int T1) {
    size_t p = B.size();
    if (p == 0 || T1 <= 0) return Matrix();
    
    // Initialisation de theta à 1.0 (log(theta)=0.0) comme dans le Python
    Matrix theta(p, Vector(T1, 1.0)); 
    Matrix log_theta(p, Vector(T1, 0.0)); // Matrice de travail sur l'échelle log
    
    // Distribution Normale pour l'échantillonnage log(theta)
    
    // --- 1. Échantillonnage de la première période (t=0) ---
    // La Log-Vraisemblance est ici traitée comme le Prior du processus AR(1)
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        double B_ii = B[i][i];
        if (std::abs(B_ii) < 1e-12) continue; // Éviter la division par zéro

        // a) Calcul de la moyenne du log (moy)
        // moy= -np.sum(B[i,:]*np.log(theta[:,0]))/B[i,i] + np.log(theta[i,0])
        double sum_B_psi = 0.0;
        for (size_t k = 0; k < p; ++k) {
            sum_B_psi += B[i][k] * log_theta[k][0]; // Initialement log(theta)=0
        }
        
        double var = sigma / B_ii;
        double log_moy = (-sum_B_psi / B_ii) + log_theta[i][0];
        
        // b) Échantillonner log(theta) ~ N(log_moy, var)
        std::normal_distribution<double> current_normal(log_moy, std::sqrt(var));
        log_theta[i][0] = current_normal(generator_prior);

        // c) Log-Normalisation (theta = exp(log_theta))
        theta[i][0] = std::exp(log_theta[i][0]);
    }

    // --- 2. Échantillonnage des périodes suivantes (t=1 à T1-1) ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            double B_ii = B[i][i];
            if (std::abs(B_ii) < 1e-12) continue;
            
            double var = sigma / B_ii;

            // a) Calcul de la moyenne Log (moy_theta) avec dépendance temporelle xi
            // moy_theta = np.sum(B[i,:]*(xi*psi[:,t] - psi[:,t+1]))/B[i,i] +psi[i,t+1]
            double sum_B_xi_log_theta = 0.0;
            for (size_t k = 0; k < p; ++k) {
                // Utiliser log_theta pour les valeurs précédentes, car elles sont échantillonnées
                sum_B_xi_log_theta += B[i][k] * (xi[k] * log_theta[k][t] - log_theta[k][t+1]);
            }
            
            double log_moy_theta = (sum_B_xi_log_theta / B_ii) + log_theta[i][t + 1];
            
            // b) Échantillonner log(theta) ~ N(log_moy, var)
            std::normal_distribution<double> current_normal(log_moy_theta, std::sqrt(var));
            log_theta[i][t + 1] = current_normal(generator_prior);
            
            // c) Log-Normalisation
            theta[i][t + 1] = std::exp(log_theta[i][t + 1]);
        }
    }
    
    return theta;
}

/// Normal distrubtion 

Matrix riorNormal(const Matrix& B, double sigma, const Vector& xi, int T1) {
    size_t p = B.size();
    if (p == 0 || T1 <= 0) return Matrix();
    
    // psi est initialisée à zéro (np.zeros((p,T1)))
    Matrix psi(p, Vector(T1, 0.0)); 
    
    // --- 1. Échantillonnage de la première période (t=0) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        double B_ii = B[i][i];
        if (std::abs(B_ii) < 1e-12 || sigma <= 0.0) continue; 

        // a) Calcul de la moyenne (moy)
        // moy= -np.sum(B[i,:]*psi[:,0])/B[i,i] + psi[i,0]
        double sum_B_psi = 0.0;
        for (size_t k = 0; k < p; ++k) {
            // psi[:,0] correspond à psi[k][0]
            sum_B_psi += B[i][k] * psi[k][0]; 
        }
        
        double var = sigma / B_ii;
        double moy = (-sum_B_psi / B_ii) + psi[i][0];
        
        // b) Échantillonner psi[i,0] ~ N(moy, var)
        std::normal_distribution<double> current_normal(moy, std::sqrt(var));
        psi[i][0] = current_normal(generator_prior);
    }

    // --- 2. Échantillonnage des périodes suivantes (t=1 à T1-1) ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            double B_ii = B[i][i];
            if (std::abs(B_ii) < 1e-12 || sigma <= 0.0) continue;
            
            double var = sigma / B_ii;

            // a) Calcul de la moyenne (moy_1) avec dépendance temporelle xi
            // moy_1 = np.sum(B[i,:]*(xi*psi[:,t] - psi[:,t+1]))/B[i,i] +psi[i,t+1]
            double sum_B_term = 0.0;
            for (size_t k = 0; k < p; ++k) {
                // Terme : xi[k] * psi[k,t] - psi[k,t+1]
                // Au premier passage de la boucle i, psi[k,t+1] est encore 0.
                sum_B_term += B[i][k] * (xi[k] * psi[k][t] - psi[k][t + 1]);
            }
            
            double moy_1 = (sum_B_term / B_ii) + psi[i][t + 1];
            
            // b) Échantillonner psi[i,t+1] ~ N(moy_1, var)
            std::normal_distribution<double> current_normal(moy_1, std::sqrt(var));
            psi[i][t + 1] = current_normal(generator_prior);
        }
    }
    
    return psi;
}