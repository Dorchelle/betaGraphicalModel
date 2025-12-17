// full_conditional.h

#ifndef FULLCONDITIONAL_H
#define FULLCONDITIONAL_H

#include "fonctions.h"
#include "prior.h" // Si des fonctions de prior sont directement appelées
#include "usefullFunctions.h"
#include "GraphicalFunctions.h"
#include "matricesCalculations.h"
/**
 * Structure de retour pour la fonction de partition
 */
struct PartitionResultPG {
    int Ncl;
    Vector CS_r;
    Vector SS;
    Vector r_r;
    Vector psi_r;
};
// Définitions de types pour la lisibilité
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
Matrix fullLambda(const Vector& mu, double mu_b, const Vector& xi, const Matrix& B, Matrix& lambda_1, double sigma, const Vector& SS, int T1);
// --- 1. Full Conditionals pour le Modèle Beta (lambda, theta) ---
Matrix fullTheta(const Vector& alpha, double alpha_b, const Vector& xi, const Matrix& B, Matrix& theta, double sigma, const Vector& SS, int T1);
// --- 2. Full Conditionals pour les Priors de Cluster (alpha, mu) ---
Vector fullAlpha(const Matrix& IDH, const Matrix& theta, Vector& alpha, double b_alpha, const Vector& mu, const Vector& SS);
Vector fullMu(const Matrix& IDH, const Matrix& lambda_1, Vector& mu, double b_mu, const Vector& alpha, const Vector& SS);
// --- 3. Full Conditionals pour les Paramètres du Processus (xi, sigma) ---
Vector fullXi(Vector& xi, double a_xi, double b_xi, const Matrix& B, const Matrix& lambda_1, double sigma_1);
double fullSigma(double a_sigma, double b_sigma, double sigma, const Matrix& lambda_1, int T1, int p, const Matrix& B, const Vector& xi);
// --- 4. Full Partition (Beta et Poisson-Gamma) ---
// Note: Ces fonctions sont très longues et nécessitent une traduction minutieuse.
// --- 5. Full Conditionals pour le Modèle Poisson-Gamma (r, psi, beta) ---
Vector fullR(const Matrix& Y, const Vector& e, Vector& r, double b_r, const Matrix& mu, const Matrix& theta, const Vector& SS);
// ... (autres fonctions du modèle Poisson-Gamma) ...
// Full Partition (Neal's Algorithm pour l'échantillonnage des clusters)
 PriorPartitionResult fullPartition(
    double b, double M, Vector& SS_in, Vector& CS_in, const Matrix& IDH, 
    Vector& mu_in, double b_mu, Vector& alpha_in, double b_alpha, 
    const Matrix& lambda_1, const Matrix& theta, const Matrix& X, int m_neal) ;
double vectorDotProduct(const Vector& A, const Vector& B);
Vector fullBetaPoll(const Vector& r, const Vector& e, const Matrix& X, Vector& beta, const Vector& sigma_beta, const Vector& psi, const Vector& y, const Vector& SS, int T1);
Matrix fullLambdaPG(const Vector& r, double r_b, double xi, const Matrix& B, Matrix& lambda_PG, double sigma, const Vector& SS, const Vector& Y_flat, const Matrix& W_matrix);
Vector fullPsiGamma(const Vector& Y, const Vector& e, const Matrix& X, const Vector& r, const Vector& beta, Vector& psi, const Matrix& lambda_1, double sigma_psi, const Vector& SS);
Vector fullSigmaPoll(const Vector& a_sigma_beta, const Vector& b_sigma_beta, const Vector& beta, Vector& sigma_beta);// ...
Vector fullR(const Vector& Y, const Vector& e, Vector& r, double b_r, const Vector& mu, const Matrix& theta, const Vector& SS);
PartitionResultPG fullPartitionPD(double b, double M, Vector& SS, Vector& CS, const Vector& Y, const Vector& e, const Matrix& X, const Vector& beta, Vector& psi, double sigma_psi, const Matrix& lambda_1, Vector& r, double b_r, const Matrix& theta, int m_neal = 3);
Matrix fullRPoll(const Vector& E, Matrix& r, const Vector& beta, const Vector& xi, const Matrix& B, const Matrix& mu, double sigma, const Vector& Y, const Vector& SS, int T1);
Matrix fullPsiPoll(const Vector& E, const Matrix& X, Matrix& psi, const Vector& beta, const Vector& xi, const Matrix& B, const Matrix& r, double sigma, const Vector& Y, const Vector& SS, int T1);
// ...
// ...
#endif // FULL_CONDITIONAL_H