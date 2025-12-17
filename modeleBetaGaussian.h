// modele.h

#ifndef MODELE_H
#define MODELE_H
#include <vector>
#include <cmath>
#include "fonctions.h"
#include "prior.h"
#include "fullConditional.h"
#include "sampling.h"
#include "usefullFunctions.h"
#include "matricesCalculations.h"

// Définitions de types pour la lisibilité
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Structure pour le résultat de la fonction principale
struct BetaGaussienResult {
    Matrix mu_save;     // Matrice des valeurs prédites (p x n_sample)
    Matrix crps_sample; // Échantillons pour le CRPS (n_sample x p)
    int n_sample_used;
    Vector IDHPre;
};

// Ajouté à full_conditional.h ou modele.h

struct HyperParameters
{
    // Valeurs par défaut extraites de beta_gaussien(...):
    double b_mu      = 0.25;    // Hyperparamètre Bêta de mu
    double a_xi      = -1.0;    // Borne inférieure du prior Uniforme pour xi
    double lambda_0  = 0.1;     // Paramètre de régularisation LASSO (MCMC Wishart)
    double M         = 1.0;     // Paramètre du processus de Dirichlet (cohesion)
    // Paramètres initialisés dans la fonction (a_sigma_1=0.5, b_sigma_1=2, etc.)
    double a_sigma_1 = 0.5;     // Prior de forme/taux Gamma pour sigma_1
    double b_sigma_1 = 2.0;     // Prior de forme/taux Gamma pour sigma_1
    double a_sigma_2 = 0.5;     // Prior de forme/taux Gamma pour sigma_2
    double b_sigma_2 = 2.0;     // Prior de forme/taux Gamma pour sigma_2
    double b_alpha   = 0.5;     // Paramètre du prior Gamma pour alpha
    int m_neal=3;  // deeply of the NealAgorithm
    double neta = 1.0;   //
};


/**
 * Fonction principale MCMC Beta-Gaussien.
 * Orchestre le chargement, l'initialisation, le burn-in et l'échantillonnage final.
 */

BetaGaussienResult betaGaussien(
    const std::string& data1,
    const std::string& data2,
    std::size_t burn_in   = 1000,
    std::size_t n_sample  = 1000,
    HyperParameters defaults = HyperParameters{},  // ou = {}
    bool State = true
);



#endif // MODELE_H

