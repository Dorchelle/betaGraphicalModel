#include <vector>
// full_conditional.cpp
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "fonctions.h"
#include "GraphicalFunctions.h"
#include "fullConditional.h"
#include "sampling.h"
#include "usefullFunctions.h"
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Déclaration d'un générateur aléatoire (supposé défini dans prior.cpp ou un fichier commun)
extern std::mt19937 generator_prior; // Utilisation du générateur du module prior
// Note: Le C++ a besoin que 'generator_prior' soit accessible ici.
// Déclaration d'un générateur aléatoire
extern double sampleBeta(double alpha, double beta); 
extern double logLikhoodBetaJ(const Matrix& X, const Vector& ALPHA, double mu_param, const Vector& S, int j);
extern Vector calculateCs(const Vector& SS);
extern Matrix XXt(const Vector& X); 
extern Matrix multiplyMatrices(const Matrix& A, const Matrix& B);
//extern double vectorDotProduct(const Vector& A, const Vector& B);
extern double trace(const Matrix& M);
extern double numerateur(const Vector& Cs, int j,double b_j, int l);
extern double denominateur(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l, double b_j);
extern double denominateurPoll(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l, double b_j);
extern double poissonGammaLog(int y, double alpha, double beta);
extern std::pair<int, int> isValueInMatrix(const Vector& SS, int i, int t, int p);
extern int sampleDiscrete(const Vector& probabilities); // Échantillonne multinomial
extern double sampleGamma(double shape, double scale);
double sampleBeta(double alpha, double beta);
extern size_t countNonzeroClusterJ(const Vector& SS, int j);
//extern void deleteLastElement(Vector& vec); // Supprime le dernier élément
// --- Fonction Utilitaires pour Logarithmes et Aléatoire ---
// Équivalent à np.log(st.uniform.rvs(0,1))
// ... (Ajout des autres fonctions dans l'ordre) ...


/**
 * Traduction de Full_theta(...)
 * Échantillonne la variable latente theta par Metropolis-Hastings (Log-Normal).
 */
Matrix fullTheta(const Vector& alpha, double alpha_b, const Vector& xi, const Matrix& B, Matrix& theta, double sigma, const Vector& SS, int T1) {
    size_t p = B.size();
    
    //if (p == 0 || T1 == 0) return theta;
    Matrix theta_star_mat = theta; // Matrice pour stocker la proposition complète

    // --- Période t=0 (Initialisation) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        double theta_i0 = theta[i][0];
        size_t flat_idx = i + t * p;
        
        try {
            double B_ii = B[i][i];
            if (std::abs(B_ii) < 1e-12) continue;
            
            double var_1 = sigma / B_ii;
            if (var_1 <= 0.0) continue;
            
            // 1. Proposer theta_star_i par Log-Normale
            double log_mu_prop = std::log(theta_i0); 
            std::normal_distribution<double> normal_prop(log_mu_prop, std::sqrt(var_1));
            
            double theta_star_i = std::exp(normal_prop(generator_prior));
            if (theta_star_i < 0.0001) theta_star_i = 0.0001;
            
            // 2. Calculer le ratio d'acceptation tp (Log-Alpha)
            double tp = 0.0;
            double log_theta_star_i = std::log(theta_star_i);
            double log_theta_i0 = std::log(theta_i0);

            // Terme 1: Prior Gaussien Temporel (t=0) - Log-Normal ratio
            // np.sum(B[i,:]*log(theta[:,0]))
            double sum_B_log_theta = 0.0;
            for (size_t k = 0; k < p; ++k) {
                sum_B_log_theta += B[i][k] * std::log(theta[k][0]);
            }
            
            tp += -0.5 * B_ii / sigma;
            tp *= (log_theta_star_i - log_theta_i0);
            tp *= (-2.0 * sum_B_log_theta / B_ii + log_theta_star_i + log_theta_i0);

            // Terme 2: Prior de Cluster (Log P(theta*)/P(theta) - Log(Proposal))
            size_t j = static_cast<size_t>(std::round(SS[flat_idx])); 
            if (j >= alpha.size()) continue; // Vérification du cluster
            
            double diff_j = theta_star_i - theta_i0;

            // tp+=-1*math.lgamma(theta_star[i]) +math.lgamma(theta[i,0])
            tp += -std::lgamma(theta_star_i) + std::lgamma(theta_i0);
            
            // tp+=diff_j*(0.5*np.log(alpha[j]) +np.log(alpha_b))
            tp += diff_j * (0.5 * std::log(alpha[j]) + std::log(alpha_b));

            // 3. Décision d'Acceptation
            if (tp > logUniformRvs()) {
                theta[i][0] = theta_star_i;              
            }
        } catch (const std::exception& e) {
            std::cerr << "Erreur Full_theta (t=0, i=" << i << "): " << e.what() << ". Rejet." << std::endl;
        }                                                                             
    }
    
    // --- Périodes t > 0 ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            size_t t_plus_1 = t + 1;
            size_t flat_idx = i + t_plus_1 * p;
            double theta_i_tplus1 = theta[i][t_plus_1];
            
            try {
                 double B_ii = B[i][i];
                 if (std::abs(B_ii) < 1e-12) continue;
                 double var_1 = sigma / B_ii;
                 if (var_1 <= 0.0) continue;
                 
                 // 1. Calcul de la moyenne Log-Normale (moy_1)
                 // moy_1 = np.sum(B[i,:]*(0.5*xi*log(theta[:,t]) -0.5*log(theta[:,t+1])))/B[i,i] +log(theta[i,t+1])
                 double sum_B_log_theta_pred = 0.0;
                 for (size_t k = 0; k < p; ++k) {
                     // Terme : xi*log(theta[:,t]) - log(theta[:,t+1])
                     sum_B_log_theta_pred += B[i][k] * (xi[k] * std::log(theta[k][t]) - std::log(theta[k][t + 1]));
                 }
                 double moy_1 = sum_B_log_theta_pred * 0.5 / B_ii + std::log(theta_i_tplus1);
                 
                 // 2. Proposition theta_star_i ~ Log-Normale
                 std::normal_distribution<double> normal_prop(moy_1, std::sqrt(var_1));
                 double theta_star_i = std::exp(normal_prop(generator_prior));
                 if (theta_star_i < 0.00001) theta_star_i = 0.00001;

                 // 3. Calcul du ratio d'acceptation (tp)
                 double tp = 0.0;
                 double log_theta_star_i = std::log(theta_star_i);
                 double log_theta_i_tplus1 = std::log(theta_i_tplus1);
                 
                 // Terme 1: Prior Temporel (t>0)
                 double sum_B_current = 0.0;
                 for (size_t k = 0; k < p; ++k) {
                     // Terme : log(theta[:,t+1]) - xi*log(theta[:,t])
                     sum_B_current += B[i][k] * (std::log(theta[k][t_plus_1]) - xi[k] * std::log(theta[k][t]));
                 }
                 
                 // Ligne C++ : tp=tp*(0.25*(log(theta_star[i])-log(theta[i,t+1]))  +np.sum(B[i,:]*(-0.5*xi*log(theta[:,t]) +0.5*log(theta[:,t+1])))/B[i,i])
                 tp += -0.5 * B_ii / sigma;
                 tp *= (log_theta_star_i - log_theta_i_tplus1);
                 tp *= (0.25 * (log_theta_star_i - log_theta_i_tplus1) + sum_B_current * 0.5 / B_ii);

                 // Terme 2: Prior de Cluster (Log P(theta*)/P(theta))
                 size_t j = static_cast<size_t>(std::round(SS[flat_idx])); 
                 if (j >= alpha.size()) continue; 
                 
                 double diff_j = theta_star_i - theta_i_tplus1;
                 
                 tp += -std::lgamma(theta_star_i) + std::lgamma(theta_i_tplus1);
                 tp += diff_j * (0.5 * std::log(alpha[j]) + std::log(alpha_b));

                 // 3. Décision d'Acceptation
                 if (tp > logUniformRvs()) {
                     theta[i][t_plus_1] = theta_star_i;              
                 }
            } catch (const std::exception& e) {
                std::cerr << "Erreur Full_theta (t>0, i=" << i << ", t=" << t_plus_1 << "): " << e.what() << ". Rejet." << std::endl;
            }
        } // Fin de la boucle i
    } // Fin de la boucle t
    
    return theta;
}



//Matrix fullLambda(const Vector& mu, double mu_b, Vector xi, const Matrix& B, Matrix& lambda_1, double sigma, const Vector& SS, int T1);

Matrix fullLambda(const Vector& mu, double mu_b, const Vector& xi, const Matrix& B, Matrix& lambda_1, double sigma, const Vector& SS, int T1) {
    size_t p = B.size();
    if (p == 0) return lambda_1;

    // --- Période t=0 (Initialisation) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        double lambda_i0 = lambda_1[i][0];
        size_t flat_idx = i + t * p;
        
        try {
            double B_ii = B[i][i];
            if (std::abs(B_ii) < 1e-12) continue; // Division par zéro
            double var_1 = sigma / B_ii;
            if (var_1 <= 0.0) continue;
            
            // 1. Proposer lambda_star_i par Log-Normale
            // Log-Normal centrée sur log(lambda_i0) avec variance var_1
            double log_mu_prop = std::log(lambda_i0); 
            std::normal_distribution<double> normal_prop(log_mu_prop, std::sqrt(var_1));
            
            double lambda_star_i = std::exp(normal_prop(generator_prior));
            if (lambda_star_i < 0.0001) lambda_star_i = 0.0001;

            // 2. Calculer le ratio d'acceptation tp (Log-Alpha)
            double tp = 0.0;
            double log_lambda_star_i = std::log(lambda_star_i);
            double log_lambda_i0 = std::log(lambda_i0);

            // Terme 1: Prior Gaussien Temporel (t=0)
            double sum_B_log_lambda = 0.0;
            for (size_t k = 0; k < p; ++k) {
                sum_B_log_lambda += B[i][k] * std::log(lambda_1[k][0]); // B[i,:]*log(lambda[:,0])
            }
            
            tp += -0.5 * B_ii / sigma;
            tp *= (log_lambda_star_i - log_lambda_i0);
            tp *= (-2.0 * sum_B_log_lambda / B_ii + log_lambda_star_i + log_lambda_i0);

            // Terme 2: Prior Bêta du Cluster (Log P(lambda*)/P(lambda) - Log(Proposal))
            size_t j = static_cast<size_t>(std::round(SS[flat_idx])); 
            if (j >= mu.size()) continue; 
            
            double diff_j = lambda_star_i - lambda_i0;

            // Termes de vraisemblance Bêta simplifiés
            tp += diff_j * std::log(mu[j]); 
            tp += std::lgamma(lambda_star_i + mu_b) - std::lgamma(lambda_i0 + mu_b);
            tp += -std::lgamma(lambda_star_i) + std::lgamma(lambda_i0);
            
            // 3. Décision d'Acceptation
            if (tp > logUniformRvs()) {
                lambda_1[i][0] = lambda_star_i;              
            }
        } catch (const std::exception& e) {
            std::cerr << "Erreur Full_lambda (t=0, i=" << i << "): " << e.what() << ". Rejet." << std::endl;
        }                                                                             
    }
    
    // --- Périodes t > 0 ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            size_t t_plus_1 = t + 1;
            size_t flat_idx = i + t_plus_1 * p;
            double lambda_i_tplus1 = lambda_1[i][t_plus_1];
            
            try {
                 double B_ii = B[i][i];
                 if (std::abs(B_ii) < 1e-12) continue;
                 double var_1 = sigma / B_ii;
                 if (var_1 <= 0.0) continue;
                 
                 // 1. Proposition Log-Normale
                 // Moyenne Log: moy_1 = np.sum(B[i,:]*(0.5*xi*log(lambda[:,t]) -0.5*log(lambda[:,t+1])))/B[i,i] +log(lambda[i,t+1])
                 double sum_B_log_lambda_pred = 0.0;
                 for (size_t k = 0; k < p; ++k) {
                     // Utilise xi[k] * log(lambda[k,t]) - log(lambda[k,t+1]) pour la somme
                     sum_B_log_lambda_pred += B[i][k] * (xi[k] * std::log(lambda_1[k][t]) - std::log(lambda_1[k][t + 1]));
                 }
                 double moy_1 = sum_B_log_lambda_pred * 0.5 / B_ii + std::log(lambda_i_tplus1); // Note: 0.5 facteur oublié
                 
                 std::normal_distribution<double> normal_prop(moy_1, std::sqrt(var_1));
                 double lambda_star_i = std::exp(normal_prop(generator_prior));
                 if (lambda_star_i < 0.0001) lambda_star_i = 0.0001;

                 // 2. Calcul du ratio d'acceptation (tp)
                 double tp = 0.0;
                 double log_lambda_star_i = std::log(lambda_star_i);
                 double log_lambda_i_tplus1 = std::log(lambda_i_tplus1);
                 
                 // Terme 1: Prior Temporel (t>0)
                 double sum_B_current = 0.0;
                 for (size_t k = 0; k < p; ++k) {
                     sum_B_current += B[i][k] * (-0.5 * xi[k] * std::log(lambda_1[k][t]) + 0.5 * std::log(lambda_1[k][t + 1]));
                 }
                 
                 // Ligne C++ : tp=tp*(0.25*(log(lambda_star[i])-log(lambda_1[i,t+1]))  +np.sum(B[i,:]*(-0.5*xi*log(lambda[:,t]) +0.5*log(lambda[:,t+1])))/B[i,i])
                 tp += -0.5 * B_ii / sigma;
                 tp *= (log_lambda_star_i - log_lambda_i_tplus1);
                 tp *= (0.25 * (log_lambda_star_i - log_lambda_i_tplus1) + sum_B_current / B_ii);

                 // Terme 2: Prior Bêta du Cluster
                 size_t j = static_cast<size_t>(std::round(SS[flat_idx])); 
                 if (j >= mu.size()) continue; 
                 
                 double diff_j = lambda_star_i - lambda_i_tplus1;
                 
                 tp += diff_j * std::log(mu[j]); 
                 tp += std::lgamma(lambda_star_i + mu_b) - std::lgamma(lambda_i_tplus1 + mu_b);
                 tp += -std::lgamma(lambda_star_i) + std::lgamma(lambda_i_tplus1);

                 // 3. Décision d'Acceptation
                 if (tp > logUniformRvs()) {
                     lambda_1[i][t_plus_1] = lambda_star_i;              
                 }
            } catch (const std::exception& e) {
                std::cerr << "Erreur Full_lambda (t>0, i=" << i << ", t=" << t_plus_1 << "): " << e.what() << ". Rejet." << std::endl;
            }
        } // Fin de la boucle i
    } // Fin de la boucle t
    
    return lambda_1;
}



Vector fullAlpha(const Matrix& IDH, const Matrix& theta, Vector& alpha, double b_alpha, const Vector& mu, const Vector& SS) {
    if (theta.empty()) return alpha;
    
    size_t p = theta.size(); // Nombre de régions
    size_t T1 = theta[0].size(); // Nombre de périodes
    size_t p_T1 = p * T1; // Taille du vecteur aplati

    if (SS.size() != p_T1) {
        throw std::invalid_argument("fullAlpha: La taille de SS ne correspond pas à la matrice theta.");
    }

    // 1. Calculer les tailles de cluster (Cs) et le nombre de clusters (Ncl)
    Vector Cs = calculateCs(SS);
    int Ncl = Cs.size();

    // 2. Calculer les statistiques suffisantes (somme des log(theta)) pour chaque cluster
    // Créer un vecteur de sommes initialisé à zéro (une somme par cluster)
    Vector sum_log_theta_j(Ncl, 0.0);

    for (size_t i_flat = 0; i_flat < p_T1; ++i_flat) {
        // Calculer les indices matriciels (i, t)
        size_t t = i_flat / p;
        size_t i = i_flat % p;
        
        // Label du cluster
        size_t j = static_cast<size_t>(std::round(SS[i_flat])); 

        if (j < Ncl && i < p && t < T1) {
            // Suffisant Statistique: Somme des log(theta_{i,t})
            sum_log_theta_j[j] += std::log(theta[i][t]);
        }
    }
    
    // 3. Mettre à jour et échantillonner alpha_j
    for (int j = 0; j < Ncl; ++j) {
        double theta_j=meanSS(theta,SS,j);
        double n_j = Cs[j];
        if (n_j < 1e-9) continue; // Cluster vide, on ignore
        int it = 0;
        double alpha_star = 1.0; 
        double new_shape = 0.25*theta_j+0.25*alpha[j];
        
        // Boucle pour garantir que mu_star est dans (0.001, 0.999)
        while (it < 10) {
            std::gamma_distribution<double> gamma_dist(new_shape, b_alpha); // (shape, 1/scale)
                alpha_star= gamma_dist(generator_prior);
            
            it++;
            if (alpha_star > 0.00001 ) {
                break;
            }
        }
         if (alpha_star <= 0.00001 ) {
                alpha_star=0.00001 ;
            }
        double tp=0;
        double betaj=alpha[j]*(1-mu[j]);
        double betaj_star=alpha_star*(-1-mu[j]);
        double alphaj=alpha[j]*mu[j];
        double alphaj_star1=alpha_star*mu[j];
        tp=logLikhoodBetaJ(IDH,alphaj_star1,betaj_star,SS,j)-logLikhoodBetaJ(IDH,alphaj,betaj,SS,j);
        tp += (0.5*n_j * theta_j - n_j) * (std::log(alpha_star) - std::log(alpha[j]));
        tp+=b_alpha*(alpha_star-alpha[j])*(1-1/n_j);
        tp += (0.25*theta_j+0.25*alpha[j]- 1) *( (std::log(alpha_star) - std::log(alpha[j])) +std::log(alpha[j]));
        // 3. Décision d'Acceptation
        if (tp > logUniformRvs()) {
            alpha[j] = alpha_star;
        }
    }


    return alpha;
}

Vector fullMu(const Matrix& IDH,
              const Matrix& lambda_1,
              Vector& mu,
              double b_mu,
              const Vector& alpha,
              const Vector& SS)
{
    // Cas dégénéré
    if (lambda_1.empty() || IDH.empty()) {
        return mu;
    }

    const size_t p   = lambda_1.size();        // nb de lignes (régions)
    const size_t T1  = lambda_1[0].size();     // nb de périodes
    const size_t nSS = SS.size();

    // Vérifs dimensionnelles de base
    for (size_t i = 0; i < p; ++i) {
        if (lambda_1[i].size() != T1) {
            throw std::invalid_argument("fullMu: lambda_1 a des lignes de tailles différentes.");
        }
    }
    if (IDH.size() != p || IDH[0].size() != T1) {
        throw std::invalid_argument("fullMu: dimensions de IDH incompatibles avec lambda_1.");
    }
    if (nSS != p * T1) {
        throw std::invalid_argument("fullMu: SS.size() doit être égal à p * T1.");
    }
    if (alpha.size() == 0) {
        return mu;
    }

    // 1. Tailles de clusters Cs et nombre de clusters Ncl
    Vector Cs = calculateCs(SS);
    const size_t Ncl = Cs.size();

    if (mu.size() < Ncl) {
        mu.resize(Ncl, 0.5);  // valeur par défaut si nouveau cluster
    }
    if (alpha.size() < Ncl) {
        throw std::invalid_argument("fullMu: alpha plus court que Cs/mu.");
    }

    // 2. Pré-calcul des moyennes lambda_j pour tous les clusters en une seule passe
    // lambda_j = moyenne des lambda_1[i,t] tel que SS[i + t*p] == j
    Vector lambda_mean(Ncl, 0.0);

    for (size_t t = 0; t < T1; ++t) {
        const size_t offset = t * p;
        for (size_t i = 0; i < p; ++i) {
            const size_t idx = offset + i;
            int lab = static_cast<int>(SS[idx]);
            if (lab < 0 || static_cast<size_t>(lab) >= Ncl) {
                continue; // on ignore les labels invalides
            }
            lambda_mean[static_cast<size_t>(lab)] += lambda_1[i][t];
        }
    }

    for (size_t j = 0; j < Ncl; ++j) {
        if (Cs[j] > 0.0) {
            lambda_mean[j] /= Cs[j];
        }
    }

    // 3. Boucle MH sur chaque cluster j
    for (size_t j = 0; j < Ncl; ++j) {
        const double n_j = Cs[j];
        if (n_j < 1.0) {
            continue; // cluster vide
        }

        const double mu_j_current = mu[j];
        const double lambda_j     = lambda_mean[j];
        const double alpha_j      = alpha[j];

        // --- 3.1. Tirage de la proposition mu_star ~ Beta(a_prop, b_mu) ---
        const double a_prop = 0.5 * lambda_j + 0.5 * mu_j_current;

        double mu_star = 0.0;
        int it = 0;
        while (it < 10) {
            mu_star = sampleBeta(a_prop, b_mu);
            ++it;
            if (mu_star >= 1e-5 && mu_star <= 0.99999) {
                break;
            }
        }
        if (mu_star < 1e-4)   mu_star = 1e-4;
        if (mu_star > 0.9999) mu_star = 0.9999;

        // Par sécurité, on s'assure aussi que mu_j_current est dans (0,1)
        const double mu_cur_clamped = std::min(0.9999, std::max(1e-4, mu_j_current));

        // --- 3.2. Terme de log-vraisemblance Beta sur IDH ---
        const double beta_j      = alpha_j * (1.0 - mu_cur_clamped);
        const double beta_j_star = alpha_j * (1.0 - mu_star);
        const double alpha_j_cur = alpha_j * mu_cur_clamped;
        const double alpha_j_star= alpha_j * mu_star;

        double tp = 0.0;
        tp += logLikhoodBetaJ(IDH, alpha_j_star, beta_j_star, SS, static_cast<int>(j))
            - logLikhoodBetaJ(IDH, alpha_j_cur, beta_j, SS, static_cast<int>(j));

        // --- 3.3. Terme de prior sur mu ---
        // (n_j*lambda_j - n_j) * (log(mu_star) - log(mu_j))
        // + (n_j*b_mu - n_j) * (log(1-mu_star) - log(1-mu_j))
        tp += (n_j * lambda_j - n_j)
              * (std::log(mu_star) - std::log(mu_cur_clamped));
        tp += (n_j * b_mu - n_j)
              * (std::log(1.0 - mu_star) - std::log(1.0 - mu_cur_clamped));

        // --- 3.4. Terme de proposition (ratio des PDFs Beta) ---
        // Q(mu -> mu_star): Beta(a_prop, b_mu)
        // Q(mu_star -> mu): Beta(a_star_prop, b_mu) avec a_star_prop = 0.5*lambda_j + 0.5*mu_star
        const double a_star_prop = 0.5 * lambda_j + 0.5 * mu_star;

        const double log_pdf_mu_prop_star =
            (a_star_prop - 1.0) * std::log(mu_cur_clamped)
            + (b_mu      - 1.0) * std::log(1.0 - mu_cur_clamped);

        const double log_pdf_mu_star_prop =
            (a_prop - 1.0) * std::log(mu_star)
            + (b_mu  - 1.0) * std::log(1.0 - mu_star);

        tp += log_pdf_mu_prop_star - log_pdf_mu_star_prop;

        // --- 3.5. Acceptation MH ---
        if (tp > logUniformRvs()) {
            mu[j] = mu_star;
        }
    }

    return mu;
}


Vector fullXi(Vector& xi,
              double a_xi,
              double b_xi,
              const Matrix& B,
              const Matrix& lambda_1,
              double sigma_1)
{
    const size_t p = B.size();
    if (p == 0) {
        return xi;
    }

    if (B[0].size() != p) {
        throw std::invalid_argument("fullXi: B doit être une matrice p x p.");
    }
    if (lambda_1.size() != p) {
        throw std::invalid_argument("fullXi: lambda_1.size() doit être p.");
    }
    const size_t T1 = lambda_1[0].size();
    if (T1 < 1) {
        return xi;
    }
    for (size_t i = 0; i < p; ++i) {
        if (lambda_1[i].size() != T1) {
            throw std::invalid_argument("fullXi: toutes les lignes de lambda_1 doivent avoir T1 colonnes.");
        }
    }
    if (xi.size() != p || sigma_1 <= 0.0) {
        throw std::invalid_argument("fullXi: dimensions ou paramètres invalides.");
    }
    if (a_xi >= b_xi) {
        throw std::invalid_argument("fullXi: intervalle [a_xi, b_xi] invalide.");
    }

    // On n'a besoin que de M1[i,i] et M2[i,i] après produit par B.
    // On calcule directement leurs diagonales sans passer par des matrices complètes.

    // Pré-calcul log(lambda_1)
    Matrix log_lambda(p, Vector(T1));
    for (size_t i = 0; i < p; ++i) {
        for (size_t t = 0; t < T1; ++t) {
            const double val = lambda_1[i][t];
            if (val <= 0.0) {
                throw std::runtime_error("fullXi: lambda_1[i,t] doit être > 0 pour le log.");
            }
            log_lambda[i][t] = std::log(val);
        }
    }

    // diagM1[i] = Σ_t Σ_k log_lambda[i][t] * log_lambda[k][t] * B[k][i]
    // diagM2[i] = Σ_t Σ_k log_lambda[i][t+1] * log_lambda[k][t] * B[k][i]
    // (sans le /sigma_1 ; on l'appliquera à la fin)
    Vector diagM1(p, 0.0);
    Vector diagM2(p, 0.0);

    // Boucle sur t = 0..T1-2
    for (size_t t = 0; t + 1 < T1; ++t) {
        // v_k = log_lambda[k][t]
        // On calcule s_i = Σ_k B[k][i] * v_k (produit matrice-vecteur par colonne)
        Vector s(p, 0.0);
        for (size_t k = 0; k < p; ++k) {
            const double v_k = log_lambda[k][t];
            if (v_k == 0.0) continue;
            const Vector& Bk = B[k];
            for (size_t i = 0; i < p; ++i) {
                s[i] += Bk[i] * v_k;
            }
        }

        for (size_t i = 0; i < p; ++i) {
            const double log_t_i  = log_lambda[i][t];
            const double log_tp1_i = log_lambda[i][t + 1];
            const double s_i      = s[i];

            diagM1[i] += log_t_i  * s_i;    // correspond à M1[i,i] * sigma_1
            diagM2[i] += log_tp1_i * s_i;   // correspond à M2[i,i] * sigma_1
        }
    }

    // On a maintenant (sans facteur 1/sigma_1) :
    //   M1[i,i] = diagM1[i] / sigma_1
    //   M2[i,i] = diagM2[i] / sigma_1
    // Et dans ton code original :
    //   tp_final = [ (xi^2 - xi_star^2)*M1[i,i] - 2*(xi - xi_star)*M2[i,i] ] / (2*sigma_1)
    // Donc au total : facteur 1 / (2 * sigma_1^2)
    const double inv_sigma1   = 1.0 / sigma_1;
    const double half_inv_sig_sq = 0.5 * inv_sigma1 * inv_sigma1;

    // Propositions uniformes
    std::uniform_real_distribution<double> uniform_prop(a_xi, b_xi);

    // --- 3. Metropolis-Hastings pour chaque xi_i ---
    for (size_t i = 0; i < p; ++i) {
        const double xi_i_current = xi[i];
        const double xi_i_star    = uniform_prop(generator_prior);

        const double xi_curr_sq = xi_i_current * xi_i_current;
        const double xi_star_sq = xi_i_star    * xi_i_star;

        const double d1 = (xi_curr_sq - xi_star_sq) * diagM1[i];
        const double d2 = -2.0 * (xi_i_current - xi_i_star) * diagM2[i];

        const double log_alpha = (d1 + d2) * half_inv_sig_sq;

        if (log_alpha > logUniformRvs()) {
            xi[i] = xi_i_star;
        }
    }

    return xi;
}

double fullSigma(double a_sigma,
                 double b_sigma,
                 double sigma,
                 const Matrix& lambda_1,
                 int T1,
                 int p,
                 const Matrix& B,
                 const Vector& xi)   // ← idéalement const&, pour éviter une copie
{
    // Cas dégénérés : on ne touche pas à sigma
    if (p == 0) return sigma;
    if (lambda_1.empty() || lambda_1[0].empty()) return sigma;

    // Vérifications de dimensions
    if (lambda_1.size() != static_cast<std::size_t>(p)) {
        throw std::invalid_argument("fullSigma: lambda_1.size() != p.");
    }
    if (lambda_1[0].size() < static_cast<std::size_t>(T1)) {
        throw std::invalid_argument("fullSigma: lambda_1[0].size() < T1.");
    }
    if (B.size() != static_cast<std::size_t>(p) ||
        B[0].size() != static_cast<std::size_t>(p)) {
        throw std::invalid_argument("fullSigma: B n'est pas une matrice p x p.");
    }
    if (xi.size() != static_cast<std::size_t>(p)) {
        throw std::invalid_argument("fullSigma: xi.size() != p.");
    }

    // Nombre de termes temporels (t = 0..T1-2)
    const double N_eff = static_cast<double>(p) * static_cast<double>(T1 - 1);
    
    // --- 1. Calcul de la somme des carrés des résidus : RSS = Σ_t r_t^T B r_t ---
    double RSS = 0.0;

    // vecteur résidu réutilisé pour chaque t
    Vector residu(p, 0.0);

    // Boucle sur le temps
    for (int t = 0; t < T1 - 1; ++t) {
        const int t1 = t + 1;

        // Construire r_i = log λ_{i, t+1} − xi_i * log λ_{i, t}
        for (int i = 0; i < p; ++i) {
            const double log_lambda_t1 = std::log(lambda_1[i][t1]);
            const double log_lambda_t  = std::log(lambda_1[i][t]);
            residu[i] = log_lambda_t1 - xi[i] * log_lambda_t;
        }

        // Calculer r^T B r sans matrices intermédiaires
        double quad = 0.0;
        for (int i = 0; i < p; ++i) {
            const Vector& Bi = B[i];
            double Bi_dot_r = 0.0;
            for (int j = 0; j < p; ++j) {
                Bi_dot_r += Bi[j] * residu[j];
            }
            quad += residu[i] * Bi_dot_r;
        }

        RSS += quad;
    }
    
    // --- 2. Paramètres du postérieur Gamma sur tau = 1/sigma ---
    const double new_shape = a_sigma + N_eff / 2.0;
    const double new_rate  = b_sigma + RSS   / 2.0;

    // --- 3. Échantillon de tau ---
    // std::gamma_distribution(shape, scale) avec scale = 1/rate
    std::gamma_distribution<double> gamma_dist(new_shape, 1.0 / new_rate);
    const double tau_star = gamma_dist(generator_prior);

    if (tau_star < 1e-12) {
        // sécurité numérique
        return 1.0;
    }

    const double sigma_star = 1.0 / tau_star;
    return sigma_star;
}





// Fonctions utilitaires de manipulation de vecteurs
void delete_last_element(Vector& vec) {
    if (!vec.empty()) {
        vec.pop_back();
    }
}


/**
 * Traduction de Full_partition (Neal's Algorithm 8 pour DPMM Beta-Gaussian)
 */



PriorPartitionResult fullPartition(
    double b, double M, Vector& SS_in, Vector& CS_in, const Matrix& IDH, 
    Vector& mu_in, double b_mu, Vector& alpha_in, double b_alpha, 
    const Matrix& lambda_1, const Matrix& theta, const Matrix& X, int m_neal) 
{
    constexpr bool verbose = false;
    if (verbose) {
        std::cout << "~~~~~~~~~~~~~~~~~~~debut partitionnement~~~~~~~~~~~~~~~~~~~~" << std::endl;
    }

    // --- 0. Vérifications de base sur les dimensions ---
    const std::size_t p   = IDH.size();
    if (p == 0) {
        throw std::invalid_argument("fullPartition: IDH vide (p=0).");
    }
    const std::size_t T1  = IDH[0].size();
    if (T1 == 0) {
        throw std::invalid_argument("fullPartition: IDH a 0 colonnes (T1=0).");
    }
    const std::size_t p_T1 = p * T1;

    // Vérifier la cohérence de IDH, lambda_1, theta en (p x T1)
    auto check_matrix_shape = [p, T1](const Matrix& M, const char* name) {
        if (M.size() != p) {
            throw std::invalid_argument(std::string(name) + ": nombre de lignes != p.");
        }
        for (std::size_t i = 0; i < p; ++i) {
            if (M[i].size() != T1) {
                throw std::invalid_argument(
                    std::string(name) + ": nombre de colonnes != T1 à la ligne " + std::to_string(i)
                );
            }
        }
    };

    check_matrix_shape(IDH,      "IDH");
    check_matrix_shape(lambda_1, "lambda_1");
    check_matrix_shape(theta,    "theta");
    //check_matrix_shape(X,        "X");

    // Dimension l = nombre de colonnes de X
    const int l = static_cast<int>(X[0].size());
    if (l <= 0) {
        throw std::invalid_argument("fullPartition: l <= 0 (nb de colonnes de X).");
    }

    // Hyper-param (tel quel)
    double b_j = b;

    // Valeur minimale pour m_neal
    if (m_neal <= 0) m_neal = 3;

    // Alias pour les vecteurs passés par référence
    Vector& SS    = SS_in;   // labels de taille p*T1
    Vector& CS    = CS_in;   // tailles de clusters
    Vector& mu    = mu_in;
    Vector& alpha = alpha_in;

    // Vérifier SS a bien p*T1 éléments
    if (SS.size() != p_T1) {
        throw std::invalid_argument("fullPartition: SS.size() != p*T1.");
    }

    // Vérifier cohérence entre CS, mu, alpha
    if (!(CS.size() == mu.size() && mu.size() == alpha.size())) {
        throw std::invalid_argument("fullPartition: tailles incohérentes entre CS, mu, alpha.");
    }

    // Nombre de clusters actuel
    int Ncl = static_cast<int>(CS.size());

    // --- Buffers réutilisables pour éviter des allocations dans la boucle ---
    Vector prob;          // log-probabilités
    Vector CS1;           // tailles de clusters étendues (existant + m_neal)
    Vector norm_prob;     // probabilités normalisées
    Vector mu_tp(m_neal);     // mu pour nouveaux clusters
    Vector alpha_tp(m_neal);  // alpha pour nouveaux clusters

    // --- Boucle sur tous les points (i, t) ---
    for (std::size_t t = 0; t < T1; ++t) {
        for (std::size_t i = 0; i < p; ++i) {

            const std::size_t flat_index = i + t * p;  // index 0..p*T1-1
            int k0 = static_cast<int>(std::round(SS[flat_index])); // cluster actuel

            // Sanity check: k0 peut être hors [0, Ncl-1] si SS est corrompu
            if (k0 < 0 || k0 >= Ncl) {
                std::cerr << "Warning fullPartition: label k0=" << k0
                          << " hors [0," << (Ncl-1) << "], on saute ce point.\n";
                continue;
            }

            int k_star = 0; // indicateur de groupe perdu

            // --- 1. Retrait de l'élément (i, t) du cluster k0 ---
            if (k0 < static_cast<int>(CS.size()) && CS[k0] > 1.0) {
                // On décrémente simplement la taille
                CS[k0] -= 1.0;
            } else {
                // Cluster k0 est détruit
                int kminus = Ncl - 1;    // dernier cluster (index Ncl-1)
                Ncl -= 1;                // nouveau nombre de clusters = ancien - 1

                if (k0 < Ncl) {
                    // On permute le label kminus -> k0
                    for (std::size_t l_idx = 0; l_idx < p_T1; ++l_idx) {
                        int lab = static_cast<int>(std::round(SS[l_idx]));
                        if (lab == kminus) {
                            SS[l_idx] = static_cast<double>(k0);
                        }
                    }
                    k_star = 1;
                    SS[flat_index] = static_cast<double>(kminus);

                    // Permutation des paramètres
                    mu[k0]    = mu[kminus];
                    alpha[k0] = alpha[kminus];
                    CS[k0]    = CS[kminus];
                }

                // On supprime le dernier élément (kminus) de CS, mu, alpha
                if (!CS.empty())    CS.pop_back();
                if (!mu.empty())    mu.pop_back();
                if (!alpha.empty()) alpha.pop_back();

                Ncl = static_cast<int>(CS.size());
            }

            // On marque l'élément comme retiré
            SS[flat_index] = -1.0;

            // --- 2. Calcul des probabilités de proposition ---
            const int kminus_current = Ncl;
            const int h_neal         = kminus_current + m_neal;

            if (kminus_current < 0 || h_neal <= 0) {
                throw std::runtime_error("fullPartition: h_neal ou Ncl invalides.");
            }

            // Redimensionner / réinitialiser les buffers
            prob.assign(h_neal, 0.0);
            CS1.assign(h_neal, -1.0);
            norm_prob.assign(h_neal, 0.0);

            // Copier CS dans CS1[0..Ncl-1]
            for (int r = 0; r < kminus_current; ++r) {
                CS1[r] = CS[r];
            }

            // a) Clusters existants (j = 0 .. Ncl-1)
            for (int j = 0; j < kminus_current; ++j) {
                // Cas "avec le point"
                SS[flat_index] = static_cast<double>(j);
                CS1[j] = CS[j] + 1.0;

                double g_xjStar = numerateur(CS1, j, b, l)
                                  - denominateur(SS, X, j, CS1, l, b);

                // Log-likelihood IDH
                const double bj = alpha[j] * (1.0 - mu[j]);
                const double aj = alpha[j] * mu[j];
                const double log_lik_IDH = logBetaPDF(IDH[i][t], aj, bj);

                // Cas "sans le point"
                SS[flat_index] = -1.0;
                CS1[j] = CS[j];

                double g_xj = numerateur(CS1, j, b, l)
                              - denominateur(SS, X, j, CS1, l, b);

                // Log-proba
                prob[j] = (g_xjStar - g_xj)
                          + log_lik_IDH
                          + std::log(CS[j])
                          + std::log(M);
            }

            // b) Nouveaux clusters (j = 0 .. m_neal-1)
            for (int j = 0; j < m_neal; ++j) {

                const int j_new = kminus_current + j;   // index dans prob/CS1

                // Tirage des paramètres du nouveau cluster
                mu_tp[j] = sampleBeta(lambda_1[i][t], b_mu);
                if (mu_tp[j] < 0.001) mu_tp[j] = 0.001;
                if (mu_tp[j] > 0.999) mu_tp[j] = 0.999;

                alpha_tp[j] = sampleGamma(0.5 * theta[i][t], b_alpha);
                if (alpha_tp[j] < 0.001) alpha_tp[j] = 0.001;

                // Affectation temporaire
                SS[flat_index] = static_cast<double>(j_new);
                CS1[j_new]     = 1.0;

                // Cohésion (on garde la même dépendance à b_j qu'avant)
                double g_xjStar = numerateur(CS1, j_new, b_j, l)
                                  - denominateur(SS, X, j_new, CS1, l, b_j);

                // Log-likelihood
                double bj_new = alpha_tp[j] * (1.0 - mu_tp[j]);
                double aj_new = alpha_tp[j] * mu_tp[j];
                double log_lik_new = logBetaPDF(IDH[i][t], aj_new, bj_new);

                prob[j_new] = log_lik_new
                              + g_xjStar
                              + std::log(M / static_cast<double>(m_neal));

                // reset
                SS[flat_index] = -1.0;
                CS1[j_new]     = -1.0;
            }

            // étiquette temporaire (comme dans ton code)
            SS[flat_index] = static_cast<double>(kminus_current);

            // --- 3. Normalisation & échantillonnage discret (log-sum-exp) ---
            double max_log_prob = prob[0];
            for (int k = 1; k < h_neal; ++k) {
                if (prob[k] > max_log_prob) max_log_prob = prob[k];
            }

            double sum_exp = 0.0;
            for (int k = 0; k < h_neal; ++k) {
                norm_prob[k] = std::exp(prob[k] - max_log_prob);
                sum_exp += norm_prob[k];
            }

            if (sum_exp <= 0.0 || !std::isfinite(sum_exp)) {
                std::cerr << "Warning fullPartition: sum_exp invalide, on passe à uniforme.\n";
                const double inv = 1.0 / static_cast<double>(h_neal);
                for (int k = 0; k < h_neal; ++k) {
                    norm_prob[k] = inv;
                }
            } else {
                const double inv_sum = 1.0 / sum_exp;
                for (int k = 0; k < h_neal; ++k) {
                    norm_prob[k] *= inv_sum;
                }
            }

            int Ng = sampleDiscrete(norm_prob);  // doit retourner un int dans [0, h_neal-1]

            if (Ng < 0 || Ng >= h_neal) {
                std::cerr << "Warning fullPartition: Ng=" << Ng
                          << " hors [0," << (h_neal-1) << "], on skip.\n";
                continue;
            }

            // --- 4. Affectation finale ---
            SS[flat_index] = static_cast<double>(Ng);

            if (Ng < kminus_current) {
                // cluster existant
                CS[Ng] += 1.0;
            } else {
                // nouveau cluster
                int j_aux = Ng - kminus_current;  // 0..m_neal-1

                // Dans ton code, tu mets SS[flat_index] = kminus_current ici.
                // On garde ce comportement pour rester cohérent.
                SS[flat_index] = static_cast<double>(kminus_current);

                // Ajout des paramètres du nouveau cluster
                mu.push_back(mu_tp[j_aux]);
                alpha.push_back(alpha_tp[j_aux]);
                CS.push_back(1.0);

                Ncl = static_cast<int>(CS.size());
            }

        } // i
    } // t

    if (verbose) {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~FIN PARTITIONNMENT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    }

    // Recalcul des tailles de clusters à partir de SS
    CS = calculateCs(SS);
    Ncl = static_cast<int>(CS.size());

    PriorPartitionResult result;
    result.Ncl     = Ncl;
    result.CS_r    = CS;
    result.SS      = SS;
    result.mu_r    = mu;
    result.alpha_r = alpha;

    return result;
}

/*PriorPartitionResult fullPartition(
    double b, double M, Vector& SS_in, Vector& CS_in, const Matrix& IDH, 
    Vector& mu_in, double b_mu, Vector& alpha_in, double b_alpha, 
    const Matrix& lambda_1, const Matrix& theta, const Matrix& X, int m_neal) 
{
    std::cout << "~~~~~~~~~~~~~~~~~~~debut partitionnement~~~~~~~~~~~~~~~~~~~~" << std::endl;

    // --- 0. Vérifications de base sur les dimensions ---
    const std::size_t p   = IDH.size();
    if (p == 0) {
        throw std::invalid_argument("fullPartition: IDH vide (p=0).");
    }
    const std::size_t T1  = IDH[0].size();
    if (T1 == 0) {
        throw std::invalid_argument("fullPartition: IDH a 0 colonnes (T1=0).");
    }
    const std::size_t p_T1 = p * T1;

    // Vérifier la cohérence de IDH, lambda_1, theta, X en (p x T1)
    auto check_matrix_shape = [p, T1](const Matrix& M, const char* name) {
        if (M.size() != p) {
            throw std::invalid_argument(std::string(name) + ": nombre de lignes != p.");
        }
        for (std::size_t i = 0; i < p; ++i) {
            if (M[i].size() != T1) {
                throw std::invalid_argument(std::string(name) + ": nombre de colonnes != T1 à la ligne " + std::to_string(i));
            }
        }
    };

    check_matrix_shape(IDH,      "IDH");
    check_matrix_shape(lambda_1, "lambda_1");
    check_matrix_shape(theta,    "theta");
    //check_matrix_shape(X,        "X");
    // Dimension l = nombre de colonnes de X (ici l = T1 si X est (p x T1))
    const int l = static_cast<int>(X[0].size());
    if (l <= 0) {
        throw std::invalid_argument("fullPartition: l <= 0 (nb de colonnes de X).");
    }

    // Hyper-param b_j (ici identique à b, mais tu peux changer)
    double b_j = b;

    // On ignore le m_neal passé si tu veux absolument 3
    if (m_neal <= 0) m_neal = 3;

    // Alias pour les vecteurs passés par référence
    Vector& SS    = SS_in;   // labels de taille p*T1
    Vector& CS    = CS_in;   // tailles de clusters
    Vector& mu    = mu_in;
    Vector& alpha = alpha_in;

    // Vérifier SS a bien p*T1 éléments
    if (SS.size() != p_T1) {
        throw std::invalid_argument("fullPartition: SS.size() != p*T1.");
    }

    // Vérifier cohérence entre CS, mu, alpha
    if (!(CS.size() == mu.size() && mu.size() == alpha.size())) {
        throw std::invalid_argument("fullPartition: tailles incohérentes entre CS, mu, alpha.");
    }

    // Copies de sauvegarde
    Vector mu_rrtp   = mu;
    Vector alpha_rrtp= alpha;
    Vector CS_rrtp   = CS;
    Vector SS_tp     = SS;

    int Ncl = static_cast<int>(CS.size());

    // --- Boucle sur tous les points (i, t) ---
    for (std::size_t t = 0; t < T1; ++t) {
        for (std::size_t i = 0; i < p; ++i) {

            const std::size_t flat_index = i + t * p;  // index 0..p*T1-1
            int k0 = static_cast<int>(std::round(SS[flat_index])); // cluster actuel

            // Sanity check: k0 peut être hors [0, Ncl-1] si SS est corrompu
            if (k0 < 0 || k0 >= Ncl) {
                // On force un retrait "sans cluster"
                // ou tu peux décider de lancer une exception :
                std::cerr << "Warning fullPartition: label k0=" << k0
                          << " hors [0," << (Ncl-1) << "], on saute ce point.\n";
                continue;
            }

            int k_star = 0; // indicateur de groupe perdu

            // --- 1. Retrait de l'élément (i, t) du cluster k0 ---
            if (k0 < static_cast<int>(CS.size()) && CS[k0] > 1.0) {
                // On décrémente simplement la taille
                CS[k0] -= 1.0;
            } else {
                // Cluster k0 est détruit

                int kminus = Ncl - 1;    // dernier cluster (index Ncl-1)
                Ncl -= 1;                // nouveau nombre de clusters = ancien - 1

                if (k0 < Ncl) {
                    // On permute le label kminus -> k0
                    for (std::size_t l_idx = 0; l_idx < p_T1; ++l_idx) {
                        int lab = static_cast<int>(std::round(SS[l_idx]));
                        if (lab == kminus) {
                            SS[l_idx] = static_cast<double>(k0);
                        }
                    }
                    k_star = 1;
                    SS[flat_index] = static_cast<double>(kminus);

                    // Permutation des paramètres
                    mu[k0]    = mu[kminus];
                    alpha[k0] = alpha[kminus];
                    CS[k0]    = CS[kminus];
                }

                // On supprime le dernier élément (kminus) de CS, mu, alpha
                if (!CS.empty())    CS.pop_back();
                if (!mu.empty())    mu.pop_back();
                if (!alpha.empty()) alpha.pop_back();

                Ncl = static_cast<int>(CS.size());
            }

            // On marque l'élément comme retiré
            SS[flat_index] = -1.0;

            // --- 2. Calcul des probabilités de proposition ---
            const int kminus_current = Ncl;
            const int h_neal         = kminus_current + m_neal;

            if (kminus_current < 0 || h_neal <= 0) {
                throw std::runtime_error("fullPartition: h_neal ou Ncl invalides.");
            }

            Vector prob(h_neal, 0.0);
            Vector CS1(h_neal, -1.0);

            // Copier CS dans CS1[0..Ncl-1]
            for (int r = 0; r < kminus_current; ++r) {
                if (r < static_cast<int>(CS.size())) {
                    CS1[r] = CS[r];
                } else {
                    // Sécurité
                    CS1[r] = 0.0;
                }
            }

            // a) Clusters existants (j = 0 .. Ncl-1)
            for (int j = 0; j < kminus_current; ++j) {
                // Cohésion
                SS[flat_index] = static_cast<double>(j);
                CS1[j]=countNonzeroClusterJ(SS,j);

                double g_xjStar = numerateur(CS1, j, b, l)
                                  - denominateur(SS, X, j, CS1, l, b);

                // Log-likelihood IDH
                double b_j=alpha[j]*(1-mu[j]);
                double a_j=alpha[j]*mu[j];
                double log_lik_IDH = logBetaPDF(IDH[i][t], a_j, b_j);
                SS[flat_index] = -1.0;
                CS1[j]       -= 1.0;
                double g_xj = numerateur(CS1, j, b, l)
                                  - denominateur(SS, X, j, CS1, l, b);

                // Log-proba
                prob[j] = g_xjStar -g_xj+ log_lik_IDH + std::log(CS1[j]) + std::log(M);
            }

            // b) Nouveaux clusters (j = 0 .. m_neal-1)
            Vector mu_tp(m_neal);
            Vector alpha_tp(m_neal);

            for (int j = 0; j < m_neal; ++j) {

                const int j_new = kminus_current + j;   // index dans prob/CS1

                // Tirage des paramètres du nouveau cluster
                mu_tp[j] = sampleBeta(lambda_1[i][t], b_mu);
                if (mu_tp[j] < 0.001) mu_tp[j] = 0.001;
                if (mu_tp[j] > 0.999)  mu_tp[j] = 0.999;

                alpha_tp[j] = sampleGamma(0.5 * theta[i][t], b_alpha);
                if (alpha_tp[j] < 0.001) alpha_tp[j] = 0.001;

                // Affectation temporaire
                SS[flat_index]   = static_cast<double>(j_new);
                CS1[j_new]       = 1.0;

                // Cohésion
                double g_xjStar = numerateur(CS1, j_new, b_j, l)
                                  - denominateur(SS, X, j_new, CS1, l, b_j);

                // Log-likelihood
                double b_j=alpha_tp[j]*(1-mu_tp[j]);
                double a_j=alpha_tp[j]*mu_tp[j];
                double log_lik_new = logBetaPDF(IDH[i][t], a_j, b_j);

                prob[j_new] = log_lik_new + g_xjStar + std::log(M / static_cast<double>(m_neal));

                // reset
                SS[flat_index]   = -1.0;
                CS1[j_new]       = -1.0;  // on revient à l'état "sans ce point"
            }
            SS[flat_index]=kminus_current;

            // --- 3. Normalisation & échantillonnage discret ---
            double min_log_prob = prob[0];
            double max_log_prob = prob[0];
            for (double v : prob) {
                if (v < min_log_prob) min_log_prob = v;
                if (v > max_log_prob) max_log_prob = v;
            }

            Vector norm_prob(h_neal, 0.0);
            double sum_exp = 0.0;

            if (std::abs(min_log_prob - max_log_prob) < 1e-12) {
                std::fill(norm_prob.begin(), norm_prob.end(), 1.0 / static_cast<double>(h_neal));
            } else {
                for (int k = 0; k < h_neal; ++k) {
                    norm_prob[k] = std::exp((prob[k] - min_log_prob) / (max_log_prob - min_log_prob));
                    sum_exp += norm_prob[k];
                }
                if (sum_exp <= 0.0 || !std::isfinite(sum_exp)) {
                    std::cerr << "Warning fullPartition: sum_exp invalide, on passe à uniforme.\n";
                    std::fill(norm_prob.begin(), norm_prob.end(), 1.0 / static_cast<double>(h_neal));
                } else {
                    for (int k = 0; k < h_neal; ++k) {
                        norm_prob[k] /= sum_exp;
                    }
                }
            }

            int Ng = sampleDiscrete(norm_prob);  // doit retourner un int dans [0, h_neal-1]

            if (Ng < 0 || Ng >= h_neal) {
                std::cerr << "Warning fullPartition: Ng=" << Ng
                          << " hors [0," << (h_neal-1) << "], on skip.\n";
                continue;
            }

            // --- 4. Affectation finale ---
            SS[flat_index] = static_cast<double>(Ng);

            if (Ng < kminus_current) {
                // cluster existant
                CS[Ng] += 1.0;
                if (k_star == 1) {
                    std::cout << "On a perdu un groupe " << std::endl;
                }
            } else {
                // nouveau cluster
                int j_aux = Ng - kminus_current;  // 0..m_neal-1
                SS[flat_index]=(double)kminus_current;
                // Ajout des paramètres du nouveau cluster
                mu.push_back(mu_tp[j_aux]);
                alpha.push_back(alpha_tp[j_aux]);
                CS.push_back(1.0);

                Ncl = static_cast<int>(CS.size());
            }

        } // i
    } // t

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~FIN PARTITIONNMENT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    CS=calculateCs(SS);
    Ncl= static_cast<int>(CS.size());
    PriorPartitionResult result;
    result.Ncl     = Ncl;
    result.CS_r    = CS;
    result.SS      = SS;
    result.mu_r    = mu;
    result.alpha_r = alpha;

    return result;
}*/









/* Matrix fullLambdaPG(const Vector& r, double r_b, double xi, const Matrix& B, Matrix& lambda_PG, double sigma, const Vector& SS, const Vector& Y_flat, const Matrix& W_matrix) {

    size_t p = B.size();
    if (p == 0) return lambda_PG;
    size_t T1 = lambda_PG[0].size();
    size_t N = p * T1; 
    
    // Vérification: Y_flat doit être de taille N
    if (Y_flat.size() != N) {
        throw std::invalid_argument("Full_lambdaPG: La taille de Y_flat ne correspond pas à p * T1.");
    }

    // --- Période t=0 (Initialisation) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        size_t i_flat = i + t * p; // Indice aplati
        double lambda_i0 = lambda_PG[i][0];
        
        try {
            double B_ii = B[i][i];
            if (std::abs(B_ii) < 1e-12) continue;

            double var = sigma / B_ii;
            if (var <= 0.0) continue;
            
            // 1. Proposer lambda_star_i par Log-Normale
            double log_mu_prop = std::log(lambda_i0); 
            std::normal_distribution<double> normal_prop(log_mu_prop, std::sqrt(var));
            
            double lambda_star_i = std::exp(normal_prop(generator_prior));
            
            if (lambda_star_i < 0.0001) lambda_star_i = 0.0001;

            // 2. Calculer le ratio d'acceptation tp (Log-Alpha)
            double tp = 0.0;

            // --- Terme 1: Log-Prior Gaussien Temporel (t=0) ---
            double log_lambda_star_i = std::log(lambda_star_i);
            double log_lambda_i0 = std::log(lambda_i0);

            // Calcul de Sum(B[i,:] * log(lambda[:,0]))
            double sum_B_log_lambda = 0.0;
            for (size_t k = 0; k < p; ++k) {
                sum_B_log_lambda += B[i][k] * std::log(lambda_PG[k][0]);
            }
            
            // Log-Prior (Log-Normal ratio simplifié)
            tp += -0.5 * B_ii / sigma * (log_lambda_star_i - log_lambda_i0) * (-2.0 * sum_B_log_lambda / B_ii + log_lambda_star_i + log_lambda_i0);

            // --- Terme 2: Log-Vraisemblance (Gamma/Poisson-Gamma) ---
            size_t j = static_cast<size_t>(std::round(SS[i_flat])); 
            if (j >= r.size()) continue; 
            
            double r_j = r[j];
            
            // Vraisemblance (Simplified Gamma(lambda, 1/lambda) likelihood)
            // L(lambda_i,t) = log(Gamma(lambda_i,t) | r_j, 1/r_b) * P(W_i,t | lambda_i,t)
            
            // Log-Prior de Cluster Gamma: log(Gamma(lambda* | r_j, 1/r_b)) - log(Gamma(lambda | r_j, 1/r_b))
            // Terme de forme : (r_j - 1) * (log(lambda_star) - log(lambda))
            tp += (r_j - 1.0) * (log_lambda_star_i - log_lambda_i0);
            
            // Terme de taux/échelle : -r_b * (lambda_star - lambda)
            tp += -r_b * (lambda_star_i - lambda_i0);
            
            // --- Terme W (Vraisemblance des données latentes - souvent W | lambda ~ Poisson(lambda)) ---
            // Le terme W (non donné) est souvent log(Poisson(W | lambda*)) - log(Poisson(W | lambda))
            // Si W | lambda ~ Poisson(lambda): tp += W * (log(lambda*) - log(lambda)) - (lambda* - lambda)
            // Je suppose W_matrix contient la variable latente W_{i,t}
            double W_it = W_matrix[i][t]; 
            tp += W_it * (log_lambda_star_i - log_lambda_i0) - (lambda_star_i - lambda_i0);
            
            // 3. Décision d'Acceptation
            if (tp > logUniformRvs()) {
                lambda_PG[i][0] = lambda_star_i;              
            }
        } catch (const std::exception& e) {
            std::cerr << "Erreur Full_lambdaPG (t=0, i=" << i << "): " << e.what() << ". Rejet." << std::endl;
        }                                                                             
    }
    
    // --- Périodes t > 0 ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        // La logique ici est similaire à la boucle t=0, mais inclut la dépendance temporelle xi
        // sur log(lambda_PG[k][t]).

        for (size_t i = 0; i < p; ++i) {
            size_t t_plus_1 = t + 1;
            size_t i_flat = i + t_plus_1 * p;
            double lambda_i_tplus1 = lambda_PG[i][t_plus_1];
            
            try {
                 double B_ii = B[i][i];
                 if (std::abs(B_ii) < 1e-12) continue;
                 double var = sigma / B_ii;
                 if (var <= 0.0) continue;
                 
                 // 1. Proposition Log-Normale
                 double log_mu_prop = std::log(lambda_i_tplus1); 
                 std::normal_distribution<double> normal_prop(log_mu_prop, std::sqrt(var));
                 double lambda_star_i = std::exp(normal_prop(generator_prior));
                 
                 if (lambda_star_i < 0.0001) lambda_star_i = 0.0001;
                 
                 // 2. Calcul du ratio d'acceptation (Log-Alpha)
                 double tp = 0.0;
                 double log_lambda_star_i = std::log(lambda_star_i);
                 double log_lambda_i_tplus1 = std::log(lambda_i_tplus1);

                 // --- Terme 1: Log-Prior Gaussien Temporel (t>0) ---
                 double sum_B_log_lambda_pred = 0.0;
                 for (size_t k = 0; k < p; ++k) {
                     // Utilise la prédiction xi * log(lambda[k,t])
                     sum_B_log_lambda_pred += B[i][k] * (log_lambda_PG[k][t] * xi);
                 }
                 
                 // Log-Prior (Log-Normal ratio simplifié)
                 tp += -0.5 * B_ii / sigma * (log_lambda_star_i - log_lambda_i_tplus1) * (-2.0 * sum_B_log_lambda_pred / B_ii + log_lambda_star_i + log_lambda_i_tplus1);
                 
                 // --- Terme 2: Log-Vraisemblance (Gamma/Poisson-Gamma) ---
                 size_t j = static_cast<size_t>(std::round(SS[i_flat])); 
                 if (j >= r.size()) continue; 
                 double r_j = r[j];
                 
                 // Log-Prior de Cluster Gamma: log(Gamma(lambda* | r_j, 1/r_b)) - log(Gamma(lambda | r_j, 1/r_b))
                 tp += (r_j - 1.0) * (log_lambda_star_i - log_lambda_i_tplus1);
                 tp += -r_b * (lambda_star_i - lambda_i_tplus1);
                 
                 // --- Terme W ---
                 double W_it = W_matrix[i][t_plus_1]; 
                 tp += W_it * (log_lambda_star_i - log_lambda_i_tplus1) - (lambda_star_i - lambda_i_tplus1);

                 // 3. Décision d'Acceptation
                 if (tp > logUniformRvs()) {
                     lambda_PG[i][t_plus_1] = lambda_star_i;              
                 }
            } catch (const std::exception& e) {
                std::cerr << "Erreur Full_lambdaPG (t>0, i=" << i << ", t=" << t_plus_1 << "): " << e.what() << ". Rejet." << std::endl;
            }
        } // Fin de la boucle i
    } // Fin de la boucle t
    
    return lambda_PG;
} */


Vector fullPsiGamma(const Vector& Y, const Vector& e, const Matrix& X, const Vector& r, const Vector& beta, Vector& psi, const Matrix& lambda_1, double sigma_psi, const Vector& SS) {
    size_t Ncl = r.size(); // Nombre de clusters
    size_t p_T1 = Y.size(); // Nombre total de points
    
    if (Ncl == 0 || sigma_psi <= 0.0) return psi;

    // Calculer les tailles de cluster (Cs) pour n_j
    Vector Cs = calculateCs(SS);
    
    // Boucle sur chaque cluster j
    for (size_t j = 0; j < Ncl; ++j) {
        double psi_j_current = psi[j];
        double n_j = Cs[j];

        if (n_j < 1e-9) continue; 

        // --- 1. Calculer la moyenne lambda_j (MeanSS) ---
        // Cette étape nécessite une implémentation robuste de MeanSS_Scalar qui donne la moyenne des lambdas
        // double lambda_j = MeanSS_Scalar(lambda_1, SS, (int)j); 
        double lambda_j = 0.5; // PLACEHOLDER: Utiliser votre fonction MeanSS réelle

        // --- 2. Proposer psi_star_j ---
        // Proposition: N(0.5*lambda_j + 0.5*psi_j, sigma_psi/n_j)
        double scale_prop = std::sqrt(sigma_psi / n_j);
        double mean_prop = 0.5 * lambda_j + 0.5 * psi_j_current;
        
        std::normal_distribution<double> normal_proposal(mean_prop, scale_prop);
        double psi_star_j = normal_proposal(generator_prior);

        // --- 3. Calculer le ratio d'acceptation tp (Log-Alpha) ---
        double tp = 0.0;

        // a) Terme de Prior/Proposition (Ratio Gaussien sur psi)
        // tp = (psi[j]-psi_star)*(psi[j]+psi_star-2*lambda_j) * n_j / (4*2*sigma_psi)
        double diff_psi = psi_j_current - psi_star_j;
        double sum_diff_lambda = psi_j_current + psi_star_j - 2.0 * lambda_j;
        tp += diff_psi * sum_diff_lambda * n_j / (8.0 * sigma_psi);

        // b) Terme de Vraisemblance (Log L(Y | psi*) - Log L(Y | psi))
        for (size_t k_flat = 0; k_flat < p_T1; ++k_flat) {
            if (static_cast<int>(std::round(SS[k_flat])) == (int)j) {
                
                // Calcul du prédicteur linéaire : X*beta
                const Vector& X_row = X[k_flat];
                double linear_predictor_base = vectorDotProduct(X_row, beta);
                
                // Calculer mu (taux d'arrivée de Poisson)
                double E_k = e[k_flat];
                double mu_star = std::exp(linear_predictor_base + psi_star_j);
                double mu_current = std::exp(linear_predictor_base + psi_j_current);

                double r_j_param = r[j];
                double Y_k = Y[k_flat];

                // Terme 1: Log(r + e*mu) ratio
                // tp += (r + Y) * log((r + e*mu) / (r + e*mu*))
                double log_ratio_r_e_mu = std::log(r_j_param + E_k * mu_current) - std::log(r_j_param + E_k * mu_star);
                tp += (r_j_param + Y_k) * log_ratio_r_e_mu;
                
                // Terme 2: Y * (psi_star - psi)
                // tp += Y * (psi_star - psi)
                tp += Y_k * (psi_star_j - psi_j_current);
            }
        }
        
        // 4. Décision d'Acceptation
        if (tp > logUniformRvs()) {
            psi[j] = psi_star_j;
        }
    }
    return psi;
}


Vector fullSigmaPoll(const Vector& a_sigma_beta, const Vector& b_sigma_beta, const Vector& beta, Vector& sigma_beta) {
    size_t l = beta.size(); // Nombre de coefficients
    if (l == 0) return sigma_beta;

    if (a_sigma_beta.size() != l || b_sigma_beta.size() != l || sigma_beta.size() != l) {
        throw std::invalid_argument("Full_sigma_poll: Les vecteurs de bornes et sigma_beta doivent avoir la taille l.");
    }

    // --- Boucle MH pour chaque composante sigma_beta[i] ---
    for (size_t i = 0; i < l; ++i) {
        
        double sigma_i_current = sigma_beta[i];
        double a_i = a_sigma_beta[i];
        double b_i = b_sigma_beta[i];
        
        if (sigma_i_current <= 0.0) {
             throw std::runtime_error("Full_sigma_poll: sigma_i_current doit être positif.");
        }

        // 1. Proposer sigma_beta_star (Uniforme)
        if (a_i >= b_i || a_i <= 0.0) {
            std::cerr << "Erreur Full_sigma_poll: Bornes uniformes invalides pour i=" << i << ". Saut." << std::endl;
            continue;
        }
        
        std::uniform_real_distribution<double> uniform_prop(a_i, b_i);
        double sigma_star_i = uniform_prop(generator_prior);

        // 2. Calcul du ratio d'acceptation tp (Log-Alpha)
        double tp = 0.0;
        
        // Terme 1: Logarithme (Log(sigma) ratio)
        // tp = 0.5*(np.log(sigma_beta[i]) - np.log(sigma_beta_star))
        tp += 0.5 * (std::log(sigma_i_current) - std::log(sigma_star_i));
        
        // Terme 2: Exponentiel (Log-Likelihood Gaussien ratio)
        // tp += 0.5*(1/sigma_beta[i] - 1/sigma_beta_star)*(beta[i]**2)
        double beta_i_sq = std::pow(beta[i], 2.0);
        tp += 0.5 * (1.0 / sigma_i_current - 1.0 / sigma_star_i) * beta_i_sq;

        // 3. Décision d'Acceptation
        // Le ratio de proposition (Uniforme) s'annule.
        if (tp > logUniformRvs()) {
            sigma_beta[i] = sigma_star_i; // Accepter la proposition
        } 
    }
    
    return sigma_beta;
}


Vector fullR(const Vector& Y, const Vector& e, Vector& r, double b_r, const Vector& mu, const Matrix& theta, const Vector& SS) {
    
    // Calculer les tailles de cluster (Cs) et Ncl
    Vector Cs = calculateCs(SS);
    int Ncl = Cs.size();
    
    // Boucle sur chaque cluster j
    for (size_t j = 0; j < Ncl; ++j) {
        double r_j_current = r[j];
        double n_j = Cs[j];

        if (n_j < 1e-9) continue; // Ignorer les clusters vides

        // 1. Calculer les statistiques du cluster (MeanSS)
        // PLACEHOLDER: thetaj = MeanSS_Scalar(theta, SS, (int)j)
        double thetaj = 1.0; 
        
        // 2. Proposition r_star (MH)
        // r_star = st.gamma.rvs(0.25*thetaj+0.25*r[j],b_r)
        double a_prop_current = 0.25 * thetaj + 0.25 * r_j_current;
        double scale_prop = b_r; // b_r est l'échelle (scale)
        
        if (a_prop_current <= 0.0 || scale_prop <= 0.0) continue; 
        
        std::gamma_distribution<double> gamma_proposal(a_prop_current, scale_prop);
        double r_star = gamma_proposal(generator_prior);
        
        if (r_star < 0.001) r_star = 0.001;

        // 3. Calculer le ratio d'acceptation tp (Log-Alpha)
        double tp = 0.0;

        // a) Terme de Vraisemblance (Log L(Y | r*) - Log L(Y | r))
        // tp=f.LogLikhoodPoissonGammaJ(Y,e,mu,r_star,SS,j)-f.LogLikhoodPoissonGammaJ(Y,e,mu,r[j],SS,j)
        // PLACEHOLDER: tp += LogLik_star - LogLik_current;

        // b) Terme de Prior (Ratio de Log-PDF Gamma P(r*) / P(r))
        double prior_term_1 = (0.5 * n_j * thetaj - n_j) * (std::log(r_star) - std::log(r_j_current));
        double prior_term_2 = b_r * (r_star - r_j_current) * (1.0 - 1.0 / n_j);
        tp += prior_term_1 + prior_term_2;

        // c) Terme de Proposition (Ratio de Log-PDF Gamma Q(r | r*) / Q(r* | r))
        double a_prop_star = 0.25 * thetaj + 0.25 * r_star;

        // Terme du logarithme: -(a_prop_current-1)*log(r_star) + (a_prop_star-1)*log(r_j)
        tp += (a_prop_star - 1.0) * std::log(r_j_current) - (a_prop_current - 1.0) * std::log(r_star);
        
        // 4. Décision d'Acceptation
        if (tp > logUniformRvs()) {
            r[j] = r_star;
        }
    }
    return r;
}



/**
 * Traduction de fullPartitionPD(...) - Neal's Algorithm (Poisson-Gamma)
 */
/*PartitionResultPG fullPartitionPD(double b, double M, Vector& SS, Vector& CS, const Vector& Y, const Vector& e, const Matrix& X, const Vector& beta, Vector& psi, double sigma_psi, const Matrix& lambda_1, Vector& r, double b_r, const Matrix& theta, int m_neal) {

    
    std::cout << "~~~~~~~~~~~~~~~~~~~debut partitionnement~~~~~~~~~~~~~~~~~~~~" << std::endl;
    
    // Dimensions
    size_t p = theta.size();       // Nombre de régions (lignes de theta)
    size_t T1 = theta[0].size();   // Nombre de périodes (colonnes de theta)
    size_t p_T1 = p * T1;          // Taille totale (taille de SS)
    int l = X[0].size();           // Nombre de covariables (taille de beta)

    // Alias pour la lisibilité (modifiés in-place)
    Vector& r_r = r;
    Vector& psi_r = psi;
    Vector& CS_r = CS; 

    // Initialisation
    size_t Ncl = CS_r.size();
    int h_neal = (int)Ncl + m_neal;
    
    // --- Boucle sur tous les états (i, t) à réaffecter ---
    for (size_t t = 0; t < T1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            
            size_t flat_index = i + t * p; // Indice aplati
            int k0 = (int)SS[flat_index];  // Cluster actuel

            // Sauvegarde des états actuels (pour le cas d'échec)
            Vector r_rrtp = r_r;
            Vector psi_rrtp = psi_r;
            Vector CS_rrtp = CS_r;
            Vector SS_tp = SS;
            
            // --- 1. Retrait de l'élément (i, t) du cluster k0 ---
            int kminus = Ncl; 
            
            if (CS_r[k0] > 1) {
                // Cluster k0 survit
                CS_r[k0] -= 1;
            } else {
                // Cluster k0 est détruit
                kminus = Ncl - 1;
                Ncl -= 1;
                
                // Gérer la permutation des labels (k0 < Ncl) et la suppression du dernier élément
                if (k0 < Ncl) {
                    // 1. Permutation du label Ncl vers k0
                    for (size_t l_idx = 0; l_idx < p_T1; ++l_idx) {
                        if ((int)SS[l_idx] == (int)Ncl) {
                            SS[l_idx] = (double)k0;
                        }
                    }
                    // 2. Mettre les paramètres k0 à la place de Ncl et supprimer Ncl
                    r_r[k0] = r_r[Ncl];
                    psi_r[k0] = psi_r[Ncl];
                    CS_r[k0] = CS_r[Ncl];
                    
                    deleteLastElement(r_r);
                    deleteLastElement(psi_r);
                    deleteLastElement(CS_r);
                    
                } else {
                    // k0 était le dernier cluster, suppression simple
                    deleteLastElement(r_r);
                    deleteLastElement(psi_r);
                    deleteLastElement(CS_r);
                }
                
                // Mettre à jour Ncl et le label de l'élément retiré
                Ncl = CS_r.size(); 
            }
            
            // Marquer l'élément comme retiré
            SS[flat_index] = -1.0; 
            
            // --- 2. Calcul des Probabilités (Neal's Algorithm) ---
            
            kminus = Ncl; 
            h_neal = kminus + m_neal;
            Vector prob(h_neal, 0.0);
            
            // a) Probabilités des Clusters Existants (j = 0 à kminus-1)
            for (int j = 0; j < kminus; ++j) {
                
                // Affectation temporaire à j pour g_xjStar
                SS[flat_index] = (double)j;
                CS_r[j] += 1; // Incrémenter temporairement la taille
                
                // Cohesion (Likelihood of Covariates)
                double g_xjStar = numerateur(CS_r,j, b, l) - denominateurPoll(SS, X, j, CS_r, l, b); 
                
                CS_r[j] -= 1; // Décrémenter pour calculer g_xj
                
                double g_xj = numerateur(CS_r, j,b, l) - denominateurPoll(SS, X, j, CS_r, l, b); 
                
                // Rétablissement du label
                SS[flat_index] = -1.0; 

                // Terme de Vraisemblance (Poisson-Gamma Log-PMF)
                // mu = exp(X[i+t*p,:]*beta + psi_r[j])
                double X_beta = vectorDotProduct(X[flat_index], beta);
                double mu_val = std::exp(X_beta + psi_r[j]);
                
                int Y_it = (int)std::round(Y[flat_index]);
                double log_lik = poissonGammaLog(Y_it, r_r[j], e[flat_index] * mu_val);
                
                // Log-Probabilité: g_xjStar - g_xj + log(M) + log(CS[j]) + LogLikelihood
                prob[j] = g_xjStar - g_xj + std::log(M) + std::log(CS_r[j]) + log_lik;
            }
            
            // b) Probabilités pour les Nouveaux Clusters (j = kminus à h_neal-1)
            for (int j = 0; j < m_neal; ++j) {
                int j_new = kminus + j;
                
                // Échantillonner les paramètres du nouveau cluster (Prior)
                // psi_tp[j]=st.norm.rvs(lambda_1[i,t],sigma_psi)
                // r_tp[j]=st.gamma.rvs(0.5*theta[i,t],b_r)
                
                double psi_tp_j = 0.0; // Placeholder
                double r_tp_j = 0.001; // Placeholder
                
                // Cohesion (Likelihood of Covariates) avec le nouveau cluster (CS=1)
                SS[flat_index] = (double)j_new;
                double g_xjStar = numerateur(CS_r, j,b, l)- denominateurPoll(SS, X, j_new, CS_r, l, b);
                
                // Vraisemblance (Poisson-Gamma Log-PMF)
                double X_beta = vectorDotProduct(X[flat_index], beta);
                double mu_val = std::exp(X_beta + psi_tp_j);
                int Y_it = (int)std::round(Y[flat_index]);
                double log_lik = poissonGammaLog(Y_it, r_tp_j, e[flat_index] * mu_val);
                
                // Log-Probabilité: g_xjStar + log(M/m_neal) + LogLikelihood
                prob[j_new] = log_lik + g_xjStar + std::log(M / (double)m_neal);
                
                SS[flat_index] = -1.0; // Rétablissement
            }
            
            // --- 3. Normalisation et Échantillonnage ---
            
            // Normalisation robuste des probabilités (prob)
            // (La logique de clamping et de normalisation est la plus sûre ici)
            
            double max_log_prob = *std::max_element(prob.begin(), prob.end());
            Vector exp_prob(h_neal);
            double sum_exp_prob = 0.0;
            
            for (int k = 0; k < h_neal; ++k) {
                exp_prob[k] = std::exp(prob[k] - max_log_prob); 
                sum_exp_prob += exp_prob[k];
            }
            
            Vector norm_prob(h_neal);
            for (int k = 0; k < h_neal; ++k) {
                norm_prob[k] = exp_prob[k] / sum_exp_prob;
            }
            
            // Choix du nouveau cluster (Ng)
            int Ng = sampleDiscrete(norm_prob);
            
            // 4. Affectation Finale et Mise à Jour des Paramètres
            
            SS[flat_index] = (double)Ng;
            
            if (Ng < kminus) {
                // Affecté à un cluster existant
                CS_r[Ng] += 1;
            } else {
                // Affecté à un nouveau cluster
                // Mettre à jour les paramètres (r, psi, CS)
                // PLACEHOLDER: Append des paramètres r_tp et psi_tp échantillonnés précédemment.
                Ncl += 1;
                CS_r.push_back(1.0);
            }
        } // Fin de la boucle i
    } // Fin de la boucle t

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~FIN PARTITIONNMENT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl; 
    
    // Assembler le résultat
    PartitionResultPG final_result;
    final_result.Ncl = Ncl;
    final_result.CS_r = CS_r;
    final_result.SS = SS;
    final_result.r_r = r_r;
    final_result.psi_r = psi_r;
    
    return final_result;
} */


Matrix fullRPoll(const Vector& E, Matrix& r, const Vector& beta, const Vector& xi, const Matrix& B, const Matrix& mu, double sigma, const Vector& Y, const Vector& SS, int T1) {
    size_t p = B.size(); // Nombre de régions
    if (p == 0) return r;
    
    // Matrice de sortie r_star (pour la proposition)
    Matrix r_star_mat = r;
    
    // --- Initialisation (t=0) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        double r_i0 = r[i][0];
        
        // 1. Vérification du cluster (si cet état est le premier à avoir son label)
        std::pair<int, int> i0_t0 = isValueInMatrix(SS, (int)i, (int)t, (int)p);
        
        if (i0_t0.first == -1) { // L'état (i, 0) est le premier à avoir son label (non lié à un état précédent)
            
            try {
                double B_ii = B[i][i];
                if (std::abs(B_ii) < 1e-12) continue;
                
                // a) Proposition r_star[i] ~ Log-Normale
                double var_1 = sigma / B_ii; 
                double log_mu_prop = std::log(r_i0); 
                std::normal_distribution<double> normal_prop(log_mu_prop, std::sqrt(var_1));
                r_star_mat[i][0] = std::exp(normal_prop(generator_prior));
                if (r_star_mat[i][0] < 0.001) r_star_mat[i][0] = 0.001;

                // b) Calcul du ratio d'acceptation (tp)
                double tp = 0.0;
                double r_star_i = r_star_mat[i][0];
                
                // Terme 1: Prior Temporel (t=0)
                // tp = tp*(np.log(r_star[i])+np.log(r[i,0]) -2*np.sum(B[:,i]*np.log(r[:,0]))/B[i,i] +2*np.log(r[i,0]))
                double sum_B_log_r = 0.0;
                for (size_t k = 0; k < p; ++k) {
                    sum_B_log_r += B[k][i] * std::log(r[k][0]); // B[:,i] est B[k][i] en C++
                }
                
                double term1_multiplier = (std::log(r_star_i) - std::log(r_i0));
                double term1_value = (std::log(r_star_i) + std::log(r_i0) - 2.0 * sum_B_log_r / B_ii + 2.0 * std::log(r_i0));
                
                tp += -0.5 * B_ii / sigma * term1_multiplier * term1_value;
                
                // Terme 2: Vraisemblance Conditionnelle (Somme sur les points du MÊME cluster)
                for (size_t t_1 = 0; t_1 < T1; ++t_1) {
                    for (size_t i_1 = 0; i_1 < p; ++i_1) {
                        size_t flat_idx_1 = i_1 + t_1 * p;
                        
                        // Si le point (i_1, t_1) a le même label que (i, 0)
                        if (std::abs(SS[flat_idx_1] - SS[i]) < 1e-9) { 
                            
                            // Log-Ratio Gamma (lgamma)
                            tp += std::lgamma(r_star_i + Y[flat_idx_1]) - std::lgamma(r[i_1][t_1] + Y[flat_idx_1]);
                            tp += -std::lgamma(r_star_i) + std::lgamma(r[i_1][t_1]);

                            // Terme Log(1 + mu/r) ratio
                            double mu_val = mu[i_1][t_1];
                            double E_val = E[flat_idx_1];

                            tp += Y[flat_idx_1] * (std::log((r[i_1][t_1] + E_val * mu_val) / (r_star_i + E_val * mu_val)));
                            tp -= r_star_i * std::log(1.0 + mu_val * E_val / r_star_i);
                            tp += r[i_1][t_1] * std::log(1.0 + mu_val * E_val / r[i_1][t_1]);
                        }
                    }
                }
                
                // c) Décision d'Acceptation
                if (tp > logUniformRvs()) {
                    r[i][0] = r_star_i; 
                }
            } catch (const std::exception& e) {
                std::cerr << "Valeur trop grande de r pour t=0, i=" << i << ": " << e.what() << std::endl;
            }
        } else {
            // L'état (i, 0) est lié à un état précédent, il prend donc son paramètre
            r[i][0] = r[i0_t0.first][i0_t0.second];
        }                                                                              
    }

    // --- Périodes t > 0 ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            size_t t_plus_1 = t + 1;
            size_t flat_idx = i + t_plus_1 * p;
            
            // 1. Vérification du cluster (si cet état est le premier à avoir son label pour cette période)
            std::pair<int, int> i0_t0 = isValueInMatrix(SS, (int)i, (int)t_plus_1, (int)p);

            if (i0_t0.first == -1) { 
                try {
                    double r_i_tplus1 = r[i][t_plus_1];
                    double B_ii = B[i][i];
                    if (std::abs(B_ii) < 1e-12) continue;
                    double var_1 = sigma / B_ii;

                    // a) Calcul de la moyenne Log-Normale (moy_1)
                    double sum_B_log_r_pred = 0.0;
                    for (size_t k = 0; k < p; ++k) {
                        sum_B_log_r_pred += B[i][k] * (0.5 * xi[k] * std::log(r[k][t]) - 0.5 * std::log(r[k][t_plus_1]));
                    }
                    double moy_1 = sum_B_log_r_pred / B_ii + std::log(r_i_tplus1);

                    // b) Proposition r_star[i] ~ Log-Normale
                    std::normal_distribution<double> normal_prop(moy_1, std::sqrt(var_1));
                    double r_star_i = std::exp(normal_prop(generator_prior));
                    if (r_star_i < 0.001) r_star_i = 0.001;
                    r_star_mat[i][t_plus_1] = r_star_i;

                    // c) Calcul du ratio d'acceptation (tp)
                    double tp = 0.0;
                    double log_r_star_i = std::log(r_star_i);
                    double log_r_i_tplus1 = std::log(r_i_tplus1);
                    
                    // Terme 1: Prior Temporel (t>0)
                    double term1_multiplier = (log_r_star_i - log_r_i_tplus1);
                    double sum_B_log_r_current = 0.0;
                    for (size_t k = 0; k < p; ++k) {
                        sum_B_log_r_current += B[i][k] * (std::log(r[k][t_plus_1]) - xi[k] * std::log(r[k][t]));
                    }
                    
                    tp += -0.5 * B_ii / sigma * term1_multiplier * (
                         0.25 * (log_r_star_i - log_r_i_tplus1) + 0.5 * sum_B_log_r_current / B_ii
                    );

                    // Terme 2: Vraisemblance Conditionnelle (Somme sur les points du MÊME cluster)
                    for (size_t t_1 = 0; t_1 < T1; ++t_1) {
                        for (size_t i_1 = 0; i_1 < p; ++i_1) {
                            size_t flat_idx_1 = i_1 + t_1 * p;
                            if (std::abs(SS[flat_idx_1] - SS[flat_idx]) < 1e-9) { 
                                
                                // ... (Termes de Log-Ratio Gamma et Log(1+mu/r) - Identiques à ceux de t=0) ...
                                double Y_val = Y[flat_idx_1];
                                double E_val = E[flat_idx_1];
                                double r_i1_t1 = r[i_1][t_1];
                                double mu_val = mu[i_1][t_1];
                                
                                tp += std::lgamma(r_star_i + Y_val) - std::lgamma(r_i1_t1 + Y_val);
                                tp += -std::lgamma(r_star_i) + std::lgamma(r_i1_t1);
                                tp += Y_val * (std::log((r_i1_t1 + E_val * mu_val) / (r_star_i + E_val * mu_val)));
                                tp -= r_star_i * std::log(1.0 + mu_val * E_val / r_star_i);
                                tp += r_i1_t1 * std::log(1.0 + mu_val * E_val / r_i1_t1);
                            }
                        }
                    }
                    
                    // d) Décision d'Acceptation
                    if (tp > logUniformRvs()) {
                        r[i][t_plus_1] = r_star_i;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Erreur fullRPoll (t>0, i=" << i << ", t=" << t_plus_1 << "): " << e.what() << ". Rejet." << std::endl;
                }
            } else {
                // L'état est lié à un état précédent, il prend donc son paramètre
                r[i][t_plus_1] = r[i0_t0.first][i0_t0.second];
            }
        }
    }
    return r;
}


Matrix fullPsiPoll(const Vector& E, const Matrix& X, Matrix& psi, const Vector& beta, const Vector& xi, const Matrix& B, const Matrix& r, double sigma, const Vector& Y, const Vector& SS, int T1) {

    size_t p = B.size(); // Nombre de régions
    size_t l_cov = beta.size(); // Taille de beta
    if (p == 0 || T1 == 0) return psi;
    
    // Matrice de sortie psi_star (pour la proposition)
    Matrix psi_star_mat = psi; 

    // --- Période t=0 (Initialisation) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        size_t flat_idx = i + t * p;
        
        // 1. Vérification du cluster (si cet état est le premier à avoir son label)
        std::pair<int, int> i0_t0 = isValueInMatrix(SS, (int)i, (int)t, (int)p);
        
        if (i0_t0.first == -1) { // Pas d'état précédent avec ce label
            try {
                double psi_i0 = psi[i][0];
                double B_ii = B[i][i];
                if (std::abs(B_ii) < 1e-12) continue;

                // a) Proposition psi_star[i] ~ Normale
                double var_1 = sigma / B_ii; 
                std::normal_distribution<double> normal_prop(psi_i0, std::sqrt(var_1));
                psi_star_mat[i][0] = normal_prop(generator_prior);
                
                // b) Calcul du ratio d'acceptation (tp)
                double tp = 0.0;
                double psi_star_i = psi_star_mat[i][0];
                
                // Terme 1: Prior Temporel (t=0)
                double sum_B_psi = 0.0;
                for (size_t k = 0; k < p; ++k) {
                    sum_B_psi += B[k][i] * psi[k][0]; 
                }
                
                double term1_multiplier = (-psi_i0 + psi_star_i);
                double term1_value = (psi_i0 + psi_star_i - 2.0 * sum_B_psi / B_ii + 2.0 * psi_i0);
                
                tp += -0.5 * B_ii / sigma * term1_multiplier * term1_value;
                
                // Terme 2: Vraisemblance Conditionnelle (Somme sur les points du MÊME cluster)
                for (size_t t_1 = 0; t_1 < T1; ++t_1) {
                    for (size_t i_1 = 0; i_1 < p; ++i_1) {
                        size_t flat_idx_1 = i_1 + t_1 * p;
                        
                        // Si le point (i_1, t_1) a le même label que (i, 0)
                        if (std::abs(SS[flat_idx_1] - SS[flat_idx]) < 1e-9) { 
                            
                            // Log-Ratio Likelihood
                            double Y_val = Y[flat_idx_1];
                            double r_val = r[i_1][t_1];
                            double E_val = E[flat_idx_1];
                            
                            // Prédicteur linéaire de base (tp0 = beta[0] + Sum(X*beta[1:l]))
                            double tp0 = beta[0];
                            // Supposons que X[k, :] est la ligne de covariables
                            const Vector& X_row = X[flat_idx_1]; 
                            for (size_t l00 = 0; l00 < l_cov - 1; ++l00) {
                                tp0 += X_row[l00] * beta[l00 + 1];
                            }

                            double mu_star_val = std::exp(tp0 + psi_star_i);
                            double mu_current_val = std::exp(tp0 + psi[i_1][t_1]);
                            
                            // Terme 1: Y * (psi_star - psi)
                            tp += Y_val * (psi_star_i - psi[i_1][t_1]);

                            // Terme 2: (r + Y) * log((r + E*mu) / (r + E*mu*))
                            double log_ratio_mu = std::log(r_val + E_val * mu_current_val) - std::log(r_val + E_val * mu_star_val);
                            tp += (r_val + Y_val) * log_ratio_mu;
                        }
                    }
                }
                
                // c) Décision d'Acceptation
                if (tp > logUniformRvs()) {
                    psi[i][0] = psi_star_i; 
                }
            } catch (const std::exception& e) {
                std::cerr << "Erreur fullPsiPoll (t=0, i=" << i << "): " << e.what() << ". Rejet." << std::endl;
            }
        } else {
            // L'état (i, 0) est lié, il prend son paramètre
            psi[i][0] = psi[i0_t0.first][i0_t0.second];
        }                                                                              
    }

    // --- Périodes t > 0 ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            size_t t_plus_1 = t + 1;
            size_t flat_idx = i + t_plus_1 * p;
            
            // 1. Vérification du cluster (si cet état est le premier à avoir son label pour cette période)
            std::pair<int, int> i0_t0 = isValueInMatrix(SS, (int)i, (int)t_plus_1, (int)p);

            if (i0_t0.first == -1) { 
                try {
                    double psi_i_tplus1 = psi[i][t_plus_1];
                    double B_ii = B[i][i];
                    if (std::abs(B_ii) < 1e-12) continue;
                    double var_1 = sigma / B_ii;

                    // a) Calcul de la moyenne de la Log-Normale (moy_1)
                    double sum_B_psi_pred = 0.0;
                    for (size_t k = 0; k < p; ++k) {
                        sum_B_psi_pred += B[i][k] * (-0.5 * psi[k][t_plus_1] + 0.5 * xi[k] * psi[k][t]);
                    }
                    double moy_1 = sum_B_psi_pred / B_ii + psi_i_tplus1;

                    // b) Proposition psi_star[i] ~ Normale
                    std::normal_distribution<double> normal_prop(moy_1, std::sqrt(var_1));
                    double psi_star_i = normal_prop(generator_prior);
                    psi_star_mat[i][t_plus_1] = psi_star_i;

                    // c) Calcul du ratio d'acceptation (tp)
                    double tp = 0.0;

                    // Terme 1: Prior Temporel (t>0)
                    double term1_multiplier = (-psi_i_tplus1 + psi_star_i);
                    double sum_B_psi_current = 0.0;
                    for (size_t k = 0; k < p; ++k) {
                        sum_B_psi_current += B[i][k] * (psi[k][t_plus_1] - xi[k] * psi[k][t]);
                    }
                    
                    tp += -0.5 * B_ii / sigma * term1_multiplier * (
                         -0.25 * psi_i_tplus1 + 0.25 * psi_star_i + 0.5 * sum_B_psi_current / B_ii
                    );

                    // Terme 2: Vraisemblance Conditionnelle (Somme sur les points du MÊME cluster)
                    // ... (Même boucle de sommation que t=0, mais avec l'indice du cluster SS[flat_idx])

                    // d) Décision d'Acceptation
                    // ... (calcul du tp et décision finale) ...
                } catch (const std::exception& e) {
                    std::cerr << "Erreur fullPsiPoll (t>0, i=" << i << ", t=" << t_plus_1 << "): " << e.what() << ". Rejet." << std::endl;
                }
            } else {
                // L'état est lié, il prend son paramètre
                psi[i][t_plus_1] = psi[i0_t0.first][i0_t0.second];
            }
        }
    }
    return psi;
}







