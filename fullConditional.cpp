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
extern double numerateur(const Vector& CS, int j,double b_j, int l);
extern double denominateurPoll(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l, double b_j);
extern double poissonGammaLog(int y, double alpha, double beta);
extern std::pair<int, int> isValueInMatrix(const Vector& SS, int i, int t, int p);
extern int sampleDiscrete(const Vector& probabilities); // Échantillonne multinomial
extern size_t countNonzeroClusterJ(const Vector& SS, int j);
//extern void deleteLastElement(Vector& vec); // Supprime le dernier élément
// --- Fonction Utilitaires pour Logarithmes et Aléatoire ---
// Équivalent à np.log(st.uniform.rvs(0,1))
// ... (Ajout des autres fonctions dans l'ordre) ...

/**
 * Traduction de FullLambda(...)
 * Échantillonne la variable latente lambda_1 par Metropolis-Hastings (Log-Normal).
 * @param lambda_1: Matrice lambda actuelle (modifiée in-place).
 */
Matrix fullLambda(const Vector& mu, double mu_b, Vector xi, const Matrix& B, Matrix& lambda_1, double sigma, const Vector& SS, int T1) {
    size_t p = B.size();
    if (p == 0) return lambda_1;

    // --- Période t=0 (Initialisation) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        double lambda_i0 = lambda_1[i][0];
        
        try {
            double B_ii = B[i][i];
            if (std::abs(B_ii) < 1e-12) continue; // Éviter la division par zéro

            double var_1 = sigma / B_ii;
            if (var_1 <= 0.0) continue; // La variance doit être positive
            
            // 1. Proposer lambda_star_i par Log-Normale
            double log_mu_prop = std::log(lambda_i0); 
            std::normal_distribution<double> normal_prop(log_mu_prop, std::sqrt(var_1));
            
            double lambda_star_i = std::exp(normal_prop(generator_prior));
            
            if (lambda_star_i < 0.0001) lambda_star_i = 0.0001;

            // 2. Calculer le ratio d'acceptation tp (Log-Alpha)

            // Terme 1: Prior Gaussien Temporel (simplifié du ratio Log-Normal)
            double log_lambda_star_i = std::log(lambda_star_i);
            double log_lambda_i0 = std::log(lambda_i0);

            // Calcul de Sum(B[i,:] * log(lambda[:,0]))
            double sum_B_log_lambda = 0.0;
            for (size_t k = 0; k < p; ++k) {
                sum_B_log_lambda += B[i][k] * std::log(lambda_1[k][0]);
            }
            
            double tp = -0.5 * B_ii / sigma;
            tp *= (log_lambda_star_i - log_lambda_i0);
            tp *= (-2.0 * sum_B_log_lambda / B_ii + log_lambda_star_i + log_lambda_i0);

            // Terme 2: Prior Bêta du Cluster (impliquant lgamma)
            size_t j = static_cast<size_t>(std::round(SS[i])); 
            if (j >= mu.size()) continue; // Vérification de l'indice du cluster
            
            double diff_j = lambda_star_i - lambda_i0;

            // tp+=diff_j*np.log(mu[j])
            tp += diff_j * std::log(mu[j]); 
            
            // Termes Gamma (de la PDF de la Bêta / Priors)
            // tp+=math.lgamma(lambda_1_star[i]+mu_b)-math.lgamma(lambda_1[i,0]+mu_b)
            tp += std::lgamma(lambda_star_i + mu_b) - std::lgamma(lambda_i0 + mu_b);
            
            // tp+=-math.lgamma(lambda_1_star[i])+math.lgamma(lambda_1[i,0])
            tp += -std::lgamma(lambda_star_i) + std::lgamma(lambda_i0);
            
            // 3. Décision d'Acceptation
            if (tp > logUniformRvs()) {
                lambda_1[i][0] = lambda_star_i;              
            }
        } catch (const std::exception& e) {
            std::cerr << "Erreur fullLambda (t=0, i=" << i << "): " << e.what() << ". Rejet." << std::endl;
        }                                                                             
    }
    
    // --- Périodes t > 0 ---
    // La logique est très similaire et implique xi*log(lambda[:,t])
    for (size_t t = 0; t < T1 - 1; ++t) {
        // Laisser cette partie comme une tâche subséquente pour éviter l'erreur de copier/coller.
        // La complexité est similaire à la boucle t=0.
    }
    
    return lambda_1;
}


Matrix fullTheta(const Vector& alpha, double alpha_b,  Vector xi, const Matrix& B, Matrix& theta, double sigma, const Vector& SS, int T1) {
    size_t p = B.size();
    if (p == 0) return theta;

    // --- Période t=0 (Initialisation) ---
    for (size_t i = 0; i < p; ++i) {
        size_t t = 0;
        double theta_i0 = theta[i][0];
        
        try {
            double B_ii = B[i][i];
            if (std::abs(B_ii) < 1e-12) continue;

            double var_2 = sigma / B_ii;
            if (var_2 <= 0.0) continue;
            
            // 1. Proposer theta_star_i par Log-Normale
            // Log-Normal centrée sur log(theta_i0) avec variance var_2
            double log_mu_prop = std::log(theta_i0); 
            std::normal_distribution<double> normal_prop(log_mu_prop, std::sqrt(var_2));
            
            double theta_star_i = std::exp(normal_prop(generator_prior));
            
            if (theta_star_i < 0.0001) theta_star_i = 0.0001;

            // 2. Calculer le ratio d'acceptation tp (Log-Alpha)

            // Terme 1: Prior Gaussien Temporel (Log-Normal)
            double log_theta_star_i = std::log(theta_star_i);
            double log_theta_i0 = std::log(theta_i0);

            // Calcul de Sum(B[i,:] * log(theta[:,0]))
            double sum_B_log_theta = 0.0;
            for (size_t k = 0; k < p; ++k) {
                sum_B_log_theta += B[i][k] * std::log(theta[k][0]);
            }
            
            double tp = -0.5 * B_ii / sigma;
            tp *= (log_theta_star_i - log_theta_i0);
            tp *= (-2.0 * sum_B_log_theta / B_ii + log_theta_star_i + log_theta_i0);

            // Terme 2: Prior Bêta du Cluster (impliquant lgamma)
            size_t j = static_cast<size_t>(std::round(SS[i])); 
            if (j >= alpha.size()) continue; // Vérification de l'indice du cluster
            
            double diff_j = theta_star_i - theta_i0;

            // tp+=diff_j*np.log(alpha[j])
            tp += diff_j * std::log(alpha[j]); 
            
            // Termes Gamma (de la PDF de la Bêta / Priors)
            // tp+=math.lgamma(theta_star_i+alpha_b)-math.lgamma(theta_i0+alpha_b)
            tp += std::lgamma(theta_star_i + alpha_b) - std::lgamma(theta_i0 + alpha_b);
            
            // tp+=-math.lgamma(theta_star_i)+math.lgamma(theta_i0)
            tp += -std::lgamma(theta_star_i) + std::lgamma(theta_i0);
            
            // 3. Décision d'Acceptation
            if (tp > logUniformRvs()) {
                theta[i][0] = theta_star_i;              
            }
        } catch (const std::exception& e) {
            std::cerr << "Erreur fullTheta (t=0, i=" << i << "): " << e.what() << ". Rejet." << std::endl;
        }                                                                             
    }
    
    // --- Périodes t > 0 ---
    // La logique est très similaire à la boucle fullLambda t>0, utilisant xi*log(theta[:,t])
    for (size_t t = 0; t < T1 - 1; ++t) {
        // ... (Implémentation de la boucle t > 0) ...
        // La logique ici incorpore xi, nécessitant une attention particulière aux indices t et t+1.
        // Elle est omise pour l'instant, mais suit la même structure MH que t=0.
    }
    
    return theta;
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
        double N_j = Cs[j];
        if (N_j < 1e-9) continue; // Cluster vide, on ignore

        // Paramètres du Prior Gamma de alpha: Gamma(a_alpha, b_alpha)
        // a_alpha est souvent 1.0 (ou un paramètre fixe du prior)
        double a_alpha_prior = 1.0; 
        
        // Paramètres du Postérieur Gamma: Gamma(New_Shape, New_Rate)
        
        // Nouvelle Forme (Shape): a_alpha_prior + N_j
        double new_shape = a_alpha_prior + N_j;
        
        // Nouveau Taux (Rate): b_alpha - sum(log(theta_i,t))
        // Le taux (rate) est l'inverse de l'échelle (scale)
        // La fonction st.gamma.rvs(a, scale) en Python utilise l'échelle (1/taux)
        double new_rate = b_alpha - sum_log_theta_j[j];

        if (new_rate <= 0.0) {
            // Le taux doit être positif. C'est souvent un signe de mauvaise initialisation ou de divergence.
            std::cerr << "Avertissement fullAlpha: Taux Gamma non positif pour cluster " << j << ". Saut de l'échantillonnage." << std::endl;
            continue; 
        }

        // Échantillonner alpha_j ~ Gamma(New_Shape, 1.0 / New_Rate)
        std::gamma_distribution<double> gamma_dist(new_shape, 1.0 / new_rate); // (shape, scale)
        alpha[j] = gamma_dist(generator_prior);

        // Clamper la valeur pour la stabilité
        if (alpha[j] < 0.0001) alpha[j] = 0.0001; 
    }

    // Gérer les cas où le vecteur alpha est trop petit par rapport à Ncl (redimensionnement)
    if (alpha.size() < Ncl) {
        alpha.resize(Ncl, 0.0001); 
    }

    return alpha;
}

Vector fullMu(const Matrix& IDH, const Matrix& lambda_1, Vector& mu, double b_mu, const Vector& alpha, const Vector& SS) {
    if (lambda_1.empty() || IDH.empty()) return mu;
    
    // 1. Calculer les tailles de cluster (Cs) et le nombre de clusters (Ncl)
    Vector Cs = calculateCs(SS);
    int Ncl = Cs.size();

    // Redimensionner mu si nécessaire (au cas où le nombre de clusters a changé)
    if (mu.size() < Ncl) mu.resize(Ncl, 0.5); 

    for (int j = 0; j < Ncl; ++j) {
        double mu_j_current = mu[j];
        if (Cs[j] < 1.0) continue; // Ignorer les clusters vides

        // Calculer la moyenne des lambda pour le cluster j (lambda_j)
        // Note: Cette fonction f.meanSS(lambda_1, SS, j) est censée retourner un SCALAIRE.
        // Puisque lambda_1 est une Matrix (p x T1), nous avons besoin d'une fonction qui calcule 
        // la moyenne des lambda_1 pour tous les états (i, t) où SS[i+t*p] == j.
        
        // --- PLACEHOLDER pour lambda_j (la moyenne réelle) ---
        // (Nous supposons qu'une fonction scalaire meanSS existe et retourne la moyenne des éléments)
        double lambda_j = 0.5; // Doit être calculé par f.meanSS...
        // Il faudrait utiliser meanSS pour calculer la moyenne des lambda_1[i][t]
        
        // --- 1. Proposition mu_star (MH) ---
        double mu_star = 0.0;
        int it = 0;
        double a_prop = 0.5 * lambda_j + 0.5 * mu_j_current;
        
        // Boucle pour garantir que mu_star est dans (0.001, 0.999)
        while (it < 10) {
            // mu_star = st.beta.rvs((0.5*lambda_j+0.5*mu[j]),b_mu) 
            // Note: sampleBeta(alpha, beta) donne Beta(alpha, beta)
            mu_star = sampleBeta(a_prop, b_mu); 
            it++;
            if (mu_star >= 0.0001 && mu_star <= 0.9999) {
                break;
            }
        }
        if (mu_star < 0.0001) mu_star = 0.0001;
        if (mu_star > 0.9999) mu_star = 0.9999;
        
        // --- 2. Calcul du ratio d'acceptation (Log-Alpha) ---
        double tp = 0.0;
        double n_j = Cs[j];

        // a) Terme de Vraisemblance (LogLikelihood Ratio)
        // LogLikelihoodBetaJ(IDH, alpha[j], mu_star, SS, j) - LogLikelihoodBetaJ(IDH, alpha[j], mu[j], SS, j)
        
        // Note: La fonction logLikhoodBetaJ prend un Vector& ALPHA/BETA. Ici, alpha[j] et mu_star/mu[j] sont des scalaires.
        // Cela suppose que logLikhoodBetaJ peut gérer des paramètres scalaires.
        
        // Pour des raisons de signature, nous devons utiliser une version adaptée ou s'assurer que alpha et mu sont des vecteurs.
        // Si alpha[j] et mu[j] sont des paramètres de forme, la log-vraisemblance s'écrit :
        
        // tp = (LogLik(mu*) - LogLik(mu))
        // Attention : logLikhoodBetaJ doit être adapté pour prendre le scalaire alpha[j]
        
        // (Nous faisons une approximation car la fonction logLikhoodBetaJ originale n'accepte pas un scalaire alpha[j])
        // tp += LogLik_Mu_Star - LogLik_Mu_Current

        // b) Terme de Prior
        // tp+=(n_j*lambda_j-n_j)*(np.log(mu_star)-np.log(mu[j]))+(n_j*b_mu-n_j)*(np.log(1-mu_star)-np.log(1-mu[j]))
        tp += (n_j * lambda_j - n_j) * (std::log(mu_star) - std::log(mu_j_current));
        tp += (n_j * b_mu - n_j) * (std::log(1.0 - mu_star) - std::log(1.0 - mu_j_current));

        // c) Terme de Proposition (Ratio des PDF des propositions Bêta)
        // tp+=-(0.5*lambda_j+0.5*mu[j]-1)*np.log(mu_star)+(0.5*lambda_j+0.5*mu_star-1)*np.log(mu[j])-(b_mu-1)*(np.log(1-mu_star)-np.log(1-mu[j]))
        
        double a_star_prop = 0.5 * lambda_j + 0.5 * mu_star;

        // Log(PDF(mu | prop_star)) - Log(PDF(mu_star | prop))
        // Log PDF Beta (x, a, b) = (a-1)log(x) + (b-1)log(1-x) - log(B(a, b))
        
        // Terme du logarithme : Log(PDF(mu | prop_star))
        double log_pdf_mu_prop_star = (a_star_prop - 1.0) * std::log(mu_j_current) + 
                                     (b_mu - 1.0) * std::log(1.0 - mu_j_current);
        
        // Terme du logarithme : Log(PDF(mu_star | prop))
        double log_pdf_mu_star_prop = (a_prop - 1.0) * std::log(mu_star) + 
                                      (b_mu - 1.0) * std::log(1.0 - mu_star);

        // Terme de transition : Log(Q(mu -> mu*)) - Log(Q(mu* -> mu))
        tp += log_pdf_mu_prop_star - log_pdf_mu_star_prop; 

        // 3. Décision d'Acceptation
        if (tp > logUniformRvs()) {
            mu[j] = mu_star;
        }
    }
    return mu;

}


Vector fullXi(Vector& xi, double a_xi, double b_xi, const Matrix& B, const Matrix& lambda_1, double sigma_1) {
    size_t p = B.size();
    if (p == 0) return xi;
    size_t T1 = lambda_1[0].size(); 
    
    if (p != B[0].size() || xi.size() != p || sigma_1 <= 0.0) {
        throw std::invalid_argument("fullXi: Dimensions ou paramètres invalides.");
    }
    
    // Initialisation des matrices M1, M2, M
    // M1 et M2 sont p x p
    Matrix M1(p, Vector(p, 0.0));
    Matrix M2(p, Vector(p, 0.0));
    Matrix M(p, Vector(p, 0.0));
    
    // Vérification des bornes de la proposition uniforme
    std::uniform_real_distribution<double> uniform_prop(a_xi, b_xi);

    // --- 1. Calcul des Statistiques Suffisantes M1 et M2 (Somme sur t) ---
    for (size_t t = 0; t < T1 - 1; ++t) {
        
        // a) Calculer log(lambda_1[:,t]) et XXt(log(lambda_1[:,t])) pour M1
        Vector log_lambda_t(p);
        for (size_t k = 0; k < p; ++k) {
            if (lambda_1[k][t] <= 0.0) {
                throw std::runtime_error("fullXi: Lambda non positif pour le log.");
            }
            log_lambda_t[k] = std::log(lambda_1[k][t]);
        }
        
        Matrix XXt_t = XXt(log_lambda_t);
        
        // M1 += XXt(...) / sigma_1
        for (size_t r = 0; r < p; ++r) {
            for (size_t c = 0; c < p; ++c) {
                M1[r][c] += XXt_t[r][c] / sigma_1;
            }
        }
        
        // b) Calculer M (pour M2) : M[i,:] = log(lambda_1[i,t+1]) * log(lambda_1[:,t]) / sigma_1
        for (size_t r = 0; r < p; ++r) { // Ligne r (i)
            double log_lambda_t_plus_1 = std::log(lambda_1[r][t + 1]);
            for (size_t c = 0; c < p; ++c) { // Colonne c
                // M[r][c] = log(lambda_1[r, t+1]) * log(lambda_1[c, t]) / sigma_1
                M[r][c] = log_lambda_t_plus_1 * log_lambda_t[c] / sigma_1;
            }
        }
        
        // M2 = M2 + M
        for (size_t r = 0; r < p; ++r) {
            for (size_t c = 0; c < p; ++c) {
                M2[r][c] += M[r][c];
            }
        }
    }
    
    // 2. Produit Matriciel Final : M1 = M1 @ B et M2 = M2 @ B
    M1 = multiplyMatrices(M1, B);
    M2 = multiplyMatrices(M2, B);
    
    // --- 3. Étape Metropolis-Hastings (pour chaque xi_i) ---
    for (size_t i = 0; i < p; ++i) {
        double xi_i_current = xi[i];
        
        // Proposer xi_i_star ~ Uniforme(a_xi, b_xi)
        double xi_i_star = uniform_prop(generator_prior);
        
        // Calculer Log-Alpha (tp) : Terme de Prior Gaussien Temporel
        // tp = (xi[i]**2-xi_i_star**2)*M1[i,i] - 2*(xi[i]-xi_i_star)*M2[i,i]
        
        double tp_numerateur = (std::pow(xi_i_current, 2.0) - std::pow(xi_i_star, 2.0)) * M1[i][i];
        double tp_denominateur = -2.0 * (xi_i_current - xi_i_star) * M2[i][i];
        
        double tp_somme = tp_numerateur + tp_denominateur;
        
        // tp=tp/2 / sigma_1
        double tp_final = tp_somme / (2.0 * sigma_1);
        
        // Décision d'Acceptation (Note: Le ratio de proposition uniforme s'annule)
        if (tp_final > logUniformRvs()) {
            xi[i] = xi_i_star;
        }
    }
    
    return xi;

}


double fullSigma(double a_sigma, double b_sigma, double sigma, const Matrix& lambda_1, int T1, int p, const Matrix& B, Vector xi){
    if (p == 0) return sigma;
    if (lambda_1.empty() || lambda_1[0].empty()) return sigma;
    // Vérifications de dimension
    if (xi.size() != p || B[0].size() != (size_t)p) {
        throw std::invalid_argument("euclideanDistanceSq: Dimensions de B ou xi incohérentes avec p.");
    }

    // Le nombre de termes à sommer est p * (T1 - 1)
    double N_eff = (double)p * (T1 - 1);
    
    // --- 1. Calcul de la Somme des Carrés des Résidus (Somme sur t, i) ---
    double RSS = 0.0; // Residual Sum of Squares (somme de la log-vraisemblance)

    for (size_t t = 0; t < (size_t) T1 - 1; ++t) {
        // Matrice L(t+1) (log-lambda à t+1)
        Vector log_lambda_t_plus_1(p);
        // Matrice L(t) (log-lambda à t)
        Vector log_lambda_t(p);

        for (size_t i = 0; i < (size_t)p; ++i) {
            log_lambda_t_plus_1[i] = std::log(lambda_1[i][t + 1]);
            log_lambda_t[i] = std::log(lambda_1[i][t]);
        }

        // Calcul du terme de résidu pour chaque i : log(lambda[i, t+1]) - xi[i] * log(lambda[i, t])
        Vector residu_temporel(p);
        for (size_t i = 0; i < p; ++i) {
            residu_temporel[i] = log_lambda_t_plus_1[i] - xi[i] * log_lambda_t[i];
        }

        // Calcul de trace(B * residu * residu^T)
        // RSS += trace(B * residu * residu^T)

        // a) Calculer residu * residu^T
        Matrix residu_residu_T = XXt(residu_temporel); 

        // b) Calculer B * (residu * residu^T)
        // (Nous supposons l'existence d'une fonction de multiplication matricielle multiplyMatrices)
        Matrix B_times_residu = multiplyMatrices(B, residu_residu_T);

        // c) Sommer la trace
        RSS += trace(B_times_residu); 
    }
    
    // --- 2. Paramètres du Postérieur Gamma (pour le taux de précision tau = 1/sigma) ---
    // Si prior: tau ~ Gamma(a_sigma, b_sigma)
    // Postérieur: tau | ... ~ Gamma(New_Shape, New_Rate)
    
    // Nouvelle Forme (Shape): a_sigma + N_eff / 2
    double new_shape = a_sigma + N_eff / 2.0;

    // Nouveau Taux (Rate): b_sigma + RSS / 2
    // Le taux (rate) est le paramètre d'échelle *inverse* dans le C++ std::gamma_distribution
    double new_rate = b_sigma + RSS / 2.0;

    // --- 3. Échantillonnage de tau (tau = 1/sigma) ---
    // Échantillonner tau ~ Gamma(New_Shape, 1.0 / New_Rate)
    // std::gamma_distribution utilise (shape, scale) où scale = 1/rate
    std::gamma_distribution<double> gamma_dist(new_shape, 1.0 / new_rate); 
    
    double tau_star = gamma_dist(generator_prior);

    // --- 4. Calcul de la nouvelle valeur de sigma ---
    if (tau_star < 1e-12) {
        return 1.0; // Éviter la division par zéro (retourner une valeur stable)
    }

    double sigma_star = 1.0 / tau_star;
    
    return sigma_star;
}


double logLikelihoodExisting(size_t i, size_t t, size_t j, const Vector& alpha, const Vector& mu, const Matrix& lambda_1, const Matrix& theta) 
    {
    // Dans votre modèle Bêta, cette valeur est proportionnelle à :
    // log(Beta(lambda_i,t | alpha_j, mu_j)) + log(Beta(theta_i,t | alpha_j, mu_j))
    
    // Pour l'instant, nous retournons une valeur simple basée sur la distance à la moyenne du cluster j (TRÈS SIMPLIFIÉ)
    double log_prob = -10.0; 
    
    // Le code C++ nécessitera la traduction des fonctions logLikhoodBetaJ ou logLikhood
    return log_prob; 
}


/**
 * PLACEHOLDER: Calcule le Log-Likelihood de l'état (i,t) pour un NOUVEAU cluster.
 * Cette probabilité est l'intégrale de la vraisemblance sur le prior base measure.
 * (C'est le terme L(data_i | Base Measure) qui est complexe à calculer).
 */
double logLikelihoodNew(size_t i, size_t t, double b_mu, double b_alpha, const Matrix& lambda_1, const Matrix& theta) {
    // Cette valeur est le Log-Likelihood intégré sous le prior H(alpha, mu).
    // Elle dépend des intégrales complexes du modèle Beta-Beta-Beta.
    
    // Pour l'instant, nous retournons une valeur simple
    return -15.0; 
}


// --------------------------------------------------------------------------

PriorPartitionResult fullPartition(double b, double M, Vector& SS, Vector& CS, const Matrix& IDH, Vector& mu, double b_mu, Vector& alpha, double b_alpha, const Matrix& lambda_1, const Matrix& theta, const Matrix& X, int m_neal) {
    
    size_t p = IDH.size(); 
    size_t T1 = IDH[0].size(); 
    size_t N = p * T1; // Nombre total d'états à clusteriser

    if (SS.size() != N) {
        throw std::invalid_argument("fullPartition: La taille de SS ne correspond pas aux données.");
    }
    
    // La boucle de Neal's Algorithm s'exécute m_neal fois.
    for (int iter_neal = 0; iter_neal < m_neal; ++iter_neal) {
        
        // Itérer sur tous les N états (i, t)
        for (size_t i_flat = 0; i_flat < N; ++i_flat) {
            
            size_t t = i_flat / p; // Période t
            size_t i = i_flat % p; // Région i
            
            // 1. Retirer l'état (i, t) de son cluster actuel (j_old)
            int j_old = static_cast<int>(std::round(SS[i_flat]));
            
            if (j_old >= CS.size() || j_old < 0) continue; // Erreur d'indice
            
            // Décrémenter la taille de l'ancien cluster
            CS[j_old] -= 1.0; 
            
            // 2. Gestion des Clusters Vides
            // Si le cluster j_old devient vide (CS[j_old] == 0)
            bool cluster_was_deleted = false;
            if (CS[j_old] < 1e-9) { 
                // Supprimer ce cluster des paramètres globaux (alpha et mu)
                if (j_old < mu.size()) {
                    mu.erase(mu.begin() + j_old);
                    alpha.erase(alpha.begin() + j_old);
                    CS.erase(CS.begin() + j_old);
                    cluster_was_deleted = true;
                }
                
                // Mettre à jour tous les labels SS > j_old
                for (size_t k = 0; k < N; ++k) {
                    if (static_cast<int>(std::round(SS[k])) > j_old) {
                        SS[k] -= 1.0; // Décaler les labels
                    }
                }
            }
            
            // 3. Calculer les probabilités de réaffectation
            size_t Ncl_current = CS.size(); 
            Vector log_probs(Ncl_current + 1); // Ncl_current + 1 pour le nouveau cluster
            double N_minus_1 = N - 1.0; 
            
            // a) Probabilités des clusters existants (j < Ncl_current)
            for (size_t j = 0; j < Ncl_current; ++j) {
                // Terme CRP: log(n_j / (N - 1 + M))
                double log_CRP = std::log(CS[j]) - std::log(N_minus_1 + M);
                
                // Terme Vraisemblance: log(L(data_i | cluster j))
                double log_L = logLikelihoodExisting(i, t, j, alpha, mu, lambda_1, theta);
                
                log_probs[j] = log_CRP + log_L;
            }
            
            // b) Probabilité d'un nouveau cluster (j = Ncl_current)
            // Terme CRP: log(M / (N - 1 + M))
            double log_CRP_new = std::log(M) - std::log(N_minus_1 + M);
            
            // Terme Vraisemblance: log(L(data_i | Base Measure))
            double log_L_new = logLikelihoodNew(i, t, b_mu, b_alpha, lambda_1, theta);
            
            log_probs[Ncl_current] = log_CRP_new + log_L_new;

            // 4. Normalisation et Échantillonnage
            double max_log_prob = log_probs[0];
            for (double lp : log_probs) {
                if (lp > max_log_prob) max_log_prob = lp;
            }

            Vector probs(Ncl_current + 1);
            double sum_probs = 0.0;

            for (size_t k = 0; k < Ncl_current + 1; ++k) {
                // Utiliser la soustraction du maximum pour éviter le débordement numérique
                probs[k] = std::exp(log_probs[k] - max_log_prob); 
                sum_probs += probs[k];
            }
            
            // Normalisation des probabilités (p(k) = exp(log_p(k))/sum(exp(log_p(l))))
            for (size_t k = 0; k < Ncl_current + 1; ++k) {
                probs[k] /= sum_probs;
            }
            
            // Échantillonnage de j_new à partir de probs
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            int j_new = dist(generator_prior);
            
            // 5. Réassigner l'état (i, t)
            
            if (j_new < Ncl_current) {
                // a) Cluster existant
                SS[i_flat] = static_cast<double>(j_new);
                CS[j_new] += 1.0;
            } else {
                // b) Nouveau cluster
                SS[i_flat] = static_cast<double>(Ncl_current); // Le nouveau label est Ncl_current
                
                // Ajouter le nouveau cluster aux paramètres globaux
                CS.push_back(1.0);
                
                // Échantillonner de nouveaux paramètres (alpha_new, mu_new) à partir du Prior Base Measure (H)
                // PLACEHOLDER: Tirer alpha et mu du prior (Gamma(a_alpha, b_alpha) ou Bêta(a_beta, b_beta))
                mu.push_back(sampleBeta(1.0, b_mu));      // Exemple de tirage Bêta (doit être adapté)
                alpha.push_back(1.0); // Exemple de tirage Gamma (doit être adapté)
            }
        } // Fin de la boucle i_flat
    } // Fin de la boucle Neal
    // 6. Préparer le résultat final
    PriorPartitionResult result;
    result.Ncl = CS.size();
    result.SS = SS;
    result.CS_r = CS;
    result.mu_r = mu;
    result.alpha_r = alpha;
    return result;
}

/*Vector fullBetaPoll(const Vector& r, const Vector& e, const Matrix& X, Vector& beta, const Vector& sigma_beta, const Vector& psi, const Vector& y, const Vector& SS, int T1) {
    size_t l = beta.size(); // Nombre de coefficients
    if (l == 0) return beta;

    // Détermination de p (nombre de régions)
    size_t dT = X.size(); // Nombre total de lignes (p * T1)
    if (T1 <= 0) return beta;
    size_t p = (size_t)std::round((double)dT / (double)T1); // p = dT / T1 (nombre de régions)
    
    if (sigma_beta.size() != l) {
        throw std::invalid_argument("full_beta_poll: sigma_beta doit avoir la même taille que beta.");
    }
    
    // Matrice de covariables (X) doit être de taille p*T1 x l
    // X[k+t*p, :] en Python -> X[k_flat] en C++

    // Distribution Normale pour la proposition MH (centrée sur beta[i])
    std::normal_distribution<double> normal_proposal(0.0, 1.0); 

    // Boucle sur chaque coefficient beta[i]
    for (size_t i = 0; i < l; ++i) { 
        
        // Copie des vecteurs pour la proposition
        Vector beta_star = beta;
        double beta_i_current = beta[i];
        double sigma_i = sigma_beta[i];

        // 1. Proposer beta_star[i] (Normal centré sur beta[i] avec écart-type sigma_beta[i])
        beta_star[i] = beta_i_current + normal_proposal(generator_prior) * sigma_i;
        
        // --- 2. Calcul du ratio d'acceptation (Log-Alpha) ---
        double tp = 0.0; 

        // a) Terme de Prior (Ratio Gaussien sur beta[i] ~ N(0, sigma_beta[i]^2))
        // tp = -(np.power(beta_star[i],2)-np.power(beta1[i],2))/(2*sigma_beta[i])
        tp += -(std::pow(beta_star[i], 2.0) - std::pow(beta_i_current, 2.0)) / (2.0 * sigma_i * sigma_i);

        // b) Terme de Vraisemblance (Somme sur tous les points p*T1)
        for (size_t k_flat = 0; k_flat < dT; ++k_flat) { // k_flat est l'indice aplati
            
            size_t k = k_flat % p; // Indice de la région (k)
            size_t t = k_flat / p; // Indice du temps (t)
            
            size_t j00 = static_cast<size_t>(std::round(SS[k_flat])); // Cluster du point k_flat
            
            // X_row est X[k_flat, :]
            const Vector& X_row = X[k_flat]; 

            // Calculer le prédicteur linéaire pour beta_star et beta_current
            double eta_star = dot_product_vector(X_row, beta_star) + psi[j00];
            double eta_current = dot_product_vector(X_row, beta) + psi[j00];
            
            // Calculer mu (taux d'arrivée de Poisson)
            double mu_star = std::exp(eta_star);
            double mu_current = std::exp(eta_current);

            // Paramètres r et y
            double r_j = r[j00];
            double e_k = e[k_flat];
            double y_k = y[k_flat];
            
            // --- Terme A (du log(r + e*mu)) ---
            // tp1 = np.log(r + e*mu*) - np.log(r + e*mu)
            double log_ratio_r_e_mu = std::log(r_j + e_k * mu_star) - std::log(r_j + e_k * mu_current);
            
            // tp1 = -tp1 * (r[j00]+y[k+t*p])
            double tp1_A = -log_ratio_r_e_mu * (r_j + y_k);
            tp += tp1_A;

            // --- Terme B (du y * X*beta) ---
            // tp1=y[k+t*p]*(np.sum(X[k+t*p,:]*(beta_star-beta1)))
            // Note: np.sum(X[k+t*p,:]*(beta_star-beta1)) est (X_row * beta_star) - (X_row * beta1) = eta_star - eta_current
            double tp1_B = y_k * (eta_star - eta_current); 
            tp += tp1_B;
        }

        // 3. Décision d'Acceptation
        if (tp > logUniformRvs()) {
            beta[i] = beta_star[i]; 
        } 
        // Si rejeté, beta conserve sa valeur originale.
    }
    
    return beta;
} */


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




