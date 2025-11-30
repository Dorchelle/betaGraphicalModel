#include <iostream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <numeric>
#include "fonctions.h"
#include "prior.h"
#include "fullConditional.h"
#include "modeleBetaGaussian.h"
#include "sampling.h" // function for sampling
#include "usefullFunctions.h" // usefull functions

// Déclarations des helpers nécessaires (doivent être définis ailleurs)
extern std::mt19937 generator_prior; 
extern size_t countNonzeroClusterJ(const Vector& SS, int j);
extern double sampleBeta(double alpha, double beta);
extern int sampleDiscrete(const Vector& probabilities); 
extern double trace(const Matrix& M); 
extern Vector calculateCs(const Vector& SS);
extern Matrix AddMatrices(const Matrix& A, const Matrix& B); 
// Note: Les fonctions MCMC et kmeansSS doivent être traduites avec des structures de retour cohérentes.
BetaGaussienResult betaGaussien(const std::string& data1, const std::string& data2, 
    size_t burn_in = 1000, 
    size_t n_sample = 1000, 
    HyperParameters defaults, 
    bool State = true
) {
    // defaults is the set of hyper parameters
    // --- 0. Chargement des données et Initialisation des constantes ---
    DonneesChargees Donnees = chargesDonnees(data1, data2, State);
    const Matrix& X = Donnees.X;
    int l = Donnees.l;
    const Matrix& IDH = Donnees.IDH;
    const Vector& IDHPre = Donnees.IDHPre;
    int T1 = Donnees.T1;
    KmeansResult Kmeans =kmeansSS(X, T1, 20);
    size_t p = X[0].size(); // Nombre de régions
    double p_double = (double)p;
    
    // Initialisation du clustering (nécessite la traduction de f.kmeansSS)
    // KmeansResult Kmeans = f.kmeansSS(X, T1);
    // Vector SS = Kmeans.SS; size_t Ncl = Kmeans.Ncl; Matrix A_Theta = Kmeans.ma;
    // PLACEHOLDER: Utilisation de valeurs simplifiées
    Vector SS= Kmeans.SS; 
    size_t Ncl = Kmeans.Ncl;
    Vector Cs= Kmeans.CS;
    Matrix A_Theta(p, Vector(p, 1.0));
    for (size_t i = 0; i < p; ++i) {
     for (size_t j = 0; j < p; ++j)  A_Theta[i][j] = Kmeans.ma[i][j]; 
    }    
    // Initialisation des Priors G-Wishart/Wishart
    Matrix D_wishart(p, Vector(p, 0.0));
    Matrix T_wishart(p, Vector(p, 0.0));
    for (size_t i = 0; i < p; ++i) { T_wishart[i][i] = 1.0 / std::sqrt(2.0);
     D_wishart[i][i] =2;
    }
    double b = 2.0 * ((double)l + 1.0); // b_j=2*(l+1)
    double b_j = b;
    
    // Psi, B, detB, R (PrecisionMatrix)
    Matrix Psi = EchantillonPsi(A_Theta, T_wishart, b);
    MatricesPrecision MP = precisionMatrix(Psi); 
    Matrix B = MP.Omega;                        
    double detB = 0.0;                          
    Matrix R = MP.Sigma;                        
    
    // Paramètres MH/Gibbs
    double b_xi = - defaults.a_xi;
    // Initialisation des chaînes MCMC
    double sigma_1 = priorSigma(defaults.a_sigma_1, defaults.b_sigma_1);
    double sigma_2 = priorSigma(defaults.a_sigma_2, defaults.b_sigma_2);
    
    Vector xi_1 = priorXI(defaults.a_xi, b_xi, (int)p);
    Vector xi_2 = priorXI(defaults.a_xi, b_xi, (int)p);
    // Lambda et Theta (Log-Normal variables latentes)
    Matrix lambda_1 = priorLogNormal(B, sigma_1, xi_1[0], T1); 
    Matrix theta = priorLogNormal(B, sigma_2, xi_2[0], T1); 
    // Priors de Cluster (alpha et mu)
    Vector alpha = priorGamma(theta, defaults.b_alpha, SS);
    Vector mu = priorBeta(lambda_1, defaults.b_mu, SS);
    Vector ss_temp(p*(T1+1),0);
    Vector CS = calculateCs(SS); // Taille des clusters
    ////////////////////// SAVING VARIABLES
    Matrix mu_save(p, Vector(n_sample, 0.0));
    Matrix SS_matrix(n_sample, Vector(SS.size(), 0.0));
    Matrix mu_matrix(n_sample, Vector(mu.size(), 0.0)); // Utilise la taille initiale de mu
    Matrix alpha_matrix(n_sample, Vector(alpha.size(), 0.0));
    Matrix crps_sample(n_sample, Vector(p, 0.0));

    // ------------ BURNT-IN STEP ---
    std::cout << "############################# BEGIN BURN-IN STEP ##############################################" << std::endl;
    for (size_t i_0 = 0; i_0 < burn_in; ++i_0) 
    {
  
        // 1. Full Partition (Clustering)
        PriorPartitionResult Partition = fullPartition(b, defaults.M, SS, CS, IDH, mu, defaults.b_mu, alpha, defaults.b_alpha, lambda_1, theta, X,defaults.m_neal);
        CS = Partition.CS_r; Ncl = Partition.Ncl; mu = Partition.mu_r; alpha = Partition.alpha_r;
        
        // 2. Mises à jour MH/Gibbs
        mu = fullMu(IDH, lambda_1, mu, defaults.b_mu, alpha, SS);
        alpha = fullAlpha(IDH, theta, alpha, defaults.b_alpha, mu, SS);
        lambda_1 = fullLambda(mu,  defaults.b_mu, xi_1, B, lambda_1, sigma_1, SS, T1);
        theta = fullTheta(alpha, defaults.b_alpha, xi_2, B, theta, sigma_2, SS, T1);
        xi_1 = fullXi(xi_1,defaults.a_xi, b_xi, B, lambda_1, sigma_1);
        xi_2 = fullXi(xi_2,defaults.a_xi, b_xi, B, theta, sigma_2);
        //fullSigma(double a_sigma, double b_sigma, double sigma, const Matrix& lambda_1, int T1, int p, const Matrix& B, Vector xi)
        sigma_1 = fullSigma(defaults.a_sigma_1, defaults.b_sigma_1, sigma_1, lambda_1,T1,(int)p, B, xi_1);
        sigma_2 = fullSigma(defaults.a_sigma_2,defaults.b_sigma_2, sigma_2, theta,T1,(int) p, B, xi_2);
        // 3. Mise à jour de la matrice de précision B (MCMC)
        Matrix S_epsi1 = AddMatrices(DivideMatrixByScalar(S_epsi(lambda_1, xi_1),sigma_1), DivideMatrixByScalar(S_epsi(theta, xi_2),sigma_2));
        MCMC_Result_Full  ResultMC= MCMC(defaults.lambda_0, B, R, detB,Psi,T_wishart,b,S_epsi1 ,defaults.neta,A_Theta, T1);
        A_Theta=ResultMC.A_Theta;
        detB =ResultMC.detB;
        R=ResultMC.Sigma;
        B=ResultMC.B;
        T_wishart = ResultMC.T;
        Psi=ResultMC.Psi;
        // Note: La fonction MCMC originale retourne 6 valeurs. Nous utilisons le placeholder.
        // MCMC_Result mcmc_res = MCMC(...); 
    }
    std::cout << "############################# END BURN-IN STEP #####################################################" << std::endl;
    /// CONTINUEZ ICI DEMAIN
    //////
    ////
    // --- SAMPLING STEP ---
    for (size_t i_0 = 0; i_0 < n_sample; ++i_0) {
         PriorPartitionResult Partition = fullPartition(b, defaults.M, SS, CS, IDH, mu, defaults.b_mu, alpha, defaults.b_alpha, lambda_1, theta, X,defaults.m_neal);
        CS = Partition.CS_r; Ncl = Partition.Ncl; mu = Partition.mu_r; alpha = Partition.alpha_r;
        // 2. Mises à jour MH/Gibbs
        mu = fullMu(IDH, lambda_1, mu, defaults.b_mu, alpha, SS);
        alpha = fullAlpha(IDH, theta, alpha, defaults.b_alpha, mu, SS);
        lambda_1 = fullLambda(mu,  defaults.b_mu, xi_1, B, lambda_1, sigma_1, SS, T1);
        theta = fullTheta(alpha, defaults.b_alpha, xi_2, B, theta, sigma_2, SS, T1);
        xi_1 = fullXi(xi_1,defaults.a_xi, b_xi, B, lambda_1, sigma_1);
        xi_2 = fullXi(xi_2,defaults.a_xi, b_xi, B, theta, sigma_2);
        //fullSigma(double a_sigma, double b_sigma, double sigma, const Matrix& lambda_1, int T1, int p, const Matrix& B, Vector xi)
        sigma_1 = fullSigma(defaults.a_sigma_1, defaults.b_sigma_1, sigma_1, lambda_1,T1,(int)p, B, xi_1);
        sigma_2 = fullSigma(defaults.a_sigma_2,defaults.b_sigma_2, sigma_2, theta,T1,(int) p, B, xi_2);
        // 3. Mise à jour de la matrice de précision B (MCMC)
        Matrix S_epsi1 = AddMatrices(DivideMatrixByScalar(S_epsi(lambda_1, xi_1),sigma_1), DivideMatrixByScalar(S_epsi(theta, xi_2),sigma_2));
        MCMC_Result_Full  ResultMC= MCMC(defaults.lambda_0, B, R, detB,Psi,T_wishart,b,S_epsi1 ,defaults.neta,A_Theta, T1);
        A_Theta=ResultMC.A_Theta;detB =ResultMC.detB;R=ResultMC.Sigma;B=ResultMC.B;T_wishart = ResultMC.T;Psi=ResultMC.Psi;
        bool Statee=false;
        for (int k_0=0; k_0<p;k_0++)
        {
           Vector tp(Ncl, 0.0);
           Vector N1(Ncl, 0.0);
           for (size_t k = 0; k < p * T1; ++k) ss_temp[k] = SS[k];   // initialization
           int pt=(int)p*T1;
           for (int j = 0; j < (int)Ncl; ++j) 
           {
                // Affectation/Désaffectation temporaire dans ss_temp
                for(int k_00=pt+k_0; k_00<(pt+T1);++k_00) ss_temp[k_00]=-1;
                double t_0=0;
        
                // Calcul du terme initial: log(CS[j]) + log(M) - (numerateur - denominateur)
                tp[j] = std::log(CS[j]) + std::log(defaults.M) - (numerateur(Cs, j,b_j, l) - denominateur(ss_temp, X, j, CS, l, b_j));
                
                // Affectation temporaire du point de prédiction (T1+1)
                ss_temp[pt + k_0] = (double)j;
                CS[j] += 1.0;
            
                // Terme final: + (numerateur* - denominateur*)
                tp[j] += numerateur(Cs, j,b_j, l)- denominateur(ss_temp, X, j, CS, l, b_j);
    
                CS[j] -= 1.0; // Rétablissement
                ss_temp[pt + k_0] = -1;
            }
            bool Statee = false;  
            for (double val : tp) {
                if (std::isnan(val) || val == -std::numeric_limits<double>::infinity()) {
                    Statee = true;
                    break;
                }
            } 
             // Statee=np.any(np.isnantp)
            if (Statee == true)
             {
                std::cout << "probability has NaN for the region: " << k_0 << std::endl;
                if (i_0 != 0) 
                {

                    // mu_save[k_0,i_0] = np.mean(mu_save[k_0,0:i_0])
                    double sum_mu = 0.0;
                    // La moyenne est calculée sur les i_0 échantillons précédents [0, i_0)
                    for (size_t prev_i = 0; prev_i < i_0; ++prev_i) sum_mu += mu_save[k_0][prev_i];
                    
                    mu_save[k_0][i_0] = sum_mu / (double)i_0;
                }  // end if i_0!=0
         
            }  // end if for Statee=True
            else
            {
                if (tp.size()!=1) 
                {
                    double max_log_prob = tp[0];
                    for (double val : tp) if (val > max_log_prob) max_log_prob = val; 
                    Vector norm_prob((size_t)Ncl, 0.0);
                    double sum_exp_prob = 0.0;
                    for (int j = 0; j < (int)Ncl; ++j) 
                    {
                        // prob = exp(log_prob - max_log_prob)
                        norm_prob[j] = std::exp(tp[j] - max_log_prob); 
                        sum_exp_prob += norm_prob[j];
                    } // end j

                    // Normalisation finale : prob = np.exp(prob)/np.sum(np.exp(prob))
                    if (sum_exp_prob < 1e-12)
                    {
                        // Cas d'échec de normalisation (toutes les probabilités sont nulles après exp)
                        std::cerr << "Toutes les probabilités sont nulles après exponentielle pour la région " << k_0 << ". Échantillonnage uniforme." << std::endl;
                        for (int j = 0; j < (int)Ncl; ++j) norm_prob[j] = 1.0 / (double)Ncl;
                        sum_exp_prob = 1.0;
                    } else 
                    {
                        for (int j = 0; j < (int)Ncl; ++j)
                        {
                            norm_prob[j] /= sum_exp_prob;
                            tp[j] =norm_prob[j];
                        } 
                        
                    }  // tp=1


                }// end if normalization
                else tp[0]=1; // just and element
                //END NORMALIZATION  
                std::discrete_distribution<> dist(tp.begin(), tp.end());
                int j_0 = dist(generator_prior); 
                // c) Sauvegarde du mu prédit
                if (j_0 >= 0 && (size_t)j_0 < mu.size())  mu_save[k_0][i_0] = mu[j_0];  
            } //end STATE ==true
            // 1. Calcul des probabilités de cluster pour la nouvelle période (T1+1)
        } // end k_0

 // ####### CRPS PARAMETERS #############
        // L'indice plat de la ligne de sauvegarde est i_0.
        size_t row_index = i_0;
        
        // Ncl est la taille dynamique des vecteurs mu et alpha.
        size_t Ncl_size = Ncl; 
        
        // --- 1. Sauvegarde de la Partition SS (Taille p*T1) ---
        // SS_matrix[i_0, :] = SS
        // Nous copions tout le vecteur SS (taille p*T1) dans la ligne i_0 de SS_matrix.
        if (row_index < SS_matrix.size() && SS.size() == SS_matrix[row_index].size()) std::copy(SS.begin(), SS.end(), SS_matrix[row_index].begin());
        // --- 2. Sauvegarde des Moyennes Mu (Taille Ncl) ---
        // mu_matrix[i_0, 0:Ncl] = mu
        if (row_index < mu_matrix.size() && Ncl_size <= mu_matrix[row_index].size()) {
            // Copie seulement les Ncl premiers éléments
            std::copy(mu.begin(), mu.begin() + Ncl_size, mu_matrix[row_index].begin());
        }
        // --- 3. Sauvegarde des Paramètres Alpha (Taille Ncl) ---
        // alpha_matrix[i_0, 0:Ncl] = alpha
        if (row_index < alpha_matrix.size() && Ncl_size <= alpha_matrix[row_index].size()) {
            // Copie seulement les Ncl premiers éléments
            std::copy(alpha.begin(), alpha.begin() + Ncl_size, alpha_matrix[row_index].begin());
        }
    }  // end i_0
    
    // WE ARE COMPUTING DE CRPS
    size_t taille_totale = (size_t)p * ((size_t)T1 + 1); 
    // Créer le vecteur SS de la taille spécifiée, où chaque élément vaut -1.0
    Vector SS(taille_totale, -1.0);
    size_t p_T1 = (size_t)p * (size_t)T1;
    // --- Étape de Prédiction (Forecasting) et Sauvegarde ---
    for (size_t k_0 = 0; k_0 < p; ++k_0) {
        for(size_t i_0=0; i_0 <n_sample;++i_0)
        {
            // Calcul de la taille de la partition à un instant T1
            // 1. n_0 = random.randint(0, n_sample - 1)
            // Génère un entier aléatoire dans l'intervalle [0, n_sample - 1]
            std::uniform_int_distribution<> distrib_sample(0, (int)n_sample - 1);
            size_t n_0 = distrib_sample(generator_prior);   // choose the corresponding clustering
            // 2. SS[0:(p*T1)] = SS_matrix[n_0, :]
            // Copie la ligne n_0 de SS_matrix dans le vecteur de travail SS
            if (n_0 < SS_matrix.size() && p_T1 <= SS.size()) {
                std::copy(SS_matrix[n_0].begin(), SS_matrix[n_0].begin() + p_T1, SS.begin());
            } else {
                throw std::out_of_range("Erreur: Indice de sample n_0 ou taille SS invalide lors de la prédiction CRPS.");
            }

            // 3. Ncl = len(set(SS_matrix[n_0, :])) et CS = np.zeros(Ncl)
            // Utiliser calculateCs sur le vecteur SS mis à jour pour obtenir Ncl et CS
            Vector CS = calculateCs(SS);
            size_t Ncl = CS.size(); // Le nombre de clusters (Ncl)
            Vector tp(Ncl, 0.0);       // tp = np.zeros(Ncl)
            for (size_t j=0;j<Ncl;++j)
            {
                updateClusterSizeJ(CS,SS,j);
                tp[j] = std::log(CS[j]) - (numerateur(CS, j,b_j, l) - denominateur(SS, X, j, CS, l, b_j));
                SS[p_T1+k_0]=(double)j;
                CS[j] += 1.0;
                // Terme final: + (numerateur* - denominateur*)
                tp[j] += numerateur(CS, j,b_j, l)- denominateur(SS, X, j, CS, l, b_j);
                CS[j] -= 1.0; // Rétablissement
                SS[p_T1 + k_0] = -1;
            } // end for  j
            bool Statee = false;  
            for (double val : tp) {
                if (std::isnan(val) || val == -std::numeric_limits<double>::infinity()) {
                    Statee = true;
                    break;
                }
            } 
             // Statee=np.any(np.isnantp)
            if (Statee == true)
             {
                std::cout << "probability has NaN for the region: " << k_0 << std::endl;
            }  // end if for Statee=True
            else
            {
                if (tp.size()!=1) 
                {
                    double max_log_prob = tp[0];
                    for (double val : tp) if (val > max_log_prob) max_log_prob = val; 
                    Vector norm_prob((size_t)Ncl, 0.0);
                    double sum_exp_prob = 0.0;
                    for (int j = 0; j < (int)Ncl; ++j) 
                    {
                        // prob = exp(log_prob - max_log_prob)
                        norm_prob[j] = std::exp(tp[j] - max_log_prob); 
                        sum_exp_prob += norm_prob[j];
                    } // end j

                    // Normalisation finale : prob = np.exp(prob)/np.sum(np.exp(prob))
                    if (sum_exp_prob < 1e-12)
                    {
                        // Cas d'échec de normalisation (toutes les probabilités sont nulles après exp)
                        std::cerr << "Toutes les probabilités sont nulles après exponentielle pour la région " << k_0 << ". Échantillonnage uniforme." << std::endl;
                        for (int j = 0; j < (int)Ncl; ++j) norm_prob[j] = 1.0 / (double)Ncl;
                        sum_exp_prob = 1.0;
                    } else 
                    {
                        for (int j = 0; j < (int)Ncl; ++j)
                        {
                            norm_prob[j] /= sum_exp_prob;
                            tp[j] =norm_prob[j];
                        } 
                        
                    }  // tp=1


                }// end if normalization
                else tp[0]=1; // just and element
                //END NORMALIZATION  
                std::discrete_distribution<> dist(tp.begin(), tp.end());
                int j_0 = dist(generator_prior); 
                // c) Sauvegarde du mu prédit
                if (j_0 >= 0 && (size_t)j_0 < mu.size()) 
                {
                    double mu_k0=mu_matrix[n_0][j_0];
                    double alpha_k0=alpha_matrix[n_0][j_0];
                    // 2. Calcul du paramètre Beta (Beta = alpha * (1/mu - 1))
                    double beta_param = alpha_k0 * (-1.0 + 1.0 / mu_k0);
                    crps_sample[i_0][k_0] = sampleBeta(alpha_k0, beta_param);
                     
                }   // end if j_0
            } //end STATE ==true
            // 1. Calcul des probabilités de cluster pour la nouvelle période (T1+1)
        } // end i_0

    }  // end k_0
    // Assembler le résultat final
    BetaGaussienResult final_result;
    final_result.mu_save = mu_save;
    final_result.crps_sample = crps_sample;
    final_result.n_sample_used = (int)n_sample;
    final_result.IDHPre = IDHPre;
    std::cout << "################ THANKS FOR THE COLLABORATION #####################" << std::endl;
    return final_result;
}