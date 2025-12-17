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

//extern Matrix fullLambda(const Vector& mu, double mu_b, Vector xi, const Matrix& B, Matrix& lambda_1, double sigma, const Vector& SS, int T1);
BetaGaussienResult betaGaussien(
    const std::string& data1,
    const std::string& data2,
    std::size_t burn_in,
    std::size_t n_sample,
    HyperParameters defaults,
    bool State
) {
    constexpr bool verbose = true; // mets false quand tu veux couper les logs

    // --- 0. Chargement des données et constantes ---
    DonneesChargees Donnees = chargesDonnees(data1, data2, State);

    const Matrix& X      = Donnees.X;
    const Matrix& IDH    = Donnees.IDH;
    const Vector& IDHPre = Donnees.IDHPre;
    const int l          = Donnees.l;
    const int T1         = Donnees.T1;

    const std::size_t p  = IDHPre.size(); // Nombre de régions
    const double p_double = static_cast<double>(p);
    (void)p_double; // si pas utilisé ensuite

    // --- 1. Initialisation du clustering (k-means) ---
    KmeansResult Kmeans = kmeansSS(X, T1, 20);  // nombre initial de groupes

    Vector SS    = Kmeans.SS;
    Vector CS    = Kmeans.CS;
    std::size_t Ncl = Kmeans.Ncl;

    // Copier directement la matrice de voisinage
    Matrix A_Theta = Kmeans.ma;  // supposé p x p

    // --- 2. Priors G-Wishart/Wishart ---
    Matrix D_wishart(p, Vector(p, 0.0));
    Matrix T_wishart(p, Vector(p, 0.0));
    for (std::size_t i = 0; i < p; ++i) {
        T_wishart[i][i] = 1.0 / std::sqrt(2.0);
        D_wishart[i][i] = 2.0;
    }

    const double b   = 2.0 * (static_cast<double>(l) + 1.0);
    const double b_j = b;

    // --- 3. Matrices de précision initiales ---
    Matrix Psi = EchantillonPsi(A_Theta, T_wishart, b);

    MatricesPrecision MP = precisionMatrix(Psi);
    Matrix B   = MP.Omega;
    Matrix R   = MP.Sigma;
    double detB = MP.detb;

    // Normalisation B, R et ajustement detB
    for (std::size_t r = 0; r < p; ++r) {
        const double sqrtSigma_r = std::sqrt(MP.Omega[r][r]);
        for (std::size_t c = 0; c < p; ++c) {
            const double sqrtSigma_c = std::sqrt(MP.Omega[c][c]);
            B[r][c] = MP.Omega[r][c] * sqrtSigma_r * sqrtSigma_c;
            R[r][c] = MP.Sigma[r][c] / (sqrtSigma_r * sqrtSigma_c);
        }
        detB += std::log(MP.Omega[r][r]);
    }

    // --- 4. Hyperparamètres / chaînes MCMC ---
    const double b_xi = -defaults.a_xi;

    double sigma_1 = priorSigma(defaults.a_sigma_1, defaults.b_sigma_1);
    double sigma_2 = priorSigma(defaults.a_sigma_2, defaults.b_sigma_2);

    Vector xi_1 = priorXI(defaults.a_xi, b_xi, static_cast<int>(p));
    Vector xi_2 = priorXI(defaults.a_xi, b_xi, static_cast<int>(p));
    //double sigma_1=1; double sigma_2=1;
    Matrix lambda_1 = priorLogNormal(B, sigma_1, xi_1, T1);
    Matrix theta    = priorLogNormal(B, sigma_2, xi_2, T1);

    Vector alpha = priorGamma(theta, defaults.b_alpha, SS);
    Vector mu    = priorBeta(lambda_1, defaults.b_mu, SS);
    CS=calculateCs(SS);
  //std::cout << "alpha sixe = " << alpha.size() << " ms" << std::endl;
   //std::cout << "mu sixe = " << mu.size() << " ms" << std::endl;
   //std::cout << "CS sixe = " << CS.size() << " ms" << std::endl;
    // Buffers de travail
    const std::size_t pt = p * static_cast<std::size_t>(T1);
    Vector ss_temp(pt + p, -1.0);

    // --- 5. Structures de sauvegarde ---
    Matrix mu_save(p, Vector(n_sample, 0.0));
    Matrix SS_matrix(n_sample, Vector(SS.size(), 0.0));
    Matrix mu_matrix(n_sample, Vector(SS.size(), 0.0));
    Matrix alpha_matrix(n_sample, Vector(SS.size(), 0.0));
    Matrix crps_sample(n_sample, Vector(p, 0.0));

    // --- 6. BURN-IN ---
    if (verbose) {
        std::cout << "############################# BEGIN BURN-IN STEP ##############################################\n";
    }

    for (std::size_t iter = 0; iter < burn_in; ++iter) {
        // 1. Clustering
        PriorPartitionResult Partition = fullPartition(
            b, defaults.M, SS, CS, IDH,
            mu, defaults.b_mu, alpha, defaults.b_alpha,
            lambda_1, theta, X, defaults.m_neal
        );
        CS    = Partition.CS_r;
        Ncl   = Partition.Ncl;
        mu    = Partition.mu_r;
        alpha = Partition.alpha_r;
        SS    = Partition.SS;

        // 2. Mises à jour MH/Gibbs
        mu    = fullMu(IDH, lambda_1, mu, defaults.b_mu, alpha, SS);
        alpha = fullAlpha(IDH, theta, alpha, defaults.b_alpha, mu, SS);

        lambda_1 = fullLambda(mu, defaults.b_mu, xi_1, B, lambda_1, sigma_1, SS, T1);
        theta    = fullTheta(alpha, defaults.b_alpha, xi_2, B, theta, sigma_2, SS, T1);

        xi_1 = fullXi(xi_1, defaults.a_xi, b_xi, B, lambda_1, sigma_1);
        xi_2 = fullXi(xi_2, defaults.a_xi, b_xi, B, theta,    sigma_2);

        sigma_1 = fullSigma(defaults.a_sigma_1, defaults.b_sigma_1, sigma_1, lambda_1, T1, static_cast<int>(p), B, xi_1);
        sigma_2 = fullSigma(defaults.a_sigma_2, defaults.b_sigma_2, sigma_2, theta,    T1, static_cast<int>(p), B, xi_2);

        // 3. Mise à jour de B via MCMC
        Matrix S_epsi1 = AddMatrices(
            DivideMatrixByScalar(S_epsi(lambda_1, xi_1), sigma_1),
            DivideMatrixByScalar(S_epsi(theta,    xi_2), sigma_2)
        );

        MCMC_Result_Full ResultMC = MCMC(
            defaults.lambda_0, B, R, detB,
            Psi, T_wishart, b, S_epsi1,
            defaults.neta, A_Theta, T1, true
        );

        A_Theta   = ResultMC.A_Theta;
        detB      = ResultMC.detB;
        R         = ResultMC.Sigma;
        B         = ResultMC.B;
        T_wishart = ResultMC.T;
        Psi       = ResultMC.Psi;
    }

    if (verbose) {
        std::cout << "############################# END BURN-IN STEP #####################################################\n";
    }

    // --- 7. SAMPLING STEP ---
    std::uniform_int_distribution<> distrib_sample(0, static_cast<int>(n_sample) - 1);

    for (std::size_t it_sample = 0; it_sample < n_sample; ++it_sample) {
        // 1. Clustering
        PriorPartitionResult Partition = fullPartition(
            b, defaults.M, SS, CS, IDH,
            mu, defaults.b_mu, alpha, defaults.b_alpha,
            lambda_1, theta, X, defaults.m_neal
        );

        CS    = Partition.CS_r;
        Ncl   = Partition.Ncl;
        mu    = Partition.mu_r;
        alpha = Partition.alpha_r;
        SS    = Partition.SS;

        // 2. Mises à jour MH/Gibbs
        mu    = fullMu(IDH, lambda_1, mu, defaults.b_mu, alpha, SS);
        alpha = fullAlpha(IDH, theta, alpha, defaults.b_alpha, mu, SS);
        if (verbose) {
            std::cout << "########### alpha update at sample " << it_sample << " ################\n";
        }

        lambda_1 = fullLambda(mu, defaults.b_mu, xi_1, B, lambda_1, sigma_1, SS, T1);
        theta    = fullTheta(alpha, defaults.b_alpha, xi_2, B, theta, sigma_2, SS, T1);

        xi_1 = fullXi(xi_1, defaults.a_xi, b_xi, B, lambda_1, sigma_1);
        xi_2 = fullXi(xi_2, defaults.a_xi, b_xi, B, theta,    sigma_2);

       sigma_1 = fullSigma(defaults.a_sigma_1, defaults.b_sigma_1, sigma_1, lambda_1, T1, static_cast<int>(p), B, xi_1);
       sigma_2 = fullSigma(defaults.a_sigma_2, defaults.b_sigma_2, sigma_2, theta,    T1, static_cast<int>(p), B, xi_2);

        Matrix S_epsi1 = AddMatrices(
            DivideMatrixByScalar(S_epsi(lambda_1, xi_1), sigma_1),
            DivideMatrixByScalar(S_epsi(theta,    xi_2), sigma_2)
        );

        MCMC_Result_Full ResultMC = MCMC(
            defaults.lambda_0, B, R, detB,
            Psi, T_wishart, b, S_epsi1,
            defaults.neta, A_Theta, T1, true
        );

        A_Theta   = ResultMC.A_Theta;
        detB      = ResultMC.detB;
        R         = ResultMC.Sigma;
        B         = ResultMC.B;
        T_wishart = ResultMC.T;
        Psi       = ResultMC.Psi;

        // --- 7.a Prédiction pour la nouvelle période (T1+1) : mu_save ---
        Ncl = CS.size();

        Vector tp;             // log-proba / proba par cluster
        Vector norm_prob;      // buffer de normalisation

        for (std::size_t k_0 = 0; k_0 < p; ++k_0) {
            tp.assign(Ncl, 0.0);

            // init ss_temp avec SS courant
            for (std::size_t k = 0; k < pt; ++k) {
                ss_temp[k] = SS[k];
            }
            // désaffecter les positions T1+1.. pour cette région
            for (std::size_t k_00 = pt + k_0; k_00 < pt + p; ++k_00) {
                ss_temp[k_00] = -1.0;
            }

            // log-proba pour chaque cluster j
            for (std::size_t j = 0; j < Ncl; ++j) {
                tp[j] = std::log(CS[j]) + std::log(defaults.M)
                        - (numerateur(CS, j, b_j, l) - denominateur(ss_temp, X, j, CS, l, b_j));

                ss_temp[pt + k_0] = static_cast<double>(j);
                CS[j] += 1.0;

                tp[j] += numerateur(CS, j, b_j, l)
                         - denominateur(ss_temp, X, j, CS, l, b_j);

                CS[j] -= 1.0;
                ss_temp[pt + k_0] = -1.0;
            }

            bool bad_state = false;
            for (double v : tp) {
                if (std::isnan(v) || v == -std::numeric_limits<double>::infinity()) {
                    bad_state = true;
                    break;
                }
            }

            if (bad_state) {
                if (verbose) {
                    std::cout << "probability has NaN for region: " << k_0 << "\n";
                }
                if (it_sample != 0) {
                    double sum_mu = 0.0;
                    for (std::size_t prev = 0; prev < it_sample; ++prev) {
                        sum_mu += mu_save[k_0][prev];
                    }
                    mu_save[k_0][it_sample] = sum_mu / static_cast<double>(it_sample);
                }
            } else {
                if (tp.size() != 1) {
                    double max_log = tp[0];
                    for (double v : tp) {
                        if (v > max_log) max_log = v;
                    }

                    norm_prob.assign(Ncl, 0.0);
                    double sum_exp = 0.0;
                    for (std::size_t j = 0; j < Ncl; ++j) {
                        norm_prob[j] = std::exp(tp[j] - max_log);
                        sum_exp += norm_prob[j];
                    }

                    if (sum_exp < 1e-12) {
                        if (verbose) {
                            std::cerr << "Toutes les probas ~0 pour region " << k_0
                                      << ", on passe à uniforme.\n";
                        }
                        const double u = 1.0 / static_cast<double>(Ncl);
                        for (std::size_t j = 0; j < Ncl; ++j) {
                            norm_prob[j] = u;
                            tp[j]        = u;
                        }
                    } else {
                        const double inv_sum = 1.0 / sum_exp;
                        for (std::size_t j = 0; j < Ncl; ++j) {
                            norm_prob[j] *= inv_sum;
                            tp[j] = norm_prob[j];
                        }
                    }
                } else {
                    tp[0] = 1.0;
                }

                std::discrete_distribution<> dist(tp.begin(), tp.end());
                int j_0 = dist(generator_prior);

                if (j_0 >= 0 && static_cast<std::size_t>(j_0) < mu.size()) {
                    mu_save[k_0][it_sample] = mu[j_0];
                }
            }
        } // fin boucle k_0 (mu_save)

        // --- 7.b Sauvegarde SS, mu, alpha pour cet échantillon ---
        for (std::size_t j = 0; j < SS.size(); ++j) {
            SS_matrix[it_sample][j] = SS[j];
        }
        for (std::size_t j = 0; j < Ncl; ++j) {
            mu_matrix[it_sample][j]    = mu[j];
            alpha_matrix[it_sample][j] = alpha[j];
        }
    } // fin boucle it_sample

    // --- 8. CRPS ---
    for (std::size_t k_0 = 0; k_0 < p; ++k_0) {
        for (std::size_t it_sample = 0; it_sample < n_sample; ++it_sample) {
            // Tirage d’une configuration de clustering
            std::size_t n_0 = static_cast<std::size_t>(distrib_sample(generator_prior));

            // Recharger SS et ss_temp
            for (std::size_t k = 0; k < pt; ++k) {
                SS[k]      = SS_matrix[n_0][k];
                ss_temp[k] = SS[k];
            }
            for (std::size_t k_00 = pt + k_0; k_00 < pt + p; ++k_00) {
                ss_temp[k_00] = -1.0;
            }

            Vector CS_pred = calculateCs(SS);
            std::size_t Ncl_pred = CS_pred.size();

            Vector tp(Ncl_pred, 0.0);
            Vector norm_prob(Ncl_pred, 0.0);

            for (std::size_t j = 0; j < Ncl_pred; ++j) {
                tp[j] = std::log(CS_pred[j])
                        - (numerateur(CS_pred, j, b_j, l) - denominateur(ss_temp, X, j, CS_pred, l, b_j));

                ss_temp[pt + k_0] = static_cast<double>(j);
                CS_pred[j] += 1.0;

                tp[j] += numerateur(CS_pred, j, b_j, l)
                         - denominateur(ss_temp, X, j, CS_pred, l, b_j);

                CS_pred[j] -= 1.0;
                ss_temp[pt + k_0] = -1.0;
            }

            bool bad_state = false;
            for (double v : tp) {
                if (std::isnan(v) || v == -std::numeric_limits<double>::infinity()) {
                    bad_state = true;
                    break;
                }
            }

            if (bad_state) {
                if (verbose) {
                    std::cout << "CRPS: NaN prob pour region " << k_0 << "\n";
                }
                continue;
            }

            if (tp.size() != 1) {
                double max_log = tp[0];
                for (double v : tp) {
                    if (v > max_log) max_log = v;
                }

                double sum_exp = 0.0;
                for (std::size_t j = 0; j < Ncl_pred; ++j) {
                    norm_prob[j] = std::exp(tp[j] - max_log);
                    sum_exp += norm_prob[j];
                }

                if (sum_exp < 1e-12) {
                    const double u = 1.0 / static_cast<double>(Ncl_pred);
                    for (std::size_t j = 0; j < Ncl_pred; ++j) {
                        norm_prob[j] = u;
                        tp[j]        = u;
                    }
                } else {
                    const double inv_sum = 1.0 / sum_exp;
                    for (std::size_t j = 0; j < Ncl_pred; ++j) {
                        norm_prob[j] *= inv_sum;
                        tp[j] = norm_prob[j];
                    }
                }
            } else {
                tp[0] = 1.0;
            }

            std::discrete_distribution<> dist(tp.begin(), tp.end());
            int j_0 = dist(generator_prior);

            if (j_0 >= 0 && static_cast<std::size_t>(j_0) < mu_matrix[n_0].size()) {
                double mu_k0    = mu_matrix[n_0][j_0];
                double alpha_k0 = alpha_matrix[n_0][j_0];

                double beta_param = alpha_k0 * (-1.0 + 1.0 / mu_k0);
                if (beta_param < 1e-12) {
                    beta_param = 1e-5;
                }

                crps_sample[it_sample][k_0] = sampleBeta(alpha_k0, beta_param);
            }
        }
    }

    // --- 9. Résultat final ---
    BetaGaussienResult final_result;
    final_result.mu_save       = std::move(mu_save);
    final_result.crps_sample   = std::move(crps_sample);
    final_result.n_sample_used = static_cast<int>(n_sample);
    final_result.IDHPre        = IDHPre;

    if (verbose) {
        std::cout << "################ THANKS FOR THE COLLABORATION #####################\n";
    }

    return final_result;
}