#include <filesystem>
#include <iostream>
#include "prior.h"
#include "modeleBetaGaussian.h"
#include "simulation.h"
#include "fonctions.h"
#include "sampling.h" // function for sampling
// ===== Helpers simples =====

// Matrice identité n x n multipliée par "scale"
Matrix eye(int n, double scale ) {
    Matrix m(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) m[i][i] = scale;
    return m;
}

// Matrice de zéros
Matrix zeros(int rows, int cols) {
    return Matrix(rows, Vector(cols, 0.0));
}
// Vecteur de zéros
Vector zeros(int n) {
    return Vector(n, 0.0);
}

// Graphe Erdos–Rényi G(n, p) -> matrice d’adjacence 0/1
Matrix erdosRenyiAdjacency(int n, double prob, std::mt19937_64 &gen) {
    std::bernoulli_distribution bern(prob);
    Matrix A(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            bool edge = bern(gen);
            if (edge) {
                A[i][j] = 1.0;
                A[j][i] = 1.0;
            }
        }
    }
    return A;
}

void createDirIfNotExists(const std::string &path) {
    struct stat info;

    // Vérifie si le dossier existe déjà
    if (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR)) {
        // Le dossier existe déjà → ne rien faire
        return;
    }

    // Crée le dossier
    if (mkdir(path.c_str(), 0777) != 0) {
        std::cerr << "Erreur : impossible de créer le dossier "
                  << path << " (" << strerror(errno) << ")\n";
    }
}


void simulationBetaGaussian(int T1, int p, int l,
                             const std::string &chemin_dossier_in,
                             const std::string &chemin_dossier_out)
{
    // "p est le nombre de regions et T1 le nombre de périodes"
    // l = nombre de covariables

    // RNG
    std::random_device rd;
    std::mt19937_64 gen(rd());
    T1+=1;  // une donnee de prevision
    p+=1;
    // D = 2 * I_p (pas utilisé ensuite, mais je le garde pour fidélité)
    Matrix D = eye(p, 2.0);

    // T = 1/sqrt(2) * I_p
    Matrix T = eye(p, 1.0 / std::sqrt(2.0));

    double prob = 0.3;  // probabilité de tirer une arête
    double b = 4.0;     // paramètre de forme de la loi initiale

    // A_Theta : matrice d’adjacence Erdos–Rényi
    Matrix A_Theta = erdosRenyiAdjacency(p, prob, gen);

    // PRIOR DE LA MATRICE DE PRÉCISION
    Matrix Psi = EchantillonPsi(A_Theta, T, b);

    MatricesPrecision MP = precisionMatrix(Psi);

    Matrix B = MP.Omega;                        
    double detB = 0;                          
    Matrix R = MP.Sigma; 


    // X : matrice p x l
    Matrix X = zeros(p, l);

    // En Python : X[:, l00] = N(0, R)  (multivariée)
    // Ici : TODO → vraie gaussienne multivariée, pour l’instant on met un placeholder :
    std::normal_distribution<double> norm01(0.0, 1.0);

    for (int l00 = 0; l00 < l; ++l00) {
        // remplacer ce bloc par :
        // Vector x = multivariate_normal_rvs_zero(R, gen);
        // for (int i = 0; i < p; ++i) X[i][l00] = x[i];
        for (int i = 0; i < p; ++i) {
            // approximation : indépendants avec var = R[i][i]
            double z = norm01(gen) * std::sqrt(R[i][i]);
            X[i][l00] = z;
        }
    }

    // prior for the partition: (SS, CS, Ncl, ma) = f.kmeansSS(X, T1)
    std::cout <<" -----------------ICI ---------------------"<<T1<<std::endl;
    KmeansResult km = kmeansSS(X, T1,10);
    Vector SS = km.SS;
    int Ncl = km.Ncl;
    Matrix ma = km.ma;  // ma non utilisé ici, mais je le récupère quand même
    
    b = 10.0;        // nouveau paramètre de forme
    int l0 = l;      // nombre de covariables
    double b_j = 2.0 * l;  // shape pour la partition
    // rng = np.random.default_rng()  → on a déjà gen

    // Initialisation des paramètres de rotation
    double a_sigma_1 = 0.5;
    double b_sigma_1 = 10.0;
    double a_sigma_2 = 0.5;
    double b_sigma_2 = 10.0;
    double a_xi = -1.0;
    double b_xi = -a_xi;
    double b_mu = 1.0;
    double b_alpha = 0.5;
    double neta = 1.0;
    int m_neal = 3;
    // h_neal = int(Ncl) + m_neal  (non utilisé)

    // Nouveau A_Theta et prior de la matrice de précision
    A_Theta = erdosRenyiAdjacency(p, prob, gen);
    Psi = EchantillonPsi(A_Theta, T, b);
    MP = precisionMatrix(Psi);
    B = MP.Omega;
   // detB = MP.detB;
    detB=MP.detb;
    R = MP.Sigma;

    // B[i,j] *= sqrt(R[i,i] * R[j,j])
    /*for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            B[i][j] *= std::sqrt(R[i][i] * R[j][j]);
        }
    }

    // detB += sum(log(diag(R)))
    for (int i = 0; i < p; ++i) {
        detB += std::log(R[i][i]);
    } */

    // PRIORS
   // Vector sigma_1 = prior_sigma(a_sigma_1, b_sigma_1);
    //Vector sigma_2 = prior_sigma(a_sigma_2, b_sigma_2);

   // Vector a_xi_vec(p, a_xi);
   // Vector b_xi_vec(p, b_xi);
   
    Vector xi_1 = priorXI(a_xi, b_xi,p);
    Vector xi_2 = priorXI(a_xi, b_xi,p);
    double sigma_1=1; double sigma_2=2;
    Matrix lambda_1 = priorLogNormal(B, sigma_1, xi_1, T1);
    Matrix theta = priorLogNormal(B, sigma_2, xi_2, T1);



    // alpha, mu
    Vector alpha = priorGamma(theta, b_alpha, SS);
    Vector mu    = priorBeta(lambda_1, b_mu, SS);

    // IDH_S: p x T1
    Matrix IDH_S = zeros(p, T1);

    // Création des bases de données
    for (int i = 0; i < p; ++i) {
        for (int t = 0; t < T1; ++t) {
            int idx = i + t * p;
            int ss  = SS[idx];          // SS est supposé 0..Ncl-1 ou 1..Ncl
            int ssi = std::max(0, ss);  // à adapter selon l’indexation de SS

            // en Python: alpha[int(SS[i+t*p])]
            double a_par = alpha[ssi];
            double mu_par = mu[ssi];
            double b_par = a_par * (1.0 - mu_par) / mu_par;

            IDH_S[i][t] = sampleBeta(a_par, b_par);
        }
    }
  
    // Écriture de index.csv
    createDirIfNotExists("chemin_dossier_in");
     
    //std::filesystem::create_directories(chemin_dossier_in);
    std::string chemin_index = chemin_dossier_in+"/index.csv";
    std::ofstream out_index(chemin_index);
    p-=1;
    for (int i = 0; i < p; ++i) {
        for (int t = 0; t < T1; ++t) {
            out_index << IDH_S[i][t];
            if (t + 1 < T1) out_index << ',';
        }
        out_index << '\n';
    }
    out_index.close();
    std::string data2 = chemin_index;

    // Écriture de covariables.csv
    std::string chemin_cov = chemin_dossier_in + "/covariables.csv";
    std::ofstream out_cov(chemin_cov);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < l; ++j) {
            out_cov << X[i][j];
            if (j + 1 < l) out_cov << ',';
        }
        out_cov << '\n';
    }
    out_cov.close();
    std::string data1 = chemin_cov;

    // Lancement du modèle
    auto debut = std::chrono::high_resolution_clock::now();
    bool State=false;
   
     HyperParameters defaults;
    defaults.b_mu=0.25;
    defaults.a_sigma_1=5;
    defaults.a_sigma_2=5;
    defaults.b_sigma_1=10;
    defaults.b_sigma_2=10;
    defaults.lambda_0=0.05;
    size_t burn_in=1000;
    size_t n_sample =1000;
    BetaGaussienResult result;
     //std::cout <<" -----------------ICI LOIN on est capable ---------------------"<<T1<<std::endl;
    result = betaGaussien(data1, data2,burn_in,n_sample,defaults,State);

    createDirIfNotExists(chemin_dossier_out);  // pour la souvegarde
    Vector means =rowMeans(result.mu_save);
    Vector crps=crpsVector(result.crps_sample,result.IDHPre);
   // exportation des resultats
    print_vector(means);
    print_vector(result.IDHPre);
    Vector std=RowsStd(result.mu_save);
    print_vector(VectorDifference(means,result.IDHPre));
    print_vector(std);
    print_vector(crps);
    saveMatrixCSV(result.crps_sample,chemin_dossier_out+"/crpsSample.csv");   // saving of the crps Sample
    saveMatrixCSV(result.mu_save,chemin_dossier_out+"/muSave.csv");    // saving of the proposal mu sample
    saveVectorCSV(means,chemin_dossier_out+"/predict.csv");     // saving of the means prediction
    saveVectorCSV(VectorDifference(means,result.IDHPre),chemin_dossier_out+"/diffForeObs.csv");   // saving of the difference
    saveVectorCSV(crps,chemin_dossier_out+"/crps.csv");    // crps proposal values
    saveVectorCSV(std,chemin_dossier_out+"/standardDeviation.csv");  // standard deviation   
    // Sauvegarde des résultats
   // resultat(res.pred_sample, res.sample_crps, res.n_sample,
       //      res.IDHPred, chemin_dossier_out);

    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = fin - debut;
    double secondes_tot = diff.count();

    // Soit tu utilises ta fonction F.secondes_vers_hms,
    // soit tu fais la conversion directement ici :
    long total = static_cast<long>(std::round(secondes_tot));
    long heures   = total / 3600;
    long minutes  = (total % 3600) / 60;
    long secondes = total % 60;

    std::cout << "####################################################################### TEMPS EXECUTION "
              << heures << "h " << minutes << "m " << secondes << "s\n";
}