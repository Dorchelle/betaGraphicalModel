// fonctions.cpp


#include <stdexcept> // Pour les erreurs
#include <algorithm> // pour std::copy
#include <limits>    // <-- NOUVEAU : Nécessaire pour std::numeric_limits
#include <random>
#include <iomanip>  // Pour formater l'affichage des doubles
#include <iostream>
#include <cmath>     // <-- NOUVEAU : Nécessaire pour std::log et std::lgamma
#include <map>
#include <string>
#include <plplot/plstream.h>// <-- for curbe
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "matricesCalculations.h"
#include "fonctions.h"
# include "GraphicalFunctions.h"

// Déclaration d'un générateur aléatoire (supposé défini ailleurs)
extern std::mt19937 generator_prior;
extern Matrix cholesky(const Matrix& A); 
extern MatricesPrecision precisionMatrix(const Matrix& Psi);
extern Matrix EchantillonPsi(const Matrix& A, const Matrix& T, double b);
extern Matrix EchantillonPsiMod(Matrix& Psi, Matrix& A, const Matrix& T, double b, size_t i1, size_t j1, bool Etat, double tp);
extern PropAjoutRetrait ajoutUnNoeud(const Matrix& A_Omega);
extern PropAjoutRetrait extractNoeud(const Matrix& A_Omega);
extern double trace(const Matrix& M);
extern Matrix multiplyMatrices(const Matrix& A, const Matrix& B);
// Déclaration statique du générateur de nombres aléatoires
// C'est une bonne pratique de n'initialiser le moteur qu'une seule fois.
static std::random_device rd;

static std::mt19937 generator(rd()); // Moteur Mersenne Twister


// --- Fonctions Algèbre Linéaire et Utilitaires ---


double logUniformRvs() {
    std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
    return std::log(uniform_01(generator_prior));
}
double logUniformRvs();


Vector calculateCs(const Vector& SS) {
    if (SS.empty()) return Vector();
    
    // Pour trouver le nombre de clusters (Ncl), nous utilisons une map temporaire 
    // ou trouvons le label maximum. L'approche la plus sûre est de trouver max_label.
    double max_label = -1.0;
    for (double label : SS) {
        if (label > max_label) {
            max_label = label;
        }
    }
    
    // Le nombre de clusters est max_label + 1 (si les labels sont contigus de 0)
    int Ncl = static_cast<int>(std::round(max_label)) + 1;
    
    // Si la matrice SS n'est pas complètement remplie (labels manquants), Ncl peut être trop grand.
    // Utiliser une map pour obtenir la taille exacte et les comptages, puis la convertir.
    
    std::map<int, double> counts_map;

    for (double label : SS) {
        int j = static_cast<int>(std::round(label));
        if (j >= 0) { 
            counts_map[j] += 1.0;
        }
    }
    
    // Convertir la map en un vecteur CS contigu de taille Ncl.
    // S'il manque des labels (ex: on a 0 et 2, mais pas 1), Ncl doit être ajusté.
    // Cependant, dans les algorithmes DPMM, les labels sont souvent compactés, 
    // donc nous utilisons la taille du plus grand label.
    
    Vector Cs(Ncl, 0.0);
    for (const auto& pair : counts_map) {
        size_t j = pair.first;
        if (j < Ncl) {
            Cs[j] = pair.second;
        }
    }

    return Cs;
}
// fonction kmeans//
/**
 * Calcule la distance euclidienne carrée entre deux vecteurs.
 * (Nécessaire pour le K-Means complet)
 */
double euclideanDistanceSq(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Erreur: Les vecteurs doivent avoir la même taille.");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::pow(a[i] - b[i], 2.0);
    }
    return sum;
}






KmeansResult kmeansSS(const Matrix& X, int T1, int K_init) {
    if (T1 <= 0) {
        throw std::invalid_argument("kmeansSS: T1 doit être > 0");
    }
    if (X.empty() || X[0].empty()) {
        throw std::invalid_argument("kmeansSS: X doit être non vide");
    }

    size_t p = X.size();        // nb de régions
    size_t l = X[0].size();     // nb de features par région
    size_t num_points = p * (size_t)T1;
    KmeansResult result;

    // Transposition correcte : (p*T1 x l)
    Matrix X_data(num_points, Vector(l, 0.0));
    for (size_t t = 0; t < (size_t)T1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            size_t idx = i + t * p;
            for (size_t f = 0; f < l; ++f) {
                X_data[idx][f] = X[i][f];
            }
        }
    }

    if (K_init <= 0 || (size_t)K_init > num_points) {
        K_init = static_cast<int>(std::sqrt((double)num_points)) + 1;
    }
    int K = K_init;

    std::uniform_int_distribution<> initial_dist(0, num_points - 1);

    // Initialiser les centroïdes
    std::vector<Vector> centroids(K, Vector(l, 0.0));
    centroids[0] = X_data[initial_dist(generator_prior)];

    std::vector<double> min_dist_sq(num_points, 0.0);

    for (int k = 1; k < K; ++k) {
        double total_dist_sq = 0.0;
        for (size_t i = 0; i < num_points; ++i) {
            double best = std::numeric_limits<double>::max();
            for (int c = 0; c < k; ++c) {
                best = std::min(best, euclideanDistanceSq(X_data[i], centroids[c]));
            }
            min_dist_sq[i] = best;
            total_dist_sq += best;
        }
        std::uniform_real_distribution<double> uniform_01(0.0, total_dist_sq);
        double r = uniform_01(generator_prior);
        double cum = 0.0;
        for (size_t i = 0; i < num_points; ++i) {
            cum += min_dist_sq[i];
            if (cum >= r) {
                centroids[k] = X_data[i];
                break;
            }
        }
    }

    Vector SS_result(num_points, 0.0);
    Vector ss_tp(num_points, -1.0);

    std::vector<int> counts(K, 0);

    const int max_iter = 100;
    for (int iter = 0; iter < max_iter; ++iter) {
        bool changed = false;

        // Affectation
        for (size_t i = 0; i < num_points; ++i) {
            double best_dist = std::numeric_limits<double>::max();
            int best_clust = 0;

            for (int c = 0; c < K; ++c) {
                double d = euclideanDistanceSq(X_data[i], centroids[c]);
                if (d < best_dist) {
                    best_dist = d;
                    best_clust = c;
                }
            }
            if (SS_result[i] != best_clust) {
                SS_result[i] = (double)best_clust;
                changed = true;
            }
        }

        // Mise à jour des centres
        std::vector<Vector> new_centroids(K, Vector(l, 0.0));
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < num_points; ++i) {
            int c = (int)SS_result[i];
            counts[c]++;
            for (size_t f = 0; f < l; ++f) {
                new_centroids[c][f] += X_data[i][f];
            }
        }

        for (int c = 0; c < K; ++c) {
            if (counts[c] > 0) {
                for (size_t f = 0; f < l; ++f) {
                    centroids[c][f] = new_centroids[c][f] / counts[c];
                }
            }
        }
        if (!changed) break;
    }

    // Construire ss_tp = étiquettes répétées par période
    for (size_t t = 0; t < (size_t)T1; ++t) {
        for (size_t i = 0; i < p; ++i) {
            ss_tp[i + t*p] = SS_result[i + t*p];
        }
    }

    result.SS   = std::move(ss_tp);
    result.CS   = Vector(K, 0.0);
    for (int c = 0; c < K; ++c) {
        result.CS[c] = static_cast<double>(counts[c] * T1);
    }
    result.Ncl = K;

    result.ma.assign(p, Vector(p, 0.0));
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            if (SS_result[i] == SS_result[j]) result.ma[i][j] = 1.0;
        }
    }

    return result;
}
  



double logBetaPDF(double x, double alpha, double beta) {
    // Vérification des conditions
    if (x <= 0.0 || x >= 1.0 || alpha <= 0.0 || beta <= 0.0) {
        // Correction du nom de la fonction std::log pour éviter l'ambiguïté avec <cmath>
        return -std::numeric_limits<double>::infinity(); // Retourne un log(0)
    }
    
    // Le C++ standard utilise std::lgamma pour log(Gamma(x))
    double log_beta_func = std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);

    return (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log(1.0 - x) - log_beta_func;
}


// matrice transposer
Matrix XXt(const Vector& X) {
    size_t l = X.size();
    Matrix M(l, Vector(l, 0.0));

    for (size_t i = 0; i < l; ++i) {
        for (size_t j = i; j < l; ++j) {
            double val = X[i] * X[j];
            M[i][j] = val;
            M[j][i] = val; // symétrie
        }
    }
    return M;
}


/**
 * Remplace trace(M)
 */
double trace(const Matrix& M) {
    if (M.empty() || M.size() != M[0].size()) {
        if (M.empty()) return 0.0;
        throw std::invalid_argument("La matrice doit être carrée pour calculer la trace.");
    }
    double tp = 0.0;
    for (size_t i = 0; i < M.size(); ++i) {
        tp += M[i][i];
    }
    return tp;
}


/**
 * Implémentation de l'sampleBeta pour une matrice triangulaire supérieure (comme en Python)
 * L'implémentation complète pour une matrice quelconque est complexe et nécessiterait Eigen.
 */


Matrix Inverse(const Matrix& Phi) {
    const size_t p = Phi.size();
    if (p == 0) {
        return Matrix();
    }

    // Vérifier que la matrice est carrée
    for (size_t r = 0; r < p; ++r) {
        if (Phi[r].size() != p) {
            throw std::invalid_argument("Inverse: la matrice doit être carrée.");
        }
    }

    Matrix IPhi(p, Vector(p, 0.0));
    constexpr double tol = 1e-9;

    // Pré-calcul des inverses de la diagonale (et vérification)
    Vector inv_diag(p);
    for (size_t i = 0; i < p; ++i) {
        const double dii = Phi[i][i];
        if (std::abs(dii) < tol) {
            throw std::runtime_error("Inverse: diagonale nulle ou trop petite.");
        }
        inv_diag[i] = 1.0 / dii;
    }

    // Inversion d'une matrice triangulaire supérieure
    for (size_t i = 0; i < p; ++i) {
        // Diagonale
        IPhi[i][i] = inv_diag[i];

        // Partie supérieure: j = i-1, ..., 0
        for (size_t j = i; j-- > 0; ) {
            double somme = 0.0;

            const Vector& rowPhi_j = Phi[j];
            const Vector& colInv_i = IPhi[i]; // on va lire IPhi[k][i] via col, mais on reste 2D
            // k = j+1..i
            for (size_t k = j + 1; k <= i; ++k) {
                somme += rowPhi_j[k] * IPhi[k][i];
            }

            // IPhi[j,i] = - somme / Phi[j,j] = - somme * inv_diag[j]
            IPhi[j][i] = -somme * inv_diag[j];
        }
    }

    return IPhi;
}
 



/**
 * Remplace logLikhood(X,ALPHA,BETA)
 */
double logLikhood(const Matrix& X, const Matrix& ALPHA, const Matrix& BETA) {
    size_t p = X.size();
    if (p == 0) return 0.0;
    size_t T = X[0].size();
    if (p != ALPHA.size() || T != ALPHA[0].size() || p != BETA.size() || T != BETA[0].size()) {
        throw std::invalid_argument("Les matrices doivent avoir les mêmes dimensions.");
    }

    double tp = 0.0;
    for (size_t i = 0; i < p; ++i) {
        for (size_t t = 0; t < T; ++t) {
            tp += logBetaPDF(X[i][t], ALPHA[i][t], BETA[i][t]);
        }
    }
    return tp;
}


/**
 * Remplace secondesVersHMS
 */
std::tuple<int, int, int> secondesVersHMS(long long secondes) {
    int heures = (int)(secondes / 3600);
    int minutes =(int)( (secondes % 3600) / 60);
    int sec = (int)secondes % 60; // Renommé 'sec' pour éviter le conflit avec le paramètre
    return std::make_tuple(heures, minutes, sec);
}
// Helper function pour lire un fichier CSV dans un format de matrice
// Cette fonction gère la lecture et la conversion des chaînes en nombres.
Matrix lire_csv_vers_matrix(const std::string& chemin_fichier, bool header, char separator) {
    std::ifstream file(chemin_fichier);
    std::string line;
    Matrix data;

    if (!file.is_open()) {
        throw std::runtime_error("Erreur: Impossible d'ouvrir le fichier " + chemin_fichier);
    }

    // Sauter l'en-tête (si header=true, le Python saute la première ligne (header=1)
    if (header) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Erreur: Le fichier est vide.");
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        Vector row;

        // Lire les cellules séparées par le séparateur
        while (std::getline(ss, cell, separator)) {
            try {
                // Tente de convertir la chaîne en double.
                // Si la première colonne est un nom de ville (Python: Ville = table[:,0]),
                // cette conversion échouera, et nous devrons sauter cette colonne.
                
                // Pour simplifier et reproduire la logique Python (qui prend le reste des colonnes après Ville),
                // on ne lit ici que des données numériques.

                // Si la ligne n'est pas vide (pour éviter les problèmes avec les séparateurs de fin de ligne)
                if (!cell.empty()) {
                    row.push_back(std::stod(cell));
                }
            } catch (const std::invalid_argument& e) {
                // Si ce n'est pas un nombre (probablement la colonne 'Ville'), on l'ignore ou le gère.
                // Pour l'instant, nous supposons que la première colonne non numérique est ignorée
                // dans la construction finale de la matrice de covariables X.
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    return data;
}


Matrix readCSV(const std::string &filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << "\n";
        return {};
    }

    Matrix data;
    std::string line;

    while (std::getline(file, line)) {
        // Ignore les lignes vides
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        std::stringstream ss(line);
        std::string cell;
        Vector row;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                std::cerr << "Erreur de conversion dans le CSV : '" << cell << "'\n";
                row.push_back(0.0); // fallback
            }
        }

        data.push_back(row);
    }

    return data;
}


// ... (Inclure la fonction lire_csv_vers_matrix ici si elle n'est pas dans le .h)
DonneesChargees chargesDonnees(const std::string& chemin_fichier_covariable,
                                const std::string& chemin_fichier_VaRep,
                                bool State) {
    DonneesChargees result;

    if (State) {
        // --- LOGIQUE STATE=TRUE (header=1, sep=';') ---
        
        // table=pd.read_csv(chemin_fichier_covariable, header=1, sep=";")
        // Le Python utilise header=1, ce qui signifie que les données commencent à la deuxième ligne.
        // Notre helper function saute la première ligne d'en-tête.
        Matrix table_cov = lire_csv_vers_matrix(chemin_fichier_covariable, true, ';');

        if (table_cov.empty()) {
            throw std::runtime_error("Erreur: Table des covariables vide.");
        }

        size_t p = table_cov.size(); // Nombre d'individus (lignes)
        size_t cols = table_cov[0].size();
        //result.Ville.assign(StringVector(p));
        
        // l = np.size(table[0,:])-1
        // Si la première colonne est la ville (nom), on assume qu'elle a été sautée
        // dans notre `lire_csv_vers_matrix` ci-dessus pour simplifier.
        // Si elle n'a pas été sautée, il faudrait cols - 1.
        // Ici, on assume que la première colonne a été traitée ou ignorée dans la lecture.
        // Basé sur: table1=table[:,1:(l+1)] -> la première colonne (indice 0) est ignorée.
        size_t l = cols; // l est le nombre de covariables (colonnes numériques restantes)

        // X: Matrice l x p (Transposée de table1 en Python)
        //result.X.assign(l, Vector(p));
        result.X.assign(p, Vector(l));

        // table1 = table[:, 1:(l+1)] (La lecture python ignore la colonne 0 (Ville))
        // Notre `lire_csv_vers_matrix` ci-dessus *doit* être ajustée pour gérer le saut de colonne.
        // Pour l'instant, nous faisons la transposition : X = table1.T
        for (size_t i = 0; i < p; ++i) { // lignes (individus)
            for (size_t j = 0; j < l; ++j) { // colonnes (covariables)
                // Le Python fait la transposition de (p x l) vers (l x p)
                result.X[i][j] = table_cov[i][j];
            }
        }

        // --- Mise à jour manuelle des lignes de Longitude (l-2) et Latitude (l-1) ---
        // Les indices Python l-2 et l-1 correspondent aux deux dernières lignes de X.
        // En C++, les lignes de X sont indexées de 0 à l-1.
        if (l >= 2 && p == 32) {
             Vector new_longitude = {-102.3726689,-115.1425107,-111.5706164,-90.0,-102.0000001,-104.0,-92.5000001,-106.0000001, -99.1441352, -104.833333 , -101.0,-100.0,-99.0,-103.6666671,-99.1331785, -101.878113 ,-99.0,-105.0000001,-94.9841472,-96.5,-98.0,-99.8837376, -88.5000001,-100.4949145,-107.5000001,-110.6666671,-92.6681659,-98.7026825,-98.166667, -96.666667,-88.8755669,-102.9333954 };
             Vector new_latitude = {21.9942689,30.0338923,25.5818014,19.0,27.3333331,19.166667,16.5000001,28.5000001,  23.7389846 ,24.833333, 20.9876996,17.666667,20.5, 20.3333331,19.4326296, 19.207098, 18.75, 22.0000001,16.2048579, 17.0,18.833333, 20.8052225, 19.6666671,22.5000001,25.0000001,29.3333331,17.9999288,23.9891553,19.416667, 19.333333,20.6845957,23.0916177};
            // X[l-2,:] = ... (Avant-dernière ligne)
            if (result.X[0].size() > (size_t)l - 2) {
                for (size_t i = 0; i < p; ++i) result.X[i][l-2] = new_longitude[i];
            }

            // X[l-1,:] = ... (Dernière ligne)
            if (result.X[0].size() > (size_t)l - 1) {
                for (size_t i = 0; i < p; ++i) result.X[i][l-1] =  new_latitude[i];
                
            }
        }
        
        // --- Chargement et manipulation de IDH ---
        // IDH1=pd.read_csv(chemin_fichier_VaRep, header=1, sep=";")
        Matrix IDH1 = lire_csv_vers_matrix(chemin_fichier_VaRep, true, ';');

        size_t T1_cols = IDH1[0].size();
        
        // IDHPre=IDH1[:,T1-1] (Dernière colonne)
        if (T1_cols > 0) {
            result.IDHPre.resize(p);
            for (size_t i = 0; i < p; ++i) {
                result.IDHPre[i] = IDH1[i][T1_cols - 1];
                //result.Ville[i]=IDH1[i][1]; 
            }
        }
        
        // IDH =IDH1[:,2:(T1-1)] (Colonnes de 2 à T1_cols-2)
        // La nouvelle T1 est T1_cols - 3
        result.T1 = (int)(T1_cols-2);
        
        if (result.T1 > 0) {
            size_t start_col = 1;
            size_t end_col = T1_cols - 2;

            result.IDH.assign(p, Vector((size_t)result.T1));
            for (size_t i = 0; i < p; ++i) {
                for (size_t j = start_col; j <= end_col; ++j) {
                    result.IDH[i][j - start_col] = IDH1[i][j];
                }
            }
        }

    } else {
        // --- LOGIQUE STATE=FALSE (header=False, sep=',') ---
         std::cout << "---Dans le else du chargement" << std::endl;
        // table=pd.read_csv(chemin_fichier_covariable) (header=False, sep=',') ICI ICI 
        Matrix table_cov = lire_csv_vers_matrix(chemin_fichier_covariable, false, ',');
        //Matrix  table_cov =readCSV(chemin_fichier_covariable);
        if (table_cov.empty()) {
            throw std::runtime_error("Erreur: Table des covariables vide.");
        }

        size_t p = table_cov.size();
        size_t l = table_cov[0].size();    // number of colons 
        std::cout << "---p vaut" << p<<std::endl;
        std::cout << "---l vaut" << l<<std::endl;
        // X = table.T (Transposition)
        result.X.assign(p, Vector(l));
        for (size_t i = 0; i < p; ++i) {
            for (size_t j = 0; j < (size_t)l; ++j) {
                result.X[i][j] = table_cov[i][j];
            }
        }
        
        // IDH1=pd.read_csv(chemin_fichier_VaRep)
        Matrix IDH1 = lire_csv_vers_matrix(chemin_fichier_VaRep, false, ',');
        // ICI ICI ICI
        //Matrix IDH1=readCSV(chemin_fichier_VaRep);
        
        size_t T1_cols = IDH1[0].size();   // number of periods
        
        // IDHPre=IDH1[:,T1-1] (Dernière colonne)
        if (T1_cols > 0) {
            result.IDHPre.resize(p);
            for (size_t i = 0; i < p; ++i) {
                result.IDHPre[i] = IDH1[i][T1_cols - 1];
            }
        }
        
        // IDH =IDH1[:,0:(T1-1)] (Toutes les colonnes sauf la dernière)
        // La nouvelle T1 est T1_cols - 1
        //result.T1 =(int)( T1_cols - 1);
        result.T1 = (int)(T1_cols-1);
        std::cout << "---T1  vaut" << result.T1<<std::endl;
        if (result.T1 > 0) {
            result.IDH.assign(p, Vector((size_t)result.T1));
            for (size_t i = 0; i < p; ++i) {
                for (size_t j = 0; j < (size_t)result.T1; ++j) {
                    result.IDH[i][j] = IDH1[i][j];
                }
            }
        }
    }

    result.l = (int) result.X[0].size(); // Mise à jour finale du l (nombre de covariables)
    return result;
}


// fonction pour charger les donnes de la pollution
DonneesPollution chargesDonneesPollution(
    const std::string& chemin_fichier_pollution, 
    bool has_header, 
    char separator) {
    
    DonneesPollution result;

    // Utilisation de la fonction utilitaire pour lire le CSV.
    // L'hypothèse est que les données sont structurées en lignes (P) x colonnes (T).
    try {
        Matrix table = lire_csv_vers_matrix(chemin_fichier_pollution, has_header, separator);

        if (table.empty()) {
            result.p = 0;
            result.T = 0;
            return result;
        }

        result.PollutionData = table; 
        
        // p = Nombre de lignes (régions/stations)
        result.p = result.PollutionData.size();
        
        // T = Nombre de colonnes (périodes de temps)
        if (result.p > 0) {
            result.T = result.PollutionData[0].size();
        } else {
            result.T = 0;
        }

    } catch (const std::exception& e) {
        // En cas d'erreur de lecture de fichier
        std::cerr << "Erreur lors du chargement des donnees de pollution: " << e.what() << std::endl;
        throw; // Renvoyer l'erreur
    }

    return result;
}


Matrix covJ(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l) {
    // tp = 0*np.eye(l)  -> Matrice l x l initialisée à zéro
    Matrix tp(l, Vector(l, 0.0));
    
    // p=np.size(X[0,:]) -> Nombre d'individus (colonnes dans X, si X est l x p)
    if (X.empty() || l == 0) return tp; // Gestion d'un cas vide
    
    // Si X est l x p, alors la taille de X[0] est p.
    size_t p = X[0].size(); 
    
    // T1=int(np.size(S)/p) -> Nombre de périodes de temps
    size_t T1 = S.size() / p;
    
    // np.size(S) -> taille totale du vecteur S
    size_t taille_S = S.size(); 

    // Boucle sur toutes les entrées du vecteur de regroupement S
    for (size_t i = 0; i < taille_S; ++i) {
        // if(S[i]==j):
        // Note: S[i] est un double, j est un int. Comparaison double/int.
        if (static_cast<int>(S[i]) == j) {
            
            // i_0 = i - int(i/p)*p  -> Calcule l'indice de l'individu (colonne dans X)
            // L'opérateur % (modulo) en C++ est l'équivalent de i - (i/p)*p
            size_t i_0 = i % p;
            
            // On extrait le vecteur X[:, i_0] (la i_0-ième colonne de X)
            Vector X_i(l); 
            for (int k = 0; k < l; ++k) {
                // X[k][i_0] car X est (l x p)
                X_i[k] = X[k][i_0]; 
            }
            
            // tp = tp + XXt(X[:,i_0])
            Matrix XXt_i = XXt(X_i);
            
            // Sommation (tp += XXt_i)
            for (int r = 0; r < l; ++r) {
                for (int c = 0; c < l; ++c) {
                    tp[r][c] += XXt_i[r][c];
                }
            }
        }
    }
    
    // return (T1*tp)
    for (int r = 0; r < l; ++r) {
        for (int c = 0; c < l; ++c) {
            tp[r][c] *= T1; // Multiplie la somme par T1
        }
    }
    
    return tp;
}


Matrix covJPoll(const Vector& S, const Matrix& Data, int j, const Vector& Cs, int l) {
    // tp = 0*np.eye(l)  -> Matrice l x l initialisée à zéro
    Matrix tp(l, Vector(l, 0.0));
    
    // p=Nombre d'entités (colonnes dans Data, si Data est l x p)
    if (Data.empty() || l == 0) return tp; 
    size_t p = Data[0].size(); 
    
    // T1=Nombre de périodes de temps
    size_t T1 = S.size() / p;
    size_t taille_S = S.size(); 

    // Boucle sur toutes les entrées du vecteur de regroupement S
    for (size_t i = 0; i < taille_S; ++i) {
        // if(S[i]==j):
        if (static_cast<int>(S[i]) == j) {
            
            // i_0 = i % p -> Indice de l'entité (colonne) dans Data
            size_t i_0 = i % p;
            
            // On extrait le vecteur Data[:, i_0] (la i_0-ième colonne de Data)
            Vector Data_i(l); 
            for (int k = 0; k < l; ++k) {
                // Data[k][i_0] car Data est (l x p)
                Data_i[k] = Data[k][i_0]; 
            }
            
            // tp = tp + XXt(Data[:,i_0])
            Matrix XXt_i = XXt(Data_i);
            
            // Sommation (tp += XXt_i)
            for (int r = 0; r < l; ++r) {
                for (int c = 0; c < l; ++c) {
                    tp[r][c] += XXt_i[r][c];
                }
            }
        }
    }
    
    // return (T1*tp)
    for (int r = 0; r < l; ++r) {
        for (int c = 0; c < l; ++c) {
            // Note: Si cette fonction est utilisée dans le contexte des lois de probabilité 
            // Wishart (comme dans votre MCMC), la multiplication par T1 est souvent une 
            // simplification ou une étape spécifique de l'algorithme Python.
            tp[r][c] *= T1; 
        }
    }
    
    return tp;
}


Vector mubarj(const Vector& S,
              const Matrix& X,   // X est de taille l x p
              int j,
              const Vector& Cs,
              int l)
{
    // Vecteur résultat de taille l initialisé à zéro
    if (l <= 0) {
        return Vector(); // ou Vector(0) si tu préfères un vecteur vide explicite
    }

    const size_t L = static_cast<size_t>(l);
    Vector tp(L, 0.0);

    if (X.empty()) {
        return tp;
    }

    // p = nombre d'individus (colonnes de X)
    const size_t p = X[0].size();
    const size_t taille_S = S.size();

    // Vérifications de dimensions
    if (X.size() != L) {
        throw std::invalid_argument("mubarj: X.size() != l (nombre de lignes).");
    }
    for (size_t row = 0; row < L; ++row) {
        if (X[row].size() != p) {
            throw std::invalid_argument("mubarj: toutes les lignes de X doivent avoir la même taille p.");
        }
    }
    if (p == 0 || taille_S == 0) {
        return tp;
    }
    if (taille_S % p != 0) {
        throw std::invalid_argument("mubarj: S.size() doit être un multiple de p.");
    }

    if (j < 0 || static_cast<size_t>(j) >= Cs.size()) {
        throw std::invalid_argument("mubarj: indice de cluster j invalide.");
    }

    const double Cs_j = Cs[static_cast<size_t>(j)];
    if (Cs_j <= 0.0) {
        // Cluster vide : on renvoie un vecteur de zéros
        return tp;
    }

    // Accumulation des colonnes correspondant au cluster j
    const int j_int = j;
    for (size_t idx = 0; idx < taille_S; ++idx) {
        if (static_cast<int>(S[idx]) == j_int) {
            // i_0 = idx % p -> indice de la colonne dans X
            const size_t i0 = idx % p;
            for (size_t k = 0; k < L; ++k) {
                tp[k] += X[k][i0];
            }
        }
    }

    // Moyenne : division par la taille du cluster
    const double inv_Cs_j = 1.0 / Cs_j;
    for (size_t k = 0; k < L; ++k) {
        tp[k] *= inv_Cs_j;
    }

    return tp;
}


double multigammaln(double a, int p) {
    if (p < 1) {
        throw std::invalid_argument("multigammaln: La dimension p doit être >= 1.");
    }
    
    // a doit être > (p - 1) / 2
    if (a <= (static_cast<double>(p) - 1.0) / 2.0) {
        std::cerr << "Attention: Argument 'a' trop petit pour la dimension 'p'." << std::endl;
        return -std::numeric_limits<double>::infinity(); 
    }
    
    double result = 0.0;
    result += (static_cast<double>(p) * (p - 1) / 4.0) * std::log(M_PI);
    
    for (int i = 1; i <= p; ++i) {
        double term = (1.0 - static_cast<double>(i)) / 2.0;
        result += std::lgamma(a + term);
    }
    
    return result;
}





double  numerateur(const Vector& Cs, int j,double b_j, int l){
    // Vérification des conditions et des indices
    if (l <= 0 || j >= Cs.size() || Cs[j] < 0.0) {
        throw std::invalid_argument("Erreur numerateur: Paramètres l ou j/Cs invalides.");
    }

    double l_double = static_cast<double>(l);
    double Cs_j = Cs[j];
    
    // --- Terme 1: tp = 0.5*l*(l-1)*np.log(2) ---
    double tp = 0.5 * l_double * (l_double - 1.0) * std::log(2.0);

    // --- Terme 2: tp += multigammaln(0.5*(b_j+l+Cs[j]), l) ---
    
    // Argument 'a' de la multigamma: a = 0.5 * (b_j + l + Cs[j])
    double a_arg = 0.5 * (b_j + l_double + Cs_j);
    
    // Vérifier la condition de validité de la fonction Gamma multivariée
    if (a_arg <= (l_double - 1.0) / 2.0) {
        std::cerr << "Erreur critique numerateur: Argument de multigammaln invalide (a <= (l-1)/2)." << std::endl;
        return -std::numeric_limits<double>::infinity(); 
    }
    
    tp += multigammaln(a_arg, l);

    
    // Votre code Python contenait aussi: tp=0.5*l*(l-1)*np.log(2)
    // Si cette partie est nécessaire à votre modèle, ajoutez-la ici:
    // tp += 0.5 * static_cast<double>(l) * (static_cast<double>(l) - 1.0) * std::log(2.0);

    return tp;
}


double det(const Matrix& M) { 
    // le code fonctionne pour les matrices symetries definies positives.
    const size_t n = M.size();
    if (n == 0) {
        return 0.0;
    }
    if (M[0].size() != n) {
        throw std::invalid_argument("detSPD: la matrice doit être carrée.");
    }

    // Cholesky: M = L L^T, L triangulaire inférieure
    Matrix L = cholesky(M);

    double diag_prod = 1.0;
    for (size_t i = 0; i < n; ++i) {
        const double lii = L[i][i];
        if (lii <= 0.0) {
            // En théorie, pour SPD on ne devrait jamais passer ici
            throw std::runtime_error("detSPD: L(i,i) <= 0, M n'est pas SPD ?");
        }
        diag_prod *= lii;
    }

    // det(M) = (prod diag(L))^2
    return diag_prod * diag_prod;
}



double denominateur(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l, double b_j) {
    
    if (j >= Cs.size() || l <= 0 || b_j <= 0.0) {
        throw std::invalid_argument("Erreur denominateur: Paramètres du cluster ou de dimension invalides.");
    }
    
    double Cs_j = Cs[j];
    double tp = 0.0;
    
    // Terme 1: tp = 0.5*l*(Cs[j]+1)*np.log(np.pi)
    // Nous utilisons M_PI de <cmath>
    tp += 0.5 * static_cast<double>(l) * (Cs_j + 1.0) * std::log(M_PI);

    // Terme 2: tp = tp + np.log(multigammaln(0.5*b_j, l))
    // Argument: a = 0.5 * b_j
    double a_arg_multi = 0.5 * b_j;
    
    // Vérification de validité
    if (a_arg_multi <= (static_cast<double>(l) - 1.0) / 2.0) {
        std::cerr << "Erreur critique: Argument '0.5*b_j' pour multigammaln invalide." << std::endl;
        return -std::numeric_limits<double>::infinity(); 
    }
    tp += multigammaln(a_arg_multi, l);

    // Terme 3: 0.5*(b_j+Cs[j]+l)*np.log(np.linalg.det(Sum))
    
    // a. Calculer Sum = np.eye(l) + covJ(S,X,j,Cs,l)
    Matrix covJValue = covJ(S, X, j, Cs, l); // Calcule la matrice de covariance sommée
    
    // Initialiser Sum = np.eye(l)
    Matrix Sum(l, Vector(l, 0.0));
    for (int i = 0; i < l; ++i) {
        Sum[i][i] = 1.0; 
    }
    
    // Ajouter la matrice de covariance sommée: Sum = Sum + covJ
    if (covJValue.size() != static_cast<size_t>(l) || covJValue[0].size() != static_cast<size_t>(l)) {
        throw std::runtime_error("Erreur denominateur: La matrice covJ n'est pas de taille l x l.");
    }
    for (int r = 0; r < l; ++r) {
        for (int c = 0; c < l; ++c) {
            Sum[r][c] += covJValue[r][c];
        }
    }
    
    // b. Calculer log(det(Sum))
    // ATTENTION: Vous devez avoir une implémentation valide de det(Matrix) ou log_det(Matrix)
    double log_det_sum = std::log(std::abs(det(Sum)));
    
    // c. Ajouter le terme final: 0.5*(b_j+Cs[j]+l) * log(|Sum|)
    tp += 0.5 * (b_j + Cs_j + static_cast<double>(l)) * log_det_sum;

    return tp;
}


double denominateurPoll(const Vector& S, const Matrix& Data, int j, const Vector& Cs, int l, double b_j) {
    
    if (j >= Cs.size() || l <= 0 || b_j <= 0.0) {
        throw std::invalid_argument("Erreur denominateurPoll: Paramètres du cluster ou de dimension invalides.");
    }
    
    double Cs_j = Cs[j];
    double tp = 0.0;
    
    // --- Terme 1: 0.5*l*(Cs[j]+1)*log(pi) ---
    tp += 0.5 * static_cast<double>(l) * (Cs_j + 1.0) * std::log(M_PI);

    // --- Terme 2: log(multigammaln(0.5*b_j, l)) ---
    double a_arg_multi = 0.5 * b_j;
    
    // Vérification de validité
    if (a_arg_multi <= (static_cast<double>(l) - 1.0) / 2.0) {
        std::cerr << "Erreur critique: Argument '0.5*b_j' pour multigammaln invalide." << std::endl;
        return -std::numeric_limits<double>::infinity(); 
    }
    tp += multigammaln(a_arg_multi, l);

    // --- Terme 3: 0.5*(b_j+Cs[j]+l)*log(|I + covJPoll|) ---
    
    // a. Calculer Sum = np.eye(l) + covJPoll(S,Data,j,Cs,l)
    // ATTENTION: Utilisation de covJPoll au lieu de covJ
    Matrix covJPollValue = covJPoll(S, Data, j, Cs, l); 
    
    // Initialiser Sum = np.eye(l)
    Matrix Sum(l, Vector(l, 0.0));
    for (int i = 0; i < l; ++i) {
        Sum[i][i] = 1.0; 
    }
    
    // Ajouter la matrice de covariance sommée
    if (covJPollValue.size() != static_cast<size_t>(l) || covJPollValue[0].size() != static_cast<size_t>(l)) {
        throw std::runtime_error("Erreur denominateurPoll: La matrice covJPoll n'est pas de taille l x l.");
    }
    for (int r = 0; r < l; ++r) {
        for (int c = 0; c < l; ++c) {
            Sum[r][c] += covJPollValue[r][c];
        }
    }
    
    // b. Calculer log(det(Sum))
    double log_det_sum = std::log(std::abs(det(Sum)));
    
    // c. Ajouter le terme final
    tp += 0.5 * (b_j + Cs_j + static_cast<double>(l)) * log_det_sum;

    return tp;
}


Vector truncatedInvgammaSample(double alpha, double beta, double a, double b, size_t size) {
    // Vérification des bornes
    if (a >= b || a <= 0.0 || b <= 0.0 || alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("Erreur truncated_invgamma: Les paramètres de borne/distribution sont invalides.");
    }

    Vector samples;
    samples.reserve(size); // Réserver la mémoire pour une meilleure performance
    
    // Paramètre de la distribution Gamma pour l'échantillonnage
    // Si InvGamma(alpha, beta), nous échantillonnons Gamma(alpha, 1/beta) (taux=1/beta)
    // std::gamma_distribution utilise alpha (forme) et le TAUX (1/échelle)
    double gamma_rate = 1.0 / beta; 
    std::gamma_distribution<double> gamma_dist(alpha, gamma_rate);

    // Boucle d'échantillonnage par acceptation-rejet
    while (samples.size() < size) {
        
        // 1. Échantillonner la distribution Gamma
        double gamma_sample = gamma_dist(generator);
        
        // 2. Transformer en Inverse-Gamma
        // Inverse-Gamma sample = 1.0 / Gamma sample
        double sample;
        if (std::abs(gamma_sample) > 1e-12) { // Évite la division par zéro
            sample = 1.0 / gamma_sample;
        } else {
            // Si la Gamma est zéro, on recommence la boucle
            continue; 
        }

        // --- Affichage de débogage (équivalent au print Python) ---
        // std::cout << "Tentative d'échantillonnage Gamma Inverse dans (" << a << ", " << b << ")" << std::endl;

        // 3. Critère d'acceptation : if a <= sample <= b:
        if (sample >= a && sample <= b) {
            samples.push_back(sample);
        }
    }
    
    return samples;
}


Vector truncatedNormalSample(double mu, double sigma, double a, double b, size_t size) {
    
    // Vérification des bornes
    if (a >= b || sigma <= 0.0) {
        throw std::invalid_argument("Erreur truncated_normal: Les bornes sont invalides (a >= b) ou sigma <= 0.");
    }

    Vector samples;
    samples.reserve(size); // Réserver la mémoire
    
    // Paramètre de la distribution Normale
    // std::normal_distribution prend la moyenne (mu) et l'écart-type (sigma)
    std::normal_distribution<double> normal_dist(mu, sigma);

    // Boucle d'échantillonnage par acceptation-rejet
    while (samples.size() < size) {
        
        // 1. Échantillonner la distribution Normale
        // La fonction generator(rd()) doit être définie statiquement dans fonctions.cpp
        // Utilisez le nom de votre générateur (ici 'generator')
        double sample = normal_dist(generator); 

        // --- Affichage de débogage (équivalent au print Python) ---
        // std::cout << "Tentative d'échantillonnage Gaussien sur (" << a << ", " << b << ")" << std::endl;

        // 2. Critère d'acceptation : if a <= sample <= b:
        if (sample >= a && sample <= b) {
            samples.push_back(sample);
        }
    }
    
    return samples;
}


Matrix EchantillonPsi(const Matrix& A, const Matrix& T, double b) {
    const size_t p = A.size();
    if (p == 0) {
        return Matrix();  // rien à faire
    }

    if (A[0].size() != p || T.size() != p || T[0].size() != p) {
        throw std::invalid_argument(
            "EchantillonPsi: A et T doivent être des matrices carrées de dimension p."
        );
    }

    // --- Initialisation ---
    Vector nui(p, 0.0);                 // nombre de voisins "suivants"
    Matrix T1(p, Vector(p, 0.0));
    Matrix Psi(p, Vector(p, 0.0));

    constexpr double tol = 1e-18;

    // --- Étape 1 : T1 et nui ---
    for (size_t j = 0; j < p; ++j) {
        const double Tjj = T[j][j];
        if (std::abs(Tjj) < tol) {
            throw std::runtime_error("EchantillonPsi: T[j,j] est nul, division impossible.");
        }
        const double invTjj = 1.0 / Tjj;

        // T1(:,j) = T(:,j) / T(j,j)
        for (size_t r = 0; r < p; ++r) {
            T1[r][j] = T[r][j] * invTjj;
        }

        // nui[j] = somme des voisins A[j,k] pour k > j
        double nuij = 0.0;
        const Vector& Aj = A[j];
        for (size_t k = j + 1; k < p; ++k) {
            nuij += Aj[k];
        }
        nui[j] = nuij;
    }

    // --- Étape 2 : échantillonnage des diagonales et des A[i,j]=1 ---
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    for (size_t i = 0; i < p; ++i) {
        const double df_chi = b + nui[i];
        if (df_chi <= 0.0) {
            throw std::runtime_error("EchantillonPsi: degré de liberté chi2 invalide (<= 0).");
        }

        std::chi_squared_distribution<double> chi_sq_dist(df_chi);
        Psi[i][i] = std::sqrt(chi_sq_dist(generator));

        // Termes hors diagonale où A[i,j] == 1
        const Vector& Ai = A[i];
        for (size_t j = i + 1; j < p; ++j) {
            if (Ai[j] != 0.0) {
                Psi[i][j] = normal_dist(generator);
            }
        }
    }

    // Helper : somme_{k = start}^{end-1} Psi[row,k] * T1[k,col]
    auto dotPsiT1 = [&](size_t rowPsi, size_t start, size_t end, size_t colT1) -> double {
        double s = 0.0;
        for (size_t k = start; k < end; ++k) {
            s += Psi[rowPsi][k] * T1[k][colT1];
        }
        return s;
    };

    // --- Étape 3 & 4 : termes A[i,j] == 0 par backward-substitution ---
    for (size_t i = 0; i < p; ++i) {
        const double Psi_ii = Psi[i][i];
        double invPsi_ii = 0.0;
        bool invPsi_ii_ready = false;

        for (size_t j = i + 1; j < p; ++j) {
            if (A[i][j] != 0.0) {
                continue; // on ne recalcul pas les liens présents
            }

            // Psi[i,j] = - sum_{k=i}^{j-1} Psi[i,k] * T1[k,j]
            double psi_ij = -dotPsiT1(i, i, j, j);

            // Correction si i > 0
            if (i > 0) {
                if (!invPsi_ii_ready) {
                    if (std::abs(Psi_ii) < tol) {
                        throw std::runtime_error(
                            "EchantillonPsi: division par zéro (Psi[i,i] nul) dans la correction."
                        );
                    }
                    invPsi_ii = 1.0 / Psi_ii;
                    invPsi_ii_ready = true;
                }

                double correction_sum = 0.0;

                for (size_t r = 0; r < i; ++r) {
                    // s1 = (Psi[r,i] + sum_{k=r}^{i-1} Psi[r,k] * T1[k,i]) / Psi[i,i]
                    const double s1_num = Psi[r][i] + dotPsiT1(r, r, i, i);
                    const double s1     = s1_num * invPsi_ii;

                    // s2 = Psi[r,j] + sum_{k=r}^{j-1} Psi[r,k] * T1[k,j]
                    const double s2 = Psi[r][j] + dotPsiT1(r, r, j, j);

                    correction_sum += s1 * s2;
                }

                psi_ij -= correction_sum;
            }

            Psi[i][j] = psi_ij;
        }
    }

    return Psi;
}


Matrix echantillonPsiMod(Matrix& Psi,
                         Matrix& A,
                         const Matrix& T,
                         double b,
                         size_t i1,
                         size_t j1,
                         bool Etat,
                         double tp) // tp inutilisé mais conservé pour la compatibilité
{
    const size_t p = A.size();
    if (p == 0) {
        // On retourne la même structure si la matrice est vide
        return Psi;
    }

    // Vérifications de base
    if (T.size() != p || T[0].size() != p) {
        throw std::invalid_argument("echantillonPsiMod: T doit être une matrice p x p.");
    }
    if (Psi.size() != p || Psi[0].size() != p) {
        throw std::invalid_argument("echantillonPsiMod: Psi doit être une matrice p x p.");
    }

    // On travaille uniquement sur la triangulaire supérieure (i1 < j1)
    if (i1 >= p || j1 >= p || i1 >= j1) {
        throw std::invalid_argument("echantillonPsiMod: indices invalides ou i1 >= j1.");
    }

    // --- Étape 1 : modification du lien A[i1, j1] ---
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    if (Etat) {
        // Ajout d'un lien
        A[i1][j1] = 1.0;
        A[j1][i1] = 1.0;
        Psi[i1][j1] = normal_dist(generator); // N(0,1) pour le nouveau lien
    } else {
        // Suppression d'un lien
        A[i1][j1] = 0.0;
        A[j1][i1] = 0.0;
        Psi[i1][j1] = 0.0; // sera recalculé avec les formules ci-dessous
    }

    // --- Étape 2 : calcul de T1 = T(:,j) / T(j,j) ---
    Matrix T1(p, Vector(p, 0.0));
    for (size_t j = 0; j < p; ++j) {
        const double Tjj = T[j][j];
        if (std::abs(Tjj) < 1e-18) {
            throw std::runtime_error("echantillonPsiMod: T[j][j] est nul, division impossible.");
        }
        const double invTjj = 1.0 / Tjj;
        for (size_t r = 0; r < p; ++r) {
            T1[r][j] = T[r][j] * invTjj;
        }
    }

    // Helper pour remplacer les sommes de type :
    // sum_{k = start}^{end-1} Psi[row, k] * T1[k, col]
    auto dotPsiT1 = [&](size_t rowPsi, size_t start, size_t end, size_t colT1) -> double {
        double s = 0.0;
        for (size_t k = start; k < end; ++k) {
            s += Psi[rowPsi][k] * T1[k][colT1];
        }
        return s;
    };

    // --- Étape 3 & 4 : backward-substitution sur les termes A[i,j] == 0, i < j ---
    for (size_t i = 0; i < p; ++i) {
        const double Psi_ii = Psi[i][i];
        double invPsi_ii = 0.0;
        bool invPsi_ii_ready = false;

        for (size_t j = i + 1; j < p; ++j) {

            // On ne recalcule Psi[i,j] que si l'arête est absente
            if (A[i][j] != 0.0) {
                continue;
            }

            // Terme de base :
            // Psi[i,j] = - sum_{k=i}^{j-1} Psi[i,k] * T1[k,j]
            double psi_ij = -dotPsiT1(i, i, j, j);

            // Correction si i > 0
            if (i > 0) {
                if (!invPsi_ii_ready) {
                    if (std::abs(Psi_ii) < 1e-18) {
                        throw std::runtime_error(
                            "echantillonPsiMod: division par zéro (Psi[i,i] nul) dans la correction."
                        );
                    }
                    invPsi_ii = 1.0 / Psi_ii;
                    invPsi_ii_ready = true;
                }

                double correction_sum = 0.0;

                for (size_t r = 0; r < i; ++r) {
                    // s1 = (Psi[r,i] + sum_{k=r}^{i-1} Psi[r,k] * T1[k,i]) / Psi[i,i]
                    const double s1_num = Psi[r][i] + dotPsiT1(r, r, i, i);
                    const double s1 = s1_num * invPsi_ii;

                    // s2 = Psi[r,j] + sum_{k=r}^{j-1} Psi[r,k] * T1[k,j]
                    const double s2 = Psi[r][j] + dotPsiT1(r, r, j, j);

                    correction_sum += s1 * s2;
                }

                psi_ij -= correction_sum;
            }

            Psi[i][j] = psi_ij;
        }
    }

    return Psi; // modifié en place, retour pour compatibilité
}
/**
 * Calcule la transposée d'une matrice M.
 */
Matrix Transpose(const Matrix& M) {
    size_t rows = M.size();
    if (rows == 0) return Matrix();
    size_t cols = M[0].size();

    Matrix MT(cols, Vector(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            MT[j][i] = M[i][j];
        }
    }
    return MT;
}


/**
 * Calcule le produit matriciel A * B.
 */



MatricesPrecision precisionMatrix(const Matrix& Psi) {
    size_t n = Psi.size();
    if (n == 0) return MatricesPrecision();
    if (n != Psi[0].size()) {
        throw std::invalid_argument("Erreur precisionMatrix: La matrice Psi doit être carrée.");
    }
    double detb=0;
    for (size_t i=0; i<n;++i) {detb +=2*std::log(Psi[i][i]);}

    MatricesPrecision result;

    // 1. Calculer Omega = Psi^T * Psi
    Matrix Psi_T = Transpose(Psi);
    result.Omega = multiplyMatrices(Psi_T, Psi);
    result.detb=detb;
    
    // 2. Calculer Sigma = Omega^-1
    // ATTENTION: Omega est une matrice PLÈNE symétrique. 
    // La fonction Inverse(Phi) traduite n'est valable que pour les matrices TRIANGULAIRES.
    // L'inversion d'une matrice pleine nécessite une librairie d'algèbre linéaire (ex: Eigen) 
    // ou une implémentation de l'inverse générale (par exemple, via décomposition LU/cholesky/SVD), 
    // ce qui dépasse la portée du C++ standard.
    
    // Si la matrice est petite, l'approche la plus stable mathématiquement est :
    // Sigma = (Psi^-1) * (Psi^-1)^T
    try {
        Matrix Psi_Inv = Inverse(Psi); // Utilise l'inverse triangulaire que nous avons implémenté
        Matrix Psi_Inv_T = Transpose(Psi_Inv);
        // Multiplication : (Psi^-1) * (Psi^-1)^T
        result.Sigma = multiplyMatrices(Psi_Inv, Psi_Inv_T);
    } catch (const std::exception& e) {
        std::cerr << "Avertissement precisionMatrix: Échec du calcul de Sigma via Psi_Inv. " << e.what() << std::endl;
        // Retourne une matrice nulle si l'inverse de Psi échoue
        result.Sigma = Matrix(n, Vector(n, 0.0));
    }
    
    return result;
}

double constanteNormalisation(const Matrix& A, double b, const Matrix& T, size_t count) {
    const size_t p = A.size();
    if (p == 0 || count == 0) {
        return 0.0;
    }

    if (T.size() != p || T[0].size() != p) {
        throw std::invalid_argument("constanteNormalisation: T doit être une matrice p x p.");
    }

    // --- 0. Pré-calculs ---
    const double log2pi = std::log(2.0 * M_PI);
    const double log2   = std::log(2.0);

    // --- 1. hi et nui ---
    // hi[j] = somme des A[j,k] pour k < j
    // nui[j] = somme des A[j,k] pour k > j
    Vector hi(p, 0.0);
    Vector nui(p, 0.0);

    for (size_t j = 0; j < p; ++j) {
        const Vector& Aj = A[j];
        double hij = 0.0;
        double nuij = 0.0;

        for (size_t k = 0; k < j; ++k) {
            hij += Aj[k];
        }
        for (size_t k = j + 1; k < p; ++k) {
            nuij += Aj[k];
        }

        hi[j]  = hij;
        nui[j] = nuij;
    }

    // --- 2. Partie analytique ---
    double const_term = 0.0;

    for (size_t i = 0; i < p; ++i) {
        const double nui_i = nui[i];
        const double hi_i  = hi[i];

        const double Tii = T[i][i];
        if (Tii <= 0.0) {
            throw std::runtime_error("constanteNormalisation: T[i,i] doit être positif.");
        }

        // np.log(gamma(0.5*(b + nui[i])))
        const_term += std::lgamma(0.5 * (b + nui_i));

        // (0.5 * nui[i]) * log(2*pi)
        const_term += 0.5 * nui_i * log2pi;

        // (b + nui[i] + hi[i]) * log(T[i,i])
        const_term += (b + nui_i + hi_i) * std::log(Tii);

        // 0.5*(nui[i] + b) * log(2)
        const_term += 0.5 * (nui_i + b) * log2;
    }

    // --- 3. Monte Carlo ---
    // AA = matrice de 1 (pour 1 - A)
    Matrix AA(p, Vector(p, 1.0));

    // On veut log( (1/count) * sum_j exp(z_j) ), z_j = -0.5 * inner_sum_j
    // Pour la stabilité numérique, on fait un log-sum-exp.
    std::vector<double> log_terms;
    log_terms.reserve(count);

    for (size_t iter = 0; iter < count; ++iter) {
        Matrix Psi = EchantillonPsi(A, T, b);

        double inner_sum = 0.0;
        for (size_t r = 0; r < p; ++r) {
            const Vector& AAr = AA[r];
            const Vector& Ar  = A[r];
            const Vector& Psir = Psi[r];
            for (size_t c = 0; c < p; ++c) {
                const double diff = AAr[c] - Ar[c];     // = 1 - A[r][c]
                const double x    = Psir[c];
                inner_sum += diff * (x * x);           // Psi^2 sans std::pow
            }
        }

        const double z = -0.5 * inner_sum;
        log_terms.push_back(z);
    }

    // log-sum-exp des z_j
    double max_log = log_terms[0];
    for (size_t k = 1; k < log_terms.size(); ++k) {
        if (log_terms[k] > max_log) {
            max_log = log_terms[k];
        }
    }

    double sum_exp = 0.0;
    for (double z : log_terms) {
        sum_exp += std::exp(z - max_log);
    }

    if (sum_exp <= 0.0 || !std::isfinite(sum_exp)) {
        std::cerr << "constanteNormalisation: somme Monte Carlo invalide, renvoie -inf.\n";
        return -std::numeric_limits<double>::infinity();
    }

    const double log_f_sum = max_log + std::log(sum_exp);
    const double log_Z = const_term + log_f_sum - std::log(static_cast<double>(count));

    return log_Z;
}


Matrix structureMatrice(const Matrix& Omega) {
    size_t p = Omega.size();
    if (p == 0) return Matrix();
    if (p != Omega[0].size()) {
        throw std::invalid_argument("Erreur structureMatrice: La matrice Omega doit être carrée.");
    }
    
    // Initialisation: aOmega = np.eye(p) (Matrice identité)
    // Nous allons plutôt initialiser à zéro puis ajouter les 1.
    Matrix aOmega(p, Vector(p, 0.0));

    // Construction de la matrice d'adjacence
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            
            // Si l'élément est non nul (au-delà d'une petite tolérance numérique)
            if (std::abs(Omega[i][j]) > 1e-12) {
                aOmega[i][j] = 1.0;
            }
        }
        
        // Assurer que la diagonale est à zéro (pas de boucle)
        // aOmega[i,i]=0
        aOmega[i][i] = 0.0;
    }
    
    return aOmega;
}


Matrix retirerUnNoeud(const Matrix& Omega) {
    const size_t p = Omega.size();
    
    // Cas matrice vide : on renvoie la même
    if (p == 0) {
        return Omega;
    }

    if (Omega[0].size() != p) {
        throw std::invalid_argument("retirerUnNoeud: la matrice Omega doit être carrée.");
    }

    // Copie de travail
    Matrix omegaPrime = Omega;

    constexpr double tol = 1e-12;

    double min_abs_value = std::numeric_limits<double>::max();
    int i_0 = -1;
    int j_0 = -1;

    // Recherche de la plus petite valeur absolue non nulle (hors diagonale, triangulaire sup)
    for (size_t i = 0; i < p; ++i) {
        const Vector& row = Omega[i];
        for (size_t j = i + 1; j < p; ++j) {
            const double val = row[j];
            const double aval = std::abs(val);

            if (aval > tol && aval < min_abs_value) {
                min_abs_value = aval;
                i_0 = static_cast<int>(i);
                j_0 = static_cast<int>(j);
            }
        }
    }

    // Si on a trouvé une arête à retirer
    if (i_0 != -1 && j_0 != -1) {
        omegaPrime[i_0][j_0] = 0.0;
        omegaPrime[j_0][i_0] = 0.0;
        // Optionnel: debug
        // std::cout << "Lien retire: (" << i_0 << ", " << j_0
        //           << ") valeur abs = " << min_abs_value << '\n';
    } else {
        // Aucune arête hors diagonale à retirer -> graphe déjà vide
        // On renvoie simplement la même structure (omegaPrime == Omega)
    }

    return omegaPrime;
}
/**
 * Traduction de ajoutUnNoeud(aOmega)
 * Propose aléatoirement l'ajout d'un lien (A[i,j]=0) par échantillonnage.
 */
PropAjoutRetrait ajoutUnNoeud(const Matrix& aOmega) {
    const size_t p = aOmega.size();
    if (p < 2) {
        throw std::invalid_argument("ajoutUnNoeud: dimension p < 2.");
    }

    PropAjoutRetrait result;
    result.A = aOmega;      // on part d'une copie de la structure existante
    result.i0 = 0;
    result.j0 = 0;

    constexpr double tol = 1e-12;
    constexpr int max_tries = 10;

    // Distribution uniforme sur {0, ..., p-1}
    std::uniform_int_distribution<size_t> distrib_p(0, p - 1);

    bool success = false;
    size_t i_0 = 0, j_0 = 0;

    // --- 1) Essais aléatoires limités ---
    for (int attempt = 0; attempt < max_tries && !success; ++attempt) {
        const size_t i = distrib_p(generator);
        const size_t j = distrib_p(generator);

        if (i == j) continue; // pas de boucle sur soi-même

        // On teste sur la matrice originale aOmega pour savoir si le lien existe déjà
        if (std::abs(aOmega[i][j]) < tol) {
            // Ajout de l'arête dans la copie
            result.A[i][j] = 1.0;
            result.A[j][i] = 1.0;

            if (j < i) {
                i_0 = j;
                j_0 = i;
            } else {
                i_0 = i;
                j_0 = j;
            }

            result.i0 = i_0;
            result.j0 = j_0;
            success   = true;
        }
    }

    // --- 2) Si les essais aléatoires échouent, recherche déterministe ---
    if (!success) {
        for (size_t i = 0; i < p; ++i) {
            for (size_t j = i + 1; j < p; ++j) {
                if (std::abs(aOmega[i][j]) < tol) {
                    result.A[i][j] = 1.0;
                    result.A[j][i] = 1.0;

                    result.i0 = i;
                    result.j0 = j;
                    success   = true;
                    return result; // on sort dès qu’on a trouvé un lien manquant
                }
            }
        }
    }

    // --- 3) Cas pire : graphe complet ---
    // Aucun lien à ajouter -> on retourne la structure d'origine telle quelle
    // result.A est déjà égal à aOmega, i0/j0 laissés à 0 par convention.
    return result;
}


/**
 * Traduction de S_epsi(theta, xi)
 * Calcule la matrice de somme de carrés des erreurs S_epsilon.
 */
Matrix S_epsi(const Matrix& theta, const Vector& xi) {
    const size_t p = theta.size(); // Nombre de lignes
    if (p == 0) {
        throw std::invalid_argument("S_epsi: Matrice theta est vide.");
    }

    const size_t T1 = theta[0].size(); // Nombre de périodes
    if (T1 < 1) {
        return Matrix(p, Vector(p, 0.0));
    }

    if (xi.size() != p) {
        throw std::invalid_argument("S_epsi: Le vecteur xi doit être de taille p.");
    }

    // Vérifier que toutes les lignes de theta ont la même longueur
    for (size_t i = 0; i < p; ++i) {
        if (theta[i].size() != T1) {
            throw std::invalid_argument("S_epsi: theta a des lignes de tailles différentes.");
        }
    }

    // --- Pré-calcul des logs de theta ---
    Matrix log_theta(p, Vector(T1));
    for (size_t i = 0; i < p; ++i) {
        for (size_t t = 0; t < T1; ++t) {
            const double val = theta[i][t];
            if (val <= 0.0) {
                throw std::runtime_error("S_epsi: Erreur log, theta[i,t] doit être > 0.");
            }
            log_theta[i][t] = std::log(val);
        }
    }

    // --- Matrice résultat (p x p), initialisée à 0 ---
    Matrix tp(p, Vector(p, 0.0));

    // --- Terme initial : XXt(log(theta[:,0])) ---
    // v = log_theta[:,0]
    {
        // produit externe v v^T, en exploitant la symétrie
        for (size_t r = 0; r < p; ++r) {
            const double vr = log_theta[r][0];
            for (size_t c = 0; c <= r; ++c) {
                const double vc   = log_theta[c][0];
                const double val  = vr * vc;
                tp[r][c] += val;
                if (c != r) {
                    tp[c][r] += val;
                }
            }
        }
    }

    // --- Boucle sur t = 0..T1-2 pour les epsi_t ---
    Vector epsi_t(p, 0.0);

    for (size_t t = 0; t + 1 < T1; ++t) {
        // epsi_t[i] = xi[i] * log_theta[i][t] - log_theta[i][t+1]
        for (size_t i = 0; i < p; ++i) {
            epsi_t[i] = xi[i] * log_theta[i][t] - log_theta[i][t + 1];
        }

        // Ajouter XXt(epsi_t) à tp, toujours en utilisant la symétrie
        for (size_t r = 0; r < p; ++r) {
            const double vr = epsi_t[r];
            for (size_t c = 0; c <= r; ++c) {
                const double vc   = epsi_t[c];
                const double val  = vr * vc;
                tp[r][c] += val;
                if (c != r) {
                    tp[c][r] += val;
                }
            }
        }
    }

    return tp;
}


double logLikhoodBetaJ(const Matrix& X, const double& ALPHA, const double& BETA, const Vector& S, int j) {
    size_t p = X.size(); // Nombre de régions (lignes)
    if (p == 0) return 0.0;
    
    size_t T = X[0].size(); // Nombre de périodes de temps (colonnes)
    size_t S_size = S.size(); // Taille totale (p * T)

    if (S_size != p * T) {
        throw std::invalid_argument("Erreur logLikhoodBetaJ: S doit être de taille p * T.");
    }
    
    double tp = 0.0; // Accumulateur de log-vraisemblance

    // Itérer sur tous les points (i, t)
    for (size_t k = 0; k < S_size; ++k) {
        
        // Vérifier si ce point appartient au cluster j
        // Note: S[k] est un double représentant l'indice du cluster
        if (static_cast<int>(S[k]) == j) {
            
            // Calcul des indices de la matrice à partir de l'indice plat k
            size_t i = k % p;    // Indice de la région (ligne)
            size_t t = k / p;    // Indice du temps (colonne)
            
            // Assurez-vous que les indices ne dépassent pas les bornes
            if (i >= p || t >= T) {
                 throw std::runtime_error("Erreur logLikhoodBetaJ: Calcul d'indice i, t incorrect.");
            }
            // Accumuler la log-vraisemblance du PDF Bêta
            tp += logBetaPDF(X[i][t], ALPHA, BETA);
        }
    }
    
    return tp;
}


double poissonGammaLog(int y, double alpha, double beta) {
    
    if (y < 0 || alpha <= 0.0 || beta <= 0.0) {
        throw std::invalid_argument("Erreur poissonGammaLog: Les paramètres ou l'observation sont invalides.");
    }
    
    // Convertir y en double pour les calculs lgamma
    double y_double = static_cast<double>(y);
    
    // --- Calcul des termes de la log-vraisemblance ---

    // 1. Termes Gamma (coefficient binomial négatif)
    // log(Gamma(alpha+y)) - log(Gamma(alpha)) - log(Gamma(y+1))
    // Note: log(Gamma(y+1)) est le log-factoriel de y (log(y!))
    double term_gamma = std::lgamma(alpha + y_double) - std::lgamma(alpha) - std::lgamma(y_double + 1.0);
    
    // 2. Termes de probabilité (selon la paramétrisation Poisson-Gamma)
    // La PMF est souvent vue comme NB(r=alpha, p = beta/(1+beta))
    
    // Terme 2a: alpha * log(beta)
    double term_2a = alpha * std::log(beta);

    // Terme 2b: - (alpha + y) * log(1 + beta)
    double term_2b = -(alpha + y_double) * std::log(1.0 + beta);
    
    // Le terme final est le log(PMF)
    double log_pmf = term_gamma + term_2a + term_2b;
    
    return log_pmf;
}


double logLikhoodPoissonGammaJ(const Matrix& X, const Matrix& ALPHA, const Matrix& BETA, const Vector& S, int j) {
    size_t p = X.size(); // Nombre de régions (lignes)
    if (p == 0) return 0.0;
    
    size_t T = X[0].size(); // Nombre de périodes de temps (colonnes)
    size_t S_size = S.size(); // Taille totale (p * T)

    if (S_size != p * T) {
        throw std::invalid_argument("Erreur logLikhoodPoissonGammaJ: S doit être de taille p * T.");
    }
    
    double tp = 0.0; // Accumulateur de log-vraisemblance

    // Itérer sur tous les points (i, t)
    for (size_t k = 0; k < S_size; ++k) {
        
        // Vérifier si ce point appartient au cluster j
        if (static_cast<int>(S[k]) == j) {
            
            // Calcul des indices de la matrice à partir de l'indice plat k
            size_t i = k % p;    // Indice de la région (ligne)
            size_t t = k / p;    // Indice du temps (colonne)
            
            // Les observations X doivent être converties en entier pour le PMF de comptage
            // Nous supposons que X contient des doubles qui sont en réalité des entiers.
            int y_it = static_cast<int>(std::round(X[i][t]));

            double alpha_it = ALPHA[i][t];
            double beta_it = BETA[i][t];

            // Accumuler la log-vraisemblance du PMF Poisson-Gamma
            tp += poissonGammaLog(y_it, alpha_it, beta_it);
        }
    }
    
    return tp;
}



Matrix extractMatrix(const Matrix& B, int j, const Vector& S, int Nj) {
    size_t n = B.size(); // Dimension de la matrice complète B
    if (n == 0 || Nj <= 0) return Matrix();
    if (n != B[0].size()) {
        throw std::invalid_argument("Erreur extractMatrix: La matrice B doit être carrée.");
    }

    // 1. Identifier les indices de ligne/colonne appartenant au cluster j
    // Les indices 'k' correspondent aux indices dans la matrice B
    std::vector<size_t> indices_cluster;
    
    // Le code Python itère sur 'ss[l]', qui est la partie de S qui est pertinente pour B.
    // Si B est une matrice p x p (covariance entre régions), S doit être de taille p.
    // Si B est une matrice (p*T1) x (p*T1), S est de taille p*T1.
    // Par convention dans le clustering, B[k, l] est souvent la covariance entre l'élément k et l.
    
    // Nous supposons ici que B est de taille n x n et que les n premiers éléments de S correspondent aux indices de B.
    if (S.size() < n) {
         throw std::invalid_argument("Erreur extractMatrix: S est trop petit par rapport à B.");
    }
    
    for (size_t k = 0; k < n; ++k) {
        // Vérifie si l'élément k appartient au cluster j
        if (static_cast<int>(S[k]) == j) {
            indices_cluster.push_back(k);
        }
    }
    
    // Vérification: Le nombre d'indices trouvés doit correspondre à Nj
    if (indices_cluster.size() != static_cast<size_t>(Nj)) {
        // C'est souvent un signe d'erreur si le Nj passé est faux
        std::cerr << "Avertissement extractMatrix: Nj (" << Nj << ") ne correspond pas aux indices trouves (" << indices_cluster.size() << "). Utilisation de la taille reelle." << std::endl;
        Nj = indices_cluster.size();
        if (Nj == 0) return Matrix();
    }
    
    // 2. Créer la sous-matrice B_Sj (Nj x Nj)
    Matrix Bsj(Nj, Vector(Nj));

    for (int r = 0; r < Nj; ++r) { // Indice de ligne dans Bsj
        for (int c = 0; c < Nj; ++c) { // Indice de colonne dans Bsj
            
            // Indice réel de la ligne dans B
            size_t idx_r = indices_cluster[r];
            
            // Indice réel de la colonne dans B
            size_t idx_c = indices_cluster[c];
            
            // Extraction: Bsj[r, c] = B[idx_r, idx_c]
            Bsj[r][c] = B[idx_r][idx_c];
        }
    }

    return Bsj;
}


Vector extractVect(const Vector& U, int j, const Vector& S, int Nj) {
    size_t n = U.size(); // Dimension du vecteur complet U
    if (n == 0 || Nj <= 0) return Vector();
    
    if (S.size() != n) {
        throw std::invalid_argument("Erreur extractVect: Les vecteurs U et S doivent avoir la même dimension.");
    }

    // 1. Créer le sous-vecteur U_Sj de la taille attendue
    Vector Usj;
    Usj.reserve(Nj); // Réserver la mémoire pour Nj éléments

    // 2. Accumuler les éléments
    // Nous itérons sur la taille du vecteur U (n)
    for (size_t i = 0; i < n; ++i) {
        
        // Vérifie si l'élément i appartient au cluster j
        if (static_cast<int>(S[i]) == j) {
            
            // Ajouter l'élément U[i] au sous-vecteur Usj
            Usj.push_back(U[i]);
        }
    }
    
    // Vérification finale: Si le Nj passé est faux, nous avertissons/levons une erreur.
    if (Usj.size() != static_cast<size_t>(Nj)) {
        std::cerr << "Avertissement extractVect: Nj passe (" << Nj << ") ne correspond pas aux éléments trouves (" << Usj.size() << ")." << std::endl;
        // La fonction retourne le vecteur trouvé, qui est de taille Usj.size()
    }

    return Usj;
}

/**
 * Traduction de isValueInMatrix(SS, i, t, p)
 * Vérifie si le label de cluster de l'état (i, t) a déjà été attribué à un état précédent.
 */
std::pair<int, int> isValueInMatrix(const Vector& SS, int i, int t, int p) {
    
    if (p <= 0 || i < 0 || t < 0) {
        throw std::invalid_argument("Erreur isValueInMatrix: Les indices i, t ou p sont invalides.");
    }
    
    // Indice plat (linéaire) de l'état cible: cible = i + p * t
    // Le Python utilise : l in range(i + p*t)
    int target_index = i + p * t;
    
    // Vérification des bornes
    if (target_index > SS.size()) {
         throw std::out_of_range("Erreur isValueInMatrix: L'indice cible dépasse la taille de SS.");
    }

    // Label du cluster cible
    double target_cluster_label = SS[target_index];

    // Boucle: for l in range(target_index)
    for (int l = 0; l < target_index; ++l) {
        
        // if(SS[l]==SS[i+p*t]):
        // Si le label du cluster est identique
        if (std::abs(SS[l] - target_cluster_label) < 1e-12) { // Comparaison de doubles
            
            // Calcul des indices de la matrice (i_prev, t_prev) à partir de l
            // i_prev = l - p * int(l / p) -> équivalent à l % p
            int i_prev = l % p;
            
            // t_prev = int(l / p)
            int t_prev = l / p;
            
            // return(i_prev, t_prev)
            return std::make_pair(i_prev, t_prev);
        }
    }
    
    // return(-1,-1) si aucun match n'est trouvé
    return std::make_pair(-1, -1);
}


double meanSS(const Matrix& X, const Vector& S, int j) {
    const size_t p = X.size();             // nombre de lignes
    if (p == 0) return 0.0;

    const size_t T1 = X[0].size();         // nombre de périodes
    const size_t expected_size = p * T1;

    if (S.size() != expected_size) {
        throw std::invalid_argument("meanSS: S.size() != p * T1");
    }

    double sum = 0.0;
    int count = 0;

    for (size_t t = 0; t < T1; ++t) {
        const size_t offset = t * p;
        for (size_t i = 0; i < p; ++i) {
            const size_t index = offset + i;  // index dans S

            if (static_cast<int>(S[index]) == j) {
                sum += X[i][t];
                ++count;
            }
        }
    }

    if (count == 0) {
        // Aucun élément dans le cluster j → éviter un NaN
        return 0.0;
    }

    return sum / count;
}


/**
 * Additionne deux matrices A + B.
 */
Matrix AddMatrices(const Matrix& A, const Matrix& B) {
    size_t rows = A.size();
    if (rows == 0) return Matrix();
    size_t cols = A[0].size();
    
    if (B.size() != rows || B[0].size() != cols) {
        throw std::invalid_argument("Erreur AddMatrices: Les matrices doivent être de mêmes dimensions.");
    }
    
    Matrix C(rows, Vector(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}



/**
 * Calcule la différence de deux matrices A - B.
 */
Matrix SubtractMatrices(const Matrix& A, const Matrix& B) {
    size_t rows = A.size();
    if (rows == 0) return Matrix();
    size_t cols = A[0].size();
    
    if (B.size() != rows || B[0].size() != cols) {
        throw std::invalid_argument("Erreur SubtractMatrices: Les matrices doivent être de mêmes dimensions.");
    }
    
    Matrix C(rows, Vector(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

/**
 * Calcule la somme des valeurs absolues des éléments de la matrice. (np.sum(np.abs(M)))
 */
double SumAbsMatrix(const Matrix& M) {
    double sum = 0.0;
    for (const auto& row : M) {
        for (double val : row) {
            sum += std::abs(val);
        }
    }
    return sum;
}


/**
 * Extrait une somme de sous-vecteur multipliée (équivalent à np.sum(Psi[r-1,(r-1):(i-1)]*T1[(r-1):(i-1),i-1])).
 * Cette fonction utilitaire permet de traduire les opérations de slicing et de produit par éléments.
 * Dans cette fonction, nous traduisons l'opération par des boucles.
 */

double numpySumSlice(const Matrix& M1, size_t r1, size_t c1_start, size_t c1_end, const Matrix& M2, size_t c2_start, size_t c2_end, size_t c2_fixed) {
    double sum = 0.0;
    // Les indices C++ sont 0-basés. Les indices Python [a:b] incluent a et excluent b.
    // L'itération Python [k:(j)] est k=k, k+1, ..., j-1.
    // Votre code Python utilise Psi[i-1,(i-1):(j-1)] * T1[(i-1):(j-1),j-1]
    
    // Simplifions l'indice réel k allant de c1_start (inclus) à c1_end (exclu).
    for (size_t k = c1_start; k < c1_end; ++k) {
        // M1[r1][k] * M2[k][c2_fixed]
        sum += M1[r1][k] * M2[k][c2_fixed];
    }
    return sum;
}




/**
 * Traduction de EchantillonPsiMod(Psi, A, T, b, i1, j1, Etat, tp)
 * Modifie Psi* et A* après l'ajout ou la suppression du lien (i1, j1).
 * Note: i1, j1 sont des indices 0-basés après la conversion depuis le Python.
 */
Matrix EchantillonPsiMod(Matrix& Psi,
                         Matrix& A,
                         const Matrix& T,
                         double b,
                         size_t i1,
                         size_t j1,
                         bool Etat,
                         double tp) // tp inutilisé mais on garde la signature
{
    const size_t p = A.size();
    if (p == 0) {
        return Matrix();
    }

    constexpr bool verbose = false;

    // --- 1. Modification du lien (A[i1, j1]) ---
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    if (Etat) {
        if (verbose) std::cout << "we are adding an edge\n";
        A[i1][j1] = 1.0;
        A[j1][i1] = 1.0;
        Psi[i1][j1] = normal_dist(generator_prior); // N(0,1) pour le nouveau lien
    } else {
        if (verbose) std::cout << "we are removing an edge\n";
        A[i1][j1] = 0.0;
        A[j1][i1] = 0.0;
        // Psi[i1][j1] sera mis à jour plus bas si nécessaire
    }

    // --- 2. Calculer T1 = T(:,j) / T(j,j) ---
    Matrix T1(p, Vector(p, 0.0));

    for (size_t j = 0; j < p; ++j) {
        const double Tjj = T[j][j];
        if (std::abs(Tjj) < 1e-18) {
            throw std::runtime_error("EchantillonPsiMod: T[j,j] est nul, division impossible.");
        }
        const double invTjj = 1.0 / Tjj;
        for (size_t r = 0; r < p; ++r) {
            T1[r][j] = T[r][j] * invTjj;
        }
    }

    // Petit helper inline pour remplacer numpySumSlice :
    // somme_{k=start}^{end-1} Psi[rowPsi, k] * T1[k, colT1]
    auto dotPsiT1 = [&](size_t rowPsi, size_t start, size_t end, size_t colT1) -> double {
        double s = 0.0;
        for (size_t k = start; k < end; ++k) {
            s += Psi[rowPsi][k] * T1[k][colT1];
        }
        return s;
    };

    // --- 3. Mise à jour première ligne (i = 0) ---
    size_t i_c = 0;
    for (size_t j_c = i_c + 1; j_c < p; ++j_c) {
        if (std::abs(A[i_c][j_c]) < 1e-12) { // A[i_c, j_c] == 0
            const double sum_term = dotPsiT1(i_c, i_c, j_c, j_c);
            Psi[i_c][j_c] = -sum_term;
        }
    }

    // --- 4. Lignes suivantes (i > 0) ---
    for (size_t i_c_p = 1; i_c_p < p; ++i_c_p) {
        i_c = i_c_p;
        const double psi_ii = Psi[i_c][i_c];
        if (std::abs(psi_ii) < 1e-18) {
            throw std::runtime_error("EchantillonPsiMod: Division par zéro (Psi[i,i] est nul).");
        }
        const double inv_psi_ii = 1.0 / psi_ii;

        for (size_t j_c = i_c + 1; j_c < p; ++j_c) {
            if (std::abs(A[i_c][j_c]) < 1e-12) { // A[i_c, j_c] == 0

                double psi_ij = 0.0;

                // Terme 1 : somme sur r = 0..i-1
                for (size_t r_c = 0; r_c < i_c; ++r_c) {

                    // s1_num = Psi[r,i] + sum_{k=r}^{i-1} Psi[r,k] * T1[k,i]
                    const double s1_num = Psi[r_c][i_c] + dotPsiT1(r_c, r_c, i_c, i_c);
                    const double s1     = s1_num * inv_psi_ii;

                    // s2 = Psi[r,j] + sum_{k=r}^{j-1} Psi[r,k] * T1[k,j]
                    const double s2 = Psi[r_c][j_c] + dotPsiT1(r_c, r_c, j_c, j_c);

                    psi_ij -= s1 * s2;
                }

                // Terme 2 : - sum_{k=i}^{j-1} Psi[i,k] * T1[k,j]
                const double sum_term_2 = dotPsiT1(i_c, i_c, j_c, j_c);
                psi_ij -= sum_term_2;

                Psi[i_c][j_c] = psi_ij;
            }
        }
    }

    return Psi; // modifié en place, retour pour compatibilité
}

MCMC_Result_Full MCMC(
    double lambda_0,
    const Matrix& B_in,
    const Matrix& Sigma,
    double detB_init,
    const Matrix& Psi_init,
    const Matrix& T_init,
    double b,
    const Matrix& S_epsi,
    double neta,
    Matrix& A_Theta,
    int T_periods,
    bool verbose // ajout d'un flag pour contrôler les prints
) {
    verbose=false;
    if (verbose)
        std::cout << "Commencement de l'echantillonnage B" << std::endl;
    
    const size_t p = B_in.size();
    if (p == 0) {
        throw std::invalid_argument("MCMC: La dimension p est nulle.");
    }
    
    // Copies locales modifiables
    Matrix B    = B_in;
    Matrix Psi  = Psi_init;
    Matrix T    = T_init;
    Matrix Sigma_current = Sigma;
    double detB = detB_init;
    
    // --- Pré-calcul de T1_star = sqrt(b) * cholesky(Sigma)' ---
    Matrix cholesky_Sigma = cholesky(Sigma_current);
    Matrix T1_star(p, Vector(p, 0.0));
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            // transposée de cholesky(Sigma)
            T1_star[i][j] = std::sqrt(b) * cholesky_Sigma[j][i];
        }
    }
    
    // --- 1. Choix du mouvement (Ajout/Retrait/Rien) ---
    std::uniform_int_distribution<> action_dist(0, 2); // 0: rien, 1: retrait, 2: ajout
    int rv = action_dist(generator_prior);
    std::bernoulli_distribution bernoulli_prop(0.5);
    
    Matrix A_Theta_star = A_Theta;
    Matrix Omega_star   = B;
    Matrix Sigma_star   = Sigma_current;
    Matrix Psi_star     = Psi;
    double detB_star    = detB;
    
    PropAjoutRetrait proposal_data;
    bool Etat = bernoulli_prop(generator_prior); // true = ajout, false = retrait
    
    if (rv > 0) {
        if (verbose) {
            if (rv == 2) std::cout << "We are adding an edge" << std::endl;
            else         std::cout << "We are removing an edge" << std::endl;
        }
        
        if (rv == 2) {
            proposal_data = ajoutUnNoeud(A_Theta);
        } else {
            proposal_data = extractNoeud(A_Theta);
        }
        A_Theta_star = proposal_data.A;
    }

    // On alloue B_star une fois pour toutes
    Matrix B_star(p, Vector(p, 0.0));

    // --- 2. Boucle de tentative de mise à jour (max 10 essais) ---
    int  it         = 0;
    bool continuous = true;
    
    while (continuous && it <= 10) {
        if (verbose)
            std::cout << "Iteration locale MCMC = " << it << std::endl;
        ++it;
        
        // Échantillonner Psi_star
        if (rv == 0) {
            // Échantillonnage sous la structure actuelle
            Psi_star = EchantillonPsi(A_Theta, T1_star, b);
        } else {
            // Échantillonnage sous la structure proposée (mouvement MH)
            Psi_star = EchantillonPsiMod(
                Psi, A_Theta_star, T1_star, b,
                proposal_data.i0, proposal_data.j0, Etat
            );
        }
        
        // Calculer Omega*, Sigma*, detB*
        MatricesPrecision MP_star = precisionMatrix(Psi_star);
        Omega_star = MP_star.Omega;
        Sigma_star = MP_star.Sigma;
        detB_star  = MP_star.detb; // on suppose que c'est le log-det B*
        
        // Calculer B_star = Omega* * sqrt(diag(Sigma*))
        // -> à vérifier avec ton modèle, c'est ce que suggère ton commentaire.
        for (size_t r = 0; r < p; ++r) {
            const double sqrtSigma_r = std::sqrt(Sigma_star[r][r]);
            for (size_t c = 0; c < p; ++c) {
                const double sqrtSigma_c = std::sqrt(Sigma_star[c][c]);
                B_star[r][c] = Omega_star[r][c] * sqrtSigma_r * sqrtSigma_c; 
                //B_star[r][c] = Omega_star[r][c]; 
               Sigma_star[r][c] = Sigma_star[r][c]/( sqrtSigma_r * sqrtSigma_c);
            }
            detB_star+=std::log(Omega_star[r][r]);
        }

        // --- 3. Vérification de la définie positivité ---
        try {
            // Ici tu utilises detB_star comme critère.
            // Si detB_star est un log-det, le seuil est à adapter.
            const double det_omega_star = detB_star;
            if (det_omega_star > 1e-12) {
                continuous = false;
            }
        } catch (const std::exception&) {
            // Si det() plante, on rejette la proposition en restant dans la boucle
        }
        
        // --- 4. MH : seulement si on a un mouvement de structure et matrice valide ---
        if (!continuous && rv > 0) {
            // a) Terme Wishart (l1)
            double l1 = detB_star - detB;
            double correction_factor =
                -1.0 * (neta + static_cast<double>(p) - static_cast<double>(T_periods)
                        + 0.5 * (b - 2.0));
            l1 *= correction_factor;
            
            // b) Terme d'erreur (trace((B* - B) S_epsi))
            double trace_term = 0.0;
            for (size_t i = 0; i < p; ++i) {
                for (size_t k = 0; k < p; ++k) {
                    trace_term += (B_star[i][k] - B[i][k]) * S_epsi[k][i];
                }
            }
            trace_term *= -0.5;
            
            // c) LASSO : -lambda_0 * (||B*||_1 - ||B||_1)
            const double SumAbs_B_star = SumAbsMatrix(B_star);
            const double SumAbs_B      = SumAbsMatrix(B);
            const double lasso_term    = -lambda_0 * (SumAbs_B_star - SumAbs_B);
            
            // d) Correction diagonale
            double diag_prior_correction = 0.0;
            for (size_t k = 0; k < p; ++k) {
                diag_prior_correction += (B_star[k][k] - B[k][k]);
            }
            
            const double cste = l1 + trace_term + lasso_term + diag_prior_correction;
            if (verbose)
                std::cout << "log-alpha (cste) = " << cste << std::endl;
            
            // e) Acceptation MH
            if (logUniformRvs() <= cste) {
                if (verbose)
                    std::cout << "### Acceptation : mise a jour de la matrice de precision ###"
                              << std::endl;
                
                B             = B_star;
                A_Theta       = A_Theta_star;
                Sigma_current = Sigma_star;
                detB          = detB_star;
                Psi           = Psi_star;
                T             = T1_star;
            }
        }
    }
    
    if (verbose)
        std::cout << "Fin de l'echantillonnage local" << std::endl;

    MCMC_Result_Full result;
    result.Psi      = Psi;
    result.T        = T;
    result.B        = B;
    result.Sigma    = Sigma_current;
    result.detB     = detB;
    result.A_Theta  = A_Theta;
    
    return result;
}
// --- Fonction MCMC Principale ---
/* MCMC_Result_Full MCMC(double lambda_0, Matrix& B_in, Matrix& Sigma, double detB_init, Matrix& Psi_init, Matrix& T_init, double b, const Matrix& S_epsi, double neta, Matrix& A_Theta, int T_periods) {
    
    std::cout << "Commencement de lechantillon B" << std::endl;
    
    size_t p = B_in.size();
    if (p == 0) throw std::invalid_argument("MCMC: La dimension p est nulle.");
    
    // Alias pour le travail MCMC (sera modifié)
    Matrix B = B_in;
    Matrix Psi = Psi_init;
    Matrix T = T_init;
    Matrix Sigma_current = Sigma;
    double detB = detB_init;
    
    // Calcul de T1_star = sqrt(b) * cholesky(Sigma)'.T
    Matrix cholesky_Sigma = cholesky(Sigma_current);
    Matrix T1_star(p, Vector(p)); 
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T1_star[i][j] = std::sqrt(b) * cholesky_Sigma[j][i]; // Transposée de cholesky
        }
    }
    
    // --- 1. Choix du mouvement (Ajout/Retrait/Rien) ---
    std::uniform_int_distribution<> action_dist(0, 2); // 0, 1, 2
    int rv = action_dist(generator_prior);
    std::bernoulli_distribution bernoulli_prop(0.5); 
    
    // Cas spéciaux (graphe plein ou vide)
    // Simplification: le cas rv=0 (sampling) est exécuté si les cas spéciaux ne s'appliquent pas.
    // rv=1 (Retrait), rv=2 (Ajout)
    
    Matrix A_Theta_star = A_Theta;
    Matrix Omega_star = B;
    Matrix Sigma_star = Sigma;
    Matrix Psi_star = Psi;
    double detB_star = detB;
    
    // Variables pour l'update MH
    PropAjoutRetrait proposal_data;
    bool Etat = (bernoulli_prop(generator_prior) == 1); // True pour ajouter, False pour retirer
    
    // Détermination de l'action rv
    if (rv > 0) {
        std::cout << (rv == 2 ? "We are adding an edge" : "We are removing an edge") << std::endl;
        
        if (rv == 2) { // Ajout
            proposal_data =ajoutUnNoeud(A_Theta);
        } else { // Retrait
            proposal_data = extractNoeud(A_Theta);
        }
        A_Theta_star = proposal_data.A;
    }

    // --- 2. Boucle de MCMC/MH (avec itération pour garantir la définie positivité) ---
    int it = 0;
    bool continuous = true;
    
    while (continuous && it <= 10) {
        std::cout << "Nous sommes a la " << it << " iterations" << std::endl;
        it++;
        
        if (rv == 0) {
            // Échantillonnage Psi* sous A_Theta
            Psi_star = EchantillonPsi(A_Theta, T1_star, b);
        } else {
            // Échantillonnage Psi* sous A_Theta_star modifié (MH)
            Psi_star = EchantillonPsiMod(Psi, A_Theta_star, T1_star, b, 
                                            proposal_data.i0, proposal_data.j0, Etat);
        }
        
        // Calculer Omega* et Sigma*
        MatricesPrecision MP_star = precisionMatrix(Psi_star);
        Omega_star = MP_star.Omega;
        Sigma_star = MP_star.Sigma;
        detB_star=MP_star.detb;
        
        // Calculer B_star = Omega* * sqrt(diag(Sigma*))
        // ICI ICI ICI ICI ICI ICI ICI ICI ICI ICI ICI 

        Matrix B_star(p, Vector(p));
        for (size_t r = 0; r < p; ++r) {
            for (size_t c = 0; c < p; ++c) {
               // B_star[r][c] = Omega_star[r][c] * std::sqrt(Sigma_star[r][r] * Sigma_star[c][c]);
               B_star[r][c] = Omega_star[r][c] ;
               //;* std::sqrt(Sigma_star[r][r] * Sigma_star[c][c]);
            }
        } 

        // --- 3. Vérification de la Définie Positivité (Eigenvalue Check) ---
        // Placeholder: Cette vérification est difficile sans une librairie externe.
        // Nous allons supposer que le calcul du déterminant suffit pour la non-singularité.
        try {
            //double det_omega_star = det(Omega_star);
            double det_omega_star = detB_star;
            if (det_omega_star > 1e-12) {
                continuous = false; // Sortir de la boucle si la matrice est non singulière
            }
        } catch (const std::exception& e) {
            // En cas d'erreur de déterminant, on rejette la proposition
        } 
        
        // --- Si rv > 0 (Proposition de structure) : Calcul du Ratio MH ---
        if (!continuous && rv > 0) { // Si Psi* est valide et que c'était un mouvement de structure
            
            // a) Terme Wishart (l1) : Log-Déterminant Ratio
            // Calculer l1 = detB_star - detB (Log-ratio de déterminants)
           // detB_star = 2.0 * std::log(std::abs(det(Psi_star))); // Calcul précis du detB* (log-déterminant)
            double l1 = detB_star - detB;
            
            // Terme de correction des degrés de liberté
            double correction_factor = -1.0 * (neta + (double)p - (double)T_periods + 0.5 * (b - 2.0)); 
            l1 = correction_factor * l1;
            
            // b) Terme d'Erreur (trace) : 0.5 * trace((B* - B) * S_epsi)
            Matrix B_diff = SubtractMatrices(B_star, B);
            Matrix B_diff_S_epsi = multiplyMatrices(B_diff, S_epsi);
            double trace_term = -0.5 * trace(B_diff_S_epsi);

            // c) Terme LASSO : -lambda_0 * (SumAbs(B*) - SumAbs(B))
            double SumAbs_B_star = SumAbsMatrix(B_star);
            double SumAbs_B = SumAbsMatrix(B);
            double lasso_term = -lambda_0 * (SumAbs_B_star - SumAbs_B);
            
            // d) Terme de Diagonale/Prior (Correction manuelle du Python)
            double diag_prior_correction = 0.0;
            for (size_t k = 0; k < p; ++k) {
                // Le Python a deux termes étranges qui semblent se simplifier à:
                // diag_prior_correction += B_star[k,k] - B[k,k]
                // Les autres termes du Python (avec A_Theta) sont souvent liés à la log-vraisemblance G-Wishart
                // Je simplifie à la forme du code Python (B*[k,k] - B[k,k])
                diag_prior_correction += B_star[k][k] - B[k][k];
            }
            
            // Final Log-Alpha (cste)
            double cste = l1 + trace_term + lasso_term + diag_prior_correction;

            std::cout << "Const ==" << cste << std::endl;
            
            // Décision d'Acceptation (np.log(st.uniform.rvs()) <= cste)
            if (logUniformRvs() <= cste) {
                std::cout << "#########Yesss, Updating of the precision matrix#######, God news###########" << std::endl;
                B = B_star;
                A_Theta = A_Theta_star;
                Sigma_current = Sigma_star;
                detB = detB_star;
                Psi = Psi_star;
                T = T1_star; 
            }
        }
    }
    
    std::cout << "terminer l'echantillonnage" << std::endl;

    // --- Remplissage de la structure de retour ---
    MCMC_Result_Full result;
    result.Psi = Psi;
    result.T = T;
    result.B = B;
    result.Sigma = Sigma_current;
    result.detB = detB;
    result.A_Theta = A_Theta;
    
    return result;
} */


// Matrice diviser par une constante
Matrix DivideMatrixByScalar(const Matrix& M, double C) {
    size_t rows = M.size();
    if (rows == 0) return Matrix();

    // Gestion de la division par zéro
    if (std::abs(C) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Erreur DivideMatrixByScalar: Division par zéro ou par un nombre trop petit.");
    }
    
    size_t cols = M[0].size();

    // Créer la matrice de résultat
    Matrix Result(rows, Vector(cols)); 

    // Itération sur tous les éléments
    for (size_t i = 0; i < rows; ++i) {
        if (M[i].size() != cols) {
             throw std::runtime_error("Erreur DivideMatrixByScalar: La matrice n'est pas rectangulaire.");
        }
        for (size_t j = 0; j < cols; ++j) {
            Result[i][j] = M[i][j] / C;
        }
    }
    return Result;
}

// CRPS empirique à partir d'un échantillon prédictif y_samples et d'une observation x_obs.
//
// Formule : CRPS = (1/n) * sum |y_i - x| - (1/n^2) * sum_{i<j} (y_j - y_i)
// (en utilisant que E|Y-Y'| = 2/(n^2) * sum_{i<j} (y_j - y_i) quand l'échantillon est trié)
double crpsFromSamples(const std::vector<double>& y_samples, double x_obs)
{
    const std::size_t n = y_samples.size();
    if (n == 0) {
        throw std::invalid_argument("crps_from_samples: l'échantillon y_samples est vide.");
    }

    // Copie triée de l'échantillon
    std::vector<double> y = y_samples;
    std::sort(y.begin(), y.end());

    // 1) terme (1/n) * sum |y_i - x|
    double sum_abs_y_minus_x = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_abs_y_minus_x += std::fabs(y[i] - x_obs);
    }
    const double term1 = sum_abs_y_minus_x / static_cast<double>(n);

    // 2) terme (1/n^2) * sum_{i<j} (y_j - y_i)
    //
    // On utilise l'identité :
    //   sum_{i<j} (y_j - y_i) = sum_{k=0}^{n-1} (2k - n + 1) * y_k
    // lorsque le vecteur y est trié.
    double sum_weighted = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        double coeff = static_cast<double>(2 * static_cast<long long>(k) - static_cast<long long>(n) + 1);
        sum_weighted += coeff * y[k];
    }

    // E|Y - Y'| = 2/(n^2) * sum_{i<j} (y_j - y_i) = 2/(n^2) * sum_weighted
    const double EYminusYprime = 2.0 * sum_weighted / (static_cast<double>(n) * static_cast<double>(n));

    // CRPS = E|Y - x| - 0.5 * E|Y - Y'|
    const double crps = term1 - 0.5 * EYminusYprime;
    return crps;
}

// Sample of the CRPS 
Vector crpsVector(const Matrix& Y_sample,const Vector& y_obs)
{
    size_t p =y_obs.size();  // number of rows
    size_t n_sample = Y_sample[0].size();  // number of colon 
    Vector column(n_sample);
    Vector tp(p,0);
    for (size_t j=0; j<p;++j)
    {
        for (std::size_t i = 0; i < n_sample; ++i) {
        column[i] = Y_sample[i][j];
     }
        tp[j]=crpsFromSamples(column,y_obs[j]);
    }  
    return tp;
}

// rowMeans 
Vector rowMeans(const Matrix& M) {
    std::size_t p = M.size();
    if (p == 0) return Vector();

    std::size_t n_sample = M[0].size();

    // Vérification que toutes les lignes ont la même taille
    for (const auto& row : M) {
        if (row.size() != n_sample) {
            throw std::invalid_argument("rowMeans: dimensions incohérentes dans la matrice.");
        }
    }

    Vector means(p, 0.0);

    for (std::size_t i = 0; i < p; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n_sample; ++j) {
            sum += M[i][j];
        }
        means[i] = sum / static_cast<double>(n_sample);
    }

    return means;
}


// exportation en fichier .txt

void exportMatrixTxt(const Matrix& M, const std::string& filename) {
    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error("Impossible d'ouvrir le fichier : " + filename);
    }

    for (const auto& row : M) {
        for (std::size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << " ";  // séparateur (espace)
        }
        out << "\n";  // nouvelle ligne pour chaque row
    }

    out.close();
}



void print_vector(const Vector& vec, const std::string& name) {
    
    std::cout << name << " [Taille: " << vec.size() << "] : [" << std::endl;
    
    // Définir le formatage pour tous les éléments suivants
    // std::fixed : Force la notation à point fixe (non scientifique)
    // std::setprecision(6) : Définit le nombre de chiffres après la virgule à 6
    std::cout << std::fixed << std::setprecision(6);
    
    // Affichage des éléments
    for (double element : vec) {
        std::cout << "  " << element;
    }
    
    // Réinitialiser le formatage si nécessaire après l'appel (non obligatoire)
    std::cout << std::defaultfloat << std::setprecision(0); 
    
    std::cout << "\n]" << std::endl;
} 


// exportation sous forme de fichier CSV
void saveMatrixCSV(const Matrix& M, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Impossible d'ouvrir le fichier " + filename);

    for (const auto& row : M) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j + 1 < row.size()) file << ",";  // séparateur CSV
        }
        file << "\n";
    }
}



// exporter un vecteur en TXT
void saveVectorTXT(const std::vector<double>& v, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Impossible d'ouvrir le fichier " + filename);

    for (double val : v) {
        file << val << "\n";   // un élément par ligne
    }
}

//exporter un vecteur en CSV
void saveVectorCSV(const std::vector<double>& v, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Impossible d'ouvrir le fichier " + filename);

    for (size_t i = 0; i < v.size(); ++i) {
        file << v[i];
        if (i + 1 < v.size()) file << ",";  // séparateur CSV
    }

    file << "\n";
}

// Calcule l'écart-type de chaque ligne d'une matrice.
// Retourne un Vector contenant l'écart-type par ligne.
Vector RowsStd(const Matrix& mat) {
    std::size_t nRows = mat.size();
    if (nRows == 0) {
        throw std::invalid_argument("La matrice est vide.");
    }

    Vector stdRows(nRows);

    for (std::size_t i = 0; i < nRows; ++i) {

        std::size_t nCols = mat[i].size();
        if (nCols == 0) {
            throw std::invalid_argument("Une ligne de la matrice est vide.");
        }

        // 1) Moyenne de la ligne
        double sum = 0.0;
        for (std::size_t j = 0; j < nCols; ++j) {
            sum += mat[i][j];
        }
        double mean = sum / nCols;

        // 2) Somme des carrés des écarts
        double sqSum = 0.0;
        for (std::size_t j = 0; j < nCols; ++j) {
            double diff = mat[i][j] - mean;
            sqSum += diff * diff;
        }

        // 3) Ecart-type (population)
        double variance = sqSum / nCols;
        stdRows[i] = std::sqrt(variance/nCols);

        // Pour écart-type échantillon → std::sqrt(sqSum / (nCols - 1))
    }

    return stdRows;
}


Vector VectorDifference(const Vector& v1, const Vector& v2) {
    std::size_t n = v1.size();
    if (n != v2.size()) {
        throw std::invalid_argument("Les vecteurs doivent avoir la même taille.");
    }

    Vector result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = v1[i] - v2[i];
    }

    return result;
}

// calcul de la moyenne d'un vecteur

double mean(const std::vector<double>& v) {
    if (v.empty())
        throw std::invalid_argument("mean: vecteur vide");

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / static_cast<double>(v.size());
}


// linespace sur C++ on retourne un vecteur s

Vector linspace(double a, double b, std::size_t n) {
    if (n == 0) return {};
    if (n == 1) return {a};

    std::vector<double> x(n);
    double step = (b - a) / static_cast<double>(n - 1);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = a + step * static_cast<double>(i);
    }
    return x;
}



// courbe 

// Trace une courbe (x,y) en PNG avec fond gris et courbe bleue
void plotCurveToPNG(const std::vector<double>& x,
                    const std::vector<double>& y,
                    const std::string& filename,
                    const std::string& title,
                    const std::string& xlabel,
                    const std::string& ylabel )
{
    if (x.size() != y.size() || x.size() < 2) {
        throw std::invalid_argument("plotCurveToPNG: x et y doivent avoir la même taille >= 2.");
    }

    // Conversion en PLFLT
    std::vector<PLFLT> xx(x.size()), yy(y.size());
    for (size_t i = 0; i < x.size(); ++i) {
        xx[i] = static_cast<PLFLT>(x[i]);
        yy[i] = static_cast<PLFLT>(y[i]);
    }

    // Bornes automatiques
    auto [xmin_it, xmax_it] = std::minmax_element(x.begin(), x.end());
    auto [ymin_it, ymax_it] = std::minmax_element(y.begin(), y.end());

    double xmin = *xmin_it, xmax = *xmax_it;
    double ymin = *ymin_it, ymax = *ymax_it;

    double padx = 0.05 * (xmax - xmin);
    double pady = 0.05 * (ymax - ymin);
    xmin -= padx; xmax += padx;
    ymin -= pady; ymax += pady;

    plstream pls;

    // --- Driver PNG ---
    pls.sdev("pngcairo");
    pls.sfnam(filename.c_str());

    // --- Palette de couleurs ---
    // Couleur 0 = fond
    pls.scol0(0, 220, 220, 220);   // gris clair

    // Couleur 1 = axes / texte
    pls.scol0(1,   0,   0,   0);   // noir

    // Couleur 2 = courbe
    pls.scol0(2,  30,  90, 200);   // bleu

    pls.init();

    // --- Cadre + axes ---
    pls.col0(1); // noir pour axes et texte
    pls.env(xmin, xmax, ymin, ymax, 0, 0);

    // --- Titre et labels ---
    pls.lab(xlabel.c_str(), ylabel.c_str(), title.c_str());

    // --- Courbe ---
    pls.col0(2); // bleu
    pls.line(static_cast<PLINT>(xx.size()), xx.data(), yy.data());

    pls.eop(); // écrit le fichier PNG
}


// index du maximum

size_t index_of_min(const std::vector<double>& v) {
    if (v.empty())
        throw std::invalid_argument("index_of_max: vecteur vide");

    return std::distance(v.begin(),
                         std::min_element(v.begin(), v.end()));
}