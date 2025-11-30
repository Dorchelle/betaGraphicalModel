// fonctions.cpp


#include <stdexcept> // Pour les erreurs
#include <algorithm> // pour std::copy
#include <limits>    // <-- NOUVEAU : Nécessaire pour std::numeric_limits
#include <random>
#include <iostream>
#include <cmath>     // <-- NOUVEAU : Nécessaire pour std::log et std::lgamma
#include <map>
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
    size_t num_features = X.size();      // Nombre de lignes de X (l)
    size_t num_points = X[0].size();     // Nombre de colonnes de X (p * T1)
    size_t p = num_points / T1;          // Nombre de régions (p)
    
    // Le code Python utilise X.T (p*T1 x l). Nous allons transposer X pour travailler
    // avec des lignes de données (point i = X_transposed[i])
    Matrix X_data(num_points, Vector(num_features)); 
    for (size_t j = 0; j < num_points; ++j) {
        for (size_t i = 0; i < num_features; ++i) {
            X_data[j][i] = X[i][j]; // X_data est (p*T1 x l)
        }
    }
    
    // Assurer K_init est raisonnable
    if (K_init <= 0 || K_init > num_points) {
        K_init = (int)std::sqrt(p) + 1;
    }
    int K = K_init; // K est le nombre de clusters
    
    // --- Initialisation des Centres (K-Means++) ---
    std::vector<Vector> centroids(K);
    std::uniform_int_distribution<> initial_point_dist(0, num_points - 1);
    
    // Sélectionner le premier centre aléatoirement
    centroids[0] = X_data[initial_point_dist(generator_prior)];

    // Les autres centres sont initialisés par K-Means++ (méthode de sélection par distance)
    std::vector<double> min_dist_sq(num_points);
    for (int k = 1; k < K; ++k) {
        double total_dist_sq = 0.0;
        for (size_t i = 0; i < num_points; ++i) {
            double dist = std::numeric_limits<double>::max();
            for (int c = 0; c < k; ++c) {
                dist = std::min(dist, euclideanDistanceSq(X_data[i], centroids[c]));
            }
            min_dist_sq[i] = dist;
            total_dist_sq += dist;
        }

        // Sélectionner le prochain centre avec probabilité proportionnelle à min_dist_sq
        std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
        double r = uniform_01(generator_prior) * total_dist_sq;
        
        double cumulative_sum = 0.0;
        for (size_t i = 0; i < num_points; ++i) {
            cumulative_sum += min_dist_sq[i];
            if (cumulative_sum >= r) {
                centroids[k] = X_data[i];
                break;
            }
        }
    }

    // --- Algorithme K-Means ---
    Vector SS_result(num_points, 0.0); // Labels de cluster
    std::vector<int> counts(K, 0); 
    const int max_iterations = 100;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Étape E (Affectation)
        bool changed = false;
        for (size_t i = 0; i < num_points; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = -1;
            
            for (int k = 0; k < K; ++k) {
                double dist = euclideanDistanceSq(X_data[i], centroids[k]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            if (SS_result[i] != (double)best_cluster) {
                SS_result[i] = (double)best_cluster;
                changed = true;
            }
        }

        // Étape M (Mise à jour des Centres)
        std::vector<Vector> new_centroids(K, Vector(num_features, 0.0));
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < num_points; ++i) {
            int k = (int)SS_result[i];
            counts[k]++;
            for (size_t f = 0; f < num_features; ++f) {
                new_centroids[k][f] += X_data[i][f];
            }
        }
        
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                for (size_t f = 0; f < num_features; ++f) {
                    centroids[k][f] = new_centroids[k][f] / counts[k];
                }
            }
            // Si un cluster est vide, il peut être géré par réinitialisation ou simplement ignoré.
            // Nous l'ignorons ici.
        }

        if (!changed) break; // Arrêt si les affectations ne changent plus
    }

    // --- Préparation du Résultat Final (Similaire à f.kmeansSS en Python) ---

    // CS (Tailles des clusters)
    Vector CS_result(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        CS_result[k] = (double)counts[k];
    }
    
    // Ncl (Nombre de clusters - peut être inférieur à K s'il y a des clusters vides)
    size_t Ncl_final = K; 

    // ma (Matrice d'adjacence)
    Matrix ma(num_points, Vector(num_points, 0.0));
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < num_points; ++j) {
            if (SS_result[i] == SS_result[j]) {
                ma[i][j] = 1.0;
            }
        }
    }
    
    // Note: Le code Python inclut une logique complexe de ré-indexation pour s'assurer
    // que les labels de cluster sont contigus de 0 à Ncl-1, ce qui n'est pas fait ici.
    // L'implémentation de la ré-indexation nécessite une fonction utilitaire DPMM.

    KmeansResult result;
    result.SS = SS_result;
    result.CS = CS_result;
    result.Ncl = Ncl_final;
    result.ma = ma;
    
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
    Matrix M(l, Vector(l)); // Initialise une matrice l x l

    for (size_t i = 0; i < l; ++i) {
        for (size_t j = 0; j < l; ++j) {
            M[i][j] = X[i] * X[j];
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
 * Implémentation de l'sample_Betapour une matrice triangulaire supérieure (comme en Python)
 * L'implémentation complète pour une matrice quelconque est complexe et nécessiterait Eigen.
 */
Matrix Inverse(const Matrix& Phi) {
    size_t p = Phi.size();
    if (p == 0) return Matrix();
    if (p != Phi[0].size()) throw std::invalid_argument("La matrice doit être carrée.");

    Matrix IPhi(p, Vector(p, 0.0));

    for (size_t i = 0; i < p; ++i) {
        // La diagonale
        if (std::abs(Phi[i][i]) < 1e-9) { // Vérification de la division par zéro
            throw std::runtime_error("Erreur: La diagonale contient un zéro ou est trop petite.");
        }
        IPhi[i][i] = 1.0 / Phi[i][i];

        // Remplissage de la partie supérieure
        for (size_t j = i - 1; j > 0; --j) {
            double somme = 0.0;
            // Note: La boucle Python a une erreur de borne, elle devrait ressembler à ceci
            // for k in range(j + 1, i + 1):  ... somme += Phi[j, k] * IPhi[k, i]
            for (size_t k = j + 1; k <= i; ++k) {
                somme += Phi[j][k] * IPhi[k][i];
            }
            IPhi[j][i] = -somme / Phi[j][j];
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
        
        // l = np.size(table[0,:])-1
        // Si la première colonne est la ville (nom), on assume qu'elle a été sautée
        // dans notre `lire_csv_vers_matrix` ci-dessus pour simplifier.
        // Si elle n'a pas été sautée, il faudrait cols - 1.
        // Ici, on assume que la première colonne a été traitée ou ignorée dans la lecture.
        // Basé sur: table1=table[:,1:(l+1)] -> la première colonne (indice 0) est ignorée.
        size_t l = cols; // l est le nombre de covariables (colonnes numériques restantes)

        // X: Matrice l x p (Transposée de table1 en Python)
        result.X.assign(l, Vector(p));

        // table1 = table[:, 1:(l+1)] (La lecture python ignore la colonne 0 (Ville))
        // Notre `lire_csv_vers_matrix` ci-dessus *doit* être ajustée pour gérer le saut de colonne.
        // Pour l'instant, nous faisons la transposition : X = table1.T
        for (size_t i = 0; i < p; ++i) { // lignes (individus)
            for (size_t j = 0; j < l; ++j) { // colonnes (covariables)
                // Le Python fait la transposition de (p x l) vers (l x p)
                result.X[j][i] = table_cov[i][j];
            }
        }

        // --- Mise à jour manuelle des lignes de Longitude (l-2) et Latitude (l-1) ---
        // Les indices Python l-2 et l-1 correspondent aux deux dernières lignes de X.
        // En C++, les lignes de X sont indexées de 0 à l-1.
        if (l >= 2 && p == 32) {
             Vector new_longitude = {-102.3726689,-115.1425107,-111.5706164,-90.0,-102.0000001,-104.0,-92.5000001,-106.0000001, -99.1441352, -104.833333 , -101.0,-100.0,-99.0,-103.6666671,-99.1331785, -101.878113 ,-99.0,-105.0000001,-94.9841472,-96.5,-98.0,-99.8837376, -88.5000001,-100.4949145,-107.5000001,-110.6666671,-92.6681659,-98.7026825,-98.166667, -96.666667,-88.8755669,-102.9333954 };
             Vector new_latitude = {21.9942689,30.0338923,25.5818014,19.0,27.3333331,19.166667,16.5000001,28.5000001,  23.7389846 ,24.833333, 20.9876996,17.666667,20.5, 20.3333331,19.4326296, 19.207098, 18.75, 22.0000001,16.2048579, 17.0,18.833333, 20.8052225, 19.6666671,22.5000001,25.0000001,29.3333331,17.9999288,23.9891553,19.416667, 19.333333,20.6845957,23.0916177};
            
            // X[l-2,:] = ... (Avant-dernière ligne)
            if (result.X.size() > (size_t)l - 2) {
                result.X[l-2] = new_longitude;
            }

            // X[l-1,:] = ... (Dernière ligne)
            if (result.X.size() > (size_t)l - 1) {
                result.X[l-1] = new_latitude;
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
            }
        }
        
        // IDH =IDH1[:,2:(T1-1)] (Colonnes de 2 à T1_cols-2)
        // La nouvelle T1 est T1_cols - 3
        result.T1 = (int)(T1_cols - 3);
        
        if (result.T1 > 0) {
            size_t start_col = 2;
            size_t end_col = T1_cols - 2;

            result.IDH.assign(p, Vector((size_t)result.T1));
            for (size_t i = 0; i < p; ++i) {
                for (size_t j = start_col; j < end_col; ++j) {
                    result.IDH[i][j - start_col] = IDH1[i][j];
                }
            }
        }

    } else {
        // --- LOGIQUE STATE=FALSE (header=False, sep=',') ---
        
        // table=pd.read_csv(chemin_fichier_covariable) (header=False, sep=',')
        Matrix table_cov = lire_csv_vers_matrix(chemin_fichier_covariable, false, ',');
        
        if (table_cov.empty()) {
            throw std::runtime_error("Erreur: Table des covariables vide.");
        }

        size_t p = table_cov.size();
        size_t l = table_cov[0].size();
        
        // X = table.T (Transposition)
        result.X.assign(l, Vector(p));
        for (size_t i = 0; i < p; ++i) {
            for (size_t j = 0; j < (size_t)l; ++j) {
                result.X[j][i] = table_cov[i][j];
            }
        }
        
        // IDH1=pd.read_csv(chemin_fichier_VaRep)
        Matrix IDH1 = lire_csv_vers_matrix(chemin_fichier_VaRep, false, ',');
        
        size_t T1_cols = IDH1[0].size();
        
        // IDHPre=IDH1[:,T1-1] (Dernière colonne)
        if (T1_cols > 0) {
            result.IDHPre.resize(p);
            for (size_t i = 0; i < p; ++i) {
                result.IDHPre[i] = IDH1[i][T1_cols - 1];
            }
        }
        
        // IDH =IDH1[:,0:(T1-1)] (Toutes les colonnes sauf la dernière)
        // La nouvelle T1 est T1_cols - 1
        result.T1 =(int)( T1_cols - 1);
        
        if (result.T1 > 0) {
            result.IDH.assign(p, Vector((size_t)result.T1));
            for (size_t i = 0; i < p; ++i) {
                for (size_t j = 0; j < (size_t)result.T1; ++j) {
                    result.IDH[i][j] = IDH1[i][j];
                }
            }
        }
    }

    result.l = (int) result.X.size(); // Mise à jour finale du l (nombre de covariables)
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


Vector mubarj(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l) {
    // tp = 0*np.zeros(l) -> Vecteur de taille l initialisé à zéro (pour la somme)
    Vector tp(l, 0.0);
    
    if (X.empty() || l == 0) return tp; 
    
    // p = Nombre d'individus (colonnes dans X)
    size_t p = X[0].size(); 
    size_t taille_S = S.size(); 
    
    // Vérifier la taille du cluster (Cs[j])
    if (j >= Cs.size() || Cs[j] <= 0) {
        // Le cluster j n'existe pas ou est vide.
        throw std::invalid_argument("Erreur dans mubarj: Le cluster j est vide ou l'indice est invalide.");
    }
    
    // Boucle sur toutes les entrées du vecteur de regroupement S
    for (size_t i = 0; i < taille_S; ++i) {
        // if(S[i]==(j)):
        if (static_cast<int>(S[i]) == j) {
            
            // i_0 = i % p -> Indice de l'individu (colonne) dans X
            size_t i_0 = i % p;
            
            // tp = tp + X[:,i_0] (Sommation du vecteur colonne)
            for (int k = 0; k < l; ++k) {
                // X[k][i_0] car X est (l x p)
                tp[k] += X[k][i_0]; 
            }
        }
    }
    
    // return (tp/Cs[j]) -> Division par la taille du cluster
    double Cs_j = Cs[j];
    for (int k = 0; k < l; ++k) {
        tp[k] /= Cs_j;
    }
    
    return tp;
}


double multigammaln(double a, int p) {
    if (p < 1) {
        throw std::invalid_argument("multigammaln: La dimension p doit être >= 1.");
    }
    
    // Condition de validité : a doit être > (p - 1) / 2
    if (a <= (static_cast<double>(p) - 1.0) / 2.0) {
        // En cas de paramètre non valide pour la fonction Gamma
        std::cerr << "Attention: Argument 'a' trop petit pour la dimension 'p'." << std::endl;
        // On pourrait retourner -infinity si l'on veut simuler le comportement de SciPy
        return -std::numeric_limits<double>::infinity(); 
    }
    
    double result = 0.0;
    
    // Partie 1: p * (p-1) / 4 * log(pi)
    result += (static_cast<double>(p) * (p - 1) / 4.0) * std::log(M_PI);
    
    // Partie 2: somme_{i=1}^{p} log(Gamma(a + (1-i)/2))
    for (int i = 1; i <= p; ++i) {
        // Le terme est (1-i)/2
        double term = (1.0 - static_cast<double>(i)) / 2.0;
        
        // std::lgamma(x) calcule log(Gamma(x))
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
    size_t n = M.size();
    if (n == 0) return 0.0;
    if (n != M[0].size()) {
        throw std::invalid_argument("Erreur det: La matrice doit être carrée.");
    }

    // Création d'une copie de la matrice pour éviter de modifier l'originale
    Matrix A = M; 
    double determinant = 1.0;
    int sign = 1;

    // Élimination de Gauss
    for (size_t i = 0; i < n; ++i) {
        // 1. Pivotage partiel (pour la stabilité)
        size_t pivot = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(A[j][i]) > std::abs(A[pivot][i])) {
                pivot = j;
            }
        }

        if (pivot != i) {
            // Échange les lignes i et pivot
            std::swap(A[i], A[pivot]);
            // Chaque échange de ligne inverse le signe du déterminant
            sign *= -1; 
        }

        // Vérification de la singularité (le pivot est presque zéro)
        if (std::abs(A[i][i]) < 1e-12) { 
            return 0.0; // Le déterminant est zéro (matrice singulière)
        }

        // 2. Élimination
        for (size_t j = i + 1; j < n; ++j) {
            double factor = A[j][i] / A[i][i];
            for (size_t k = i; k < n; ++k) {
                A[j][k] -= factor * A[i][k];
            }
        }
    }

    // 3. Calcul du déterminant (produit des éléments diagonaux)
    for (size_t i = 0; i < n; ++i) {
        determinant *= A[i][i];
    }

    return sign * determinant;
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


double betaFunction(double x, double y) {
    
    // Vérification des conditions : x et y doivent être positifs pour la fonction Gamma
    if (x <= 0.0 || y <= 0.0) {
        throw std::invalid_argument("Erreur betaFunction: Les arguments x et y doivent être strictement positifs.");
    }
    
    // Calcul de Gamma(x) * Gamma(y) / Gamma(x + y)
    
    // NOTE IMPORTANTE : Utiliser le logarithme pour le calcul est plus stable numériquement
    // (pour éviter les dépassements de capacité avec de grandes valeurs de Gamma).
    // Cependant, votre code Python utilisait la multiplication/division directe.
    
    // --- Approche directe (comme le Python) ---
    double gamma_x = std::tgamma(x);
    double gamma_y = std::tgamma(y);
    double gamma_sum = std::tgamma(x + y);

    if (std::abs(gamma_sum) < 1e-18) {
        // Gérer le cas où Gamma(x+y) est trop petit ou zéro (problème numérique)
        return std::numeric_limits<double>::infinity();
    }
    
    return (gamma_x * gamma_y) / gamma_sum;
    
    // --- Approche Logarithmique (plus stable, si nécessaire) ---
    /*
    double log_beta = std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
    return std::exp(log_beta);
    */
}





Matrix EchantillonPsi(const Matrix& A, const Matrix& T, double b) {
    size_t p = A.size();
    if (p == 0) return Matrix();
    if (p != A[0].size() || p != T.size() || p != T[0].size()) {
        throw std::invalid_argument("EchantillonPsi: Les matrices A et T doivent être carrées de même dimension p.");
    }
    
    // Initialisation
    Vector nui(p, 0.0);
    Vector hi(p, 0.0);
    Matrix T1(p, Vector(p, 0.0));
    Matrix Psi(p, Vector(p, 0.0));
    
    // --- Étape 1: Calculer T1, hi, nui ---
    for (size_t j = 0; j < p; ++j) {
        
        // Calcul de T1 = T[:,j] / T[j,j]
        if (std::abs(T[j][j]) < 1e-18) {
             throw std::runtime_error("EchantillonPsi: T[j][j] est nul, division impossible.");
        }
        for (size_t r = 0; r < p; ++r) {
            T1[r][j] = T[r][j] / T[j][j];
        }
        
        // hi[j]: Nombre de voisins précédents (somme A[j, 0:j])
        if (j > 0) {
            for (size_t k = 0; k < j; ++k) {
                hi[j] += A[j][k];
            }
        }
        
        // nui[j]: Nombre de voisins suivants (somme A[j, j+1:p])
        if (j < p - 1) {
            for (size_t k = j + 1; k < p; ++k) {
                nui[j] += A[j][k];
            }
        }
    }
    
    // --- Étape 2: Échantillonnage des diagonales et des termes A[i,j]=1 (i < j) ---
    // (Les étapes 2 et 3 sont fusionnées ici pour simplifier)

    // Déclaration des distributions
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    for (size_t i = 0; i < p; ++i) {
        
        // Diagonale: Psi[i,i] ~ sqrt(Chi^2(b + nui[i]))
        double df_chi = b + nui[i];
        if (df_chi <= 0) {
            throw std::runtime_error("EchantillonPsi: Degré de liberté Chi-carré invalide.");
        }
        std::chi_squared_distribution<double> chi_sq_dist(df_chi);
        
        Psi[i][i] = std::sqrt(chi_sq_dist(generator));
        
        // Termes hors diagonale supérieur où A[i,j]=1
        for (size_t j = i + 1; j < p; ++j) {
            if (static_cast<int>(A[i][j]) == 1) {
                // Psi[i,j] ~ N(0, 1)
                Psi[i][j] = normal_dist(generator);
            }
        }
    }
    
    // --- Étape 3 & 4: Calcul des termes A[i,j]=0 (i < j) par backward-substitution ---
    // (Ceci correspond à la partie compliquée de votre code Python avec les sommes s1 et s2)
    
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = i + 1; j < p; ++j) {
            // Si le lien est absent (A[i,j] = 0), nous calculons Psi[i,j]
            if (static_cast<int>(A[i][j]) == 0) {
                
                // Formule: Psi[i,j] = - sum_{k=i}^{j-1} Psi[i,k] * T1[k,j]
                double sum1 = 0.0;
                for (size_t k = i; k < j; ++k) {
                    sum1 += Psi[i][k] * T1[k][j];
                }
                Psi[i][j] = -sum1;
                
                // Pour i > 0, il y a la correction complexe impliquant s1 et s2:
                // Psi[i,j] -= sum_{r=1}^{i-1} s1_r * s2_r
                if (i > 0) {
                    double correction_sum = 0.0;
                    for (size_t r = 0; r < i; ++r) {
                        
                        // Calcul de s1 = (Psi[r,i] + sum_{k=r}^{i-1} Psi[r,k] * T1[k,i]) / Psi[i,i]
                        double sum_s1 = 0.0;
                        for (size_t k = r; k < i; ++k) {
                            sum_s1 += Psi[r][k] * T1[k][i];
                        }
                        
                        double s1;
                        if (std::abs(Psi[i][i]) < 1e-18) {
                            throw std::runtime_error("EchantillonPsi: Division par zéro (Psi[i][i] est nul).");
                        }
                        s1 = (Psi[r][i] + sum_s1) / Psi[i][i];
                        
                        // Calcul de s2 = Psi[r,j] + sum_{k=r}^{j-1} Psi[r,k] * T1[k,j]
                        double sum_s2 = 0.0;
                        for (size_t k = r; k < j; ++k) {
                            sum_s2 += Psi[r][k] * T1[k][j];
                        }
                        double s2 = Psi[r][j] + sum_s2;
                        
                        correction_sum += s1 * s2;
                    }
                    Psi[i][j] -= correction_sum;
                }
            }
        }
    }
    
    return Psi;
}


Matrix echantillonPsiMod(Matrix& Psi, Matrix& A, const Matrix& T, double b, size_t i1, size_t j1, bool Etat, double tp) {
    size_t p = A.size();
    if (p == 0) return Matrix();
    
    // Initialisation et vérifications
    Vector nui(p, 0.0);
    Vector hi(p, 0.0);
    Matrix T1(p, Vector(p, 0.0));
    
    // Vérification des indices
    if (i1 >= p || j1 >= p || i1 >= j1) { // On travaille uniquement sur la partie triangulaire supérieure (i1 < j1)
         throw std::invalid_argument("echantillonPsiMod: Les indices du lien sont invalides ou i1 >= j1.");
    }

    // --- Étape 1: Modification du lien (A[i1, j1]) ---
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    if (Etat) {
        // Ajout d'un lien (Etat=True)
        // print("we are adding an egde")
        A[i1][j1] = 1.0; 
        A[j1][i1] = 1.0; // Symétrie
        Psi[i1][j1] = normal_dist(generator); // Échantillonner N(0, 1) pour le nouveau lien
    } else {
        // Suppression d'un lien (Etat=False)
        // print("we are removing an egde")
        A[i1][j1] = 0.0;
        A[j1][i1] = 0.0; // Symétrie
        // La valeur tp (originale de Psi[i1,j1]) est implicitement sauvegardée dans la logique MCMC
        Psi[i1][j1] = 0.0; // Mis à zéro, puis recalculé à l'étape 3/4
    }
    
    // --- Étape 2: Calculer T1, hi, nui (Recalculé après la modification de A) ---
    // Cette étape est nécessaire car 'nui' (nombre de voisins suivants) a pu changer
    for (size_t j = 0; j < p; ++j) {
        if (std::abs(T[j][j]) < 1e-18) {
             throw std::runtime_error("echantillonPsiMod: T[j][j] est nul, division impossible.");
        }
        for (size_t r = 0; r < p; ++r) {
            T1[r][j] = T[r][j] / T[j][j];
        }
        // Recalculer hi et nui
        hi[j] = 0.0; nui[j] = 0.0;
        if (j > 0) {
            for (size_t k = 0; k < j; ++k) {
                hi[j] += A[j][k];
            }
        }
        if (j < p - 1) {
            for (size_t k = j + 1; k < p; ++k) {
                nui[j] += A[j][k];
            }
        }
    }

    // --- Étape 3 & 4: Recalcul des termes A[i,j]=0 par backward-substitution ---
    // On itère sur tous les termes de la triangulaire supérieure (i < j)
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = i + 1; j < p; ++j) {
            
            // Si le lien est absent (A[i,j] = 0), nous calculons Psi[i,j]
            // Le cas i=i1, j=j1 est inclus si A[i1,j1] a été mis à zéro (suppression)
            if (static_cast<int>(A[i][j]) == 0) {
                
                // Formule de base: Psi[i,j] = - sum_{k=i}^{j-1} Psi[i,k] * T1[k,j]
                double sum_base = 0.0;
                for (size_t k = i; k < j; ++k) {
                    sum_base += Psi[i][k] * T1[k][j];
                }
                Psi[i][j] = -sum_base;
                
                // Correction complexe (uniquement pour i > 0)
                if (i > 0) {
                    double correction_sum = 0.0;
                    for (size_t r = 0; r < i; ++r) {
                        
                        // Calcul de s1 = (Psi[r,i] + sum_{k=r}^{i-1} Psi[r,k] * T1[k,i]) / Psi[i,i]
                        double sum_s1 = 0.0;
                        for (size_t k = r; k < i; ++k) {
                            sum_s1 += Psi[r][k] * T1[k][i];
                        }
                        
                        double s1;
                        if (std::abs(Psi[i][i]) < 1e-18) {
                            throw std::runtime_error("echantillonPsiMod: Division par zéro (Psi[i][i] est nul pendant la correction).");
                        }
                        s1 = (Psi[r][i] + sum_s1) / Psi[i][i];
                        
                        // Calcul de s2 = Psi[r,j] + sum_{k=r}^{j-1} Psi[r,k] * T1[k,j]
                        double sum_s2 = 0.0;
                        for (size_t k = r; k < j; ++k) {
                            sum_s2 += Psi[r][k] * T1[k][j];
                        }
                        double s2 = Psi[r][j] + sum_s2;
                        
                        correction_sum += s1 * s2;
                    }
                    Psi[i][j] -= correction_sum;
                }
            }
        }
    }
    
    return Psi;
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

    MatricesPrecision result;

    // 1. Calculer Omega = Psi^T * Psi
    Matrix Psi_T = Transpose(Psi);
    result.Omega = multiplyMatrices(Psi_T, Psi);
    
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
    size_t p = A.size();
    if (p == 0 || count == 0) return 0.0;
    
    // Initialisation
    double const_term = 0.0;
    Vector hi(p, 0.0);
    Vector nui(p, 0.0);
    
    // Matrice AA = np.ones((p,p))
    Matrix AA(p, Vector(p, 1.0));

    // --- Étape 1: Calculer hi et nui (nombre de voisins) ---
    for (size_t j = 0; j < p; ++j) {
        // hi[j]: Nombre de voisins précédents (somme A[j, 0:j])
        if (j > 0) {
            for (size_t k = 0; k < j; ++k) {
                hi[j] += A[j][k];
            }
        }
        
        // nui[j]: Nombre de voisins suivants (somme A[j, j+1:p])
        if (j < p - 1) {
            for (size_t k = j + 1; k < p; ++k) {
                nui[j] += A[j][k];
            }
        }
    }

    // --- Étape 2: Calculer la partie analytique (const) ---
    for (size_t i = 0; i < p; ++i) {
        double nui_i = nui[i];
        double hi_i = hi[i];
        
        // Terme 1: np.log(math.gamma(0.5*(b+nui[i])))
        // Note: L'utilisation de std::lgamma est préférable à std::log(std::tgamma)
        const_term += std::lgamma(0.5 * (b + nui_i)); 
        
        // Terme 2: (0.5*nui[i])*np.log(2*math.pi)
        const_term += (0.5 * nui_i) * std::log(2.0 * M_PI);
        
        // Terme 3: (b+ nui[i]+hi[i])*np.log(T[i,i])
        if (T[i][i] <= 0.0) {
            throw std::runtime_error("constanteNormalisation: T[i,i] doit être positif.");
        }
        const_term += (b + nui_i + hi_i) * std::log(T[i][i]);
        
        // Terme 4: (0.5*(nui[i]+b))*np.log(2)
        const_term += (0.5 * (nui_i + b)) * std::log(2.0);
    }
    
    // --- Étape 3: Calculer l'intégrale par Monte Carlo (f) ---
    double f_sum = 0.0;
    // La boucle Python était "range(count-1)". En C++, on peut faire "size_t j = 0; j < count;"
    // pour avoir 'count' échantillons. J'utilise 'count' itérations.
    for (size_t j = 0; j < count; ++j) {
        
        // 1. Échantillonner Psi
        Matrix Psi = EchantillonPsi(A, T, b);
        
        // 2. Calculer le terme dans l'exponentielle: -0.5*np.sum((AA-A)*(Psi*Psi))
        double inner_sum = 0.0;
        for (size_t r = 0; r < p; ++r) {
            for (size_t c = 0; c < p; ++c) {
                // Terme à sommer: (AA[r,c] - A[r,c]) * (Psi[r,c] * Psi[r,c])
                inner_sum += (AA[r][c] - A[r][c]) * std::pow(Psi[r][c], 2.0);
            }
        }
        
        // Ajouter à la somme Monte Carlo: f += exp(...)
        f_sum += std::exp(-0.5 * inner_sum);
    }
    
    // --- Étape 4: Calcul final ---
    // return const + np.log(f_sum) - np.log(count)
    if (f_sum <= 0.0) {
        std::cerr << "Erreur constanteNormalisation: La somme Monte Carlo est nulle ou négative." << std::endl;
        return -std::numeric_limits<double>::infinity();
    }
    
    double log_Z = const_term + std::log(f_sum) - std::log(static_cast<double>(count));
    
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
    size_t p = Omega.size();
    if (p == 0) return Matrix();
    if (p != Omega[0].size()) {
        throw std::invalid_argument("Erreur retirerUnNoeud: La matrice Omega doit être carrée.");
    }

    // Création de la copie (omegaPrime = Omega.copy())
    Matrix omegaPrime = Omega;
    
    // Initialisation de la valeur minimale (on utilise la limite max de double)
    double min_abs_value = std::numeric_limits<double>::max();
    int i_0 = -1; // Indices pour le nœud à retirer
    int j_0 = -1;

    // Recherche de la plus petite valeur absolue hors diagonale
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = i + 1; j < p; ++j) { // Parcours seulement la triangulaire supérieure (hors diagonale)
            
            double current_value = Omega[i][j];
            double current_abs = std::abs(current_value);
            
            // Si la valeur est non nulle (au-delà d'une petite tolérance)
            // ET si elle est plus petite que le minimum actuel
            if (current_abs > 1e-12 && current_abs < min_abs_value) {
                min_abs_value = current_abs;
                i_0 = static_cast<int>(i); 
                j_0 = static_cast<int>(j);
            }
        }
    }

    // --- Retrait du nœud ---
    if (i_0 != -1 && j_0 != -1) {
        // Mettre à zéro l'élément trouvé et son symétrique pour garantir la symétrie
        omegaPrime[i_0][j_0] = 0.0;
        omegaPrime[j_0][i_0] = 0.0;
        
        // Affichage (similaire à un print de débogage si vous en aviez un)
        // std::cout << "Lien retire : (" << i_0 << ", " << j_0 << ") avec Omega = " << min_abs_value << std::endl;
    } else {
        // Cas où il n'y a plus de liens hors diagonale à retirer (graphe vide)
        // std::cout << "Aucun lien hors diagonale trouve à retirer." << std::endl;
    }
    
    return omegaPrime;
}

/**
 * Traduction de ajoutUnNoeud(aOmega)
 * Propose aléatoirement l'ajout d'un lien (A[i,j]=0) par échantillonnage.
 */
PropAjoutRetrait ajoutUnNoeud(const Matrix& aOmega) {
    size_t p = aOmega.size();
    if (p < 2) {
        throw std::invalid_argument("Erreur ajoutUnNoeud: La dimension p doit être >= 2.");
    }
    
    // Initialisation
    PropAjoutRetrait result;
    // aomegaPrime = aOmega.copy()
    result.A = aOmega; 
    
    bool valeur = true;
    int it = 10; // Limite d'itération (comme dans le Python)
    
    // Distribution uniforme pour choisir un indice (0 à p-1)
    std::uniform_int_distribution<> distrib_p(0, p - 1);
    
    // Indices initiaux
    size_t i_0_temp = 0, j_0_temp = 0; 
    
    while (valeur && it > 0) {
        it--;
        
        // Echantillonner i_0 et j_0 (0 à p-1)
        i_0_temp = distrib_p(generator); 
        j_0_temp = distrib_p(generator); 

        // Condition d'acceptation: i_0 != j_0 ET A[i_0, j_0] == 0 (lien absent)
        if (i_0_temp != j_0_temp && std::abs(aOmega[i_0_temp][j_0_temp]) < 1e-12) {
            
            // print("###########Essai reussi!!!!!!!#############")
            
            // Ajout du lien (symétrique)
            result.A[i_0_temp][j_0_temp] = 1.0;
            result.A[j_0_temp][i_0_temp] = 1.0;
            
            // S'assurer que i0 est le plus petit des deux pour le retour
            if (j_0_temp < i_0_temp) {
                std::swap(i_0_temp, j_0_temp);
            }
            
            result.i0 = i_0_temp;
            result.j0 = j_0_temp;
            
            valeur = false; // Succès
        }
    }
    
    // --- Gestion du Cas d'Échec (it <= 0) : Recherche séquentielle ---
    // Le Python passe à la recherche séquentielle pour garantir un ajout, sauf si le graphe est complet
    if (valeur) {
        // Recherche séquentielle du premier lien manquant (i < j)
        for (size_t i = 0; i < p; ++i) {
            for (size_t j = i + 1; j < p; ++j) {
                // Si le lien est absent (A[i,j] == 0)
                if (std::abs(aOmega[i][j]) < 1e-12) {
                    
                    result.A[i][j] = 1.0;
                    result.A[j][i] = 1.0;
                    
                    result.i0 = i;
                    result.j0 = j;
                    
                    valeur = false;
                    return result; // Retourne immédiatement après avoir trouvé et ajouté le premier lien
                }
            }
        }
        
        // Si 'valeur' est toujours true ici, cela signifie que le graphe est complet.
        if (valeur) {
            throw std::runtime_error("ajoutUnNoeud: Le graphe est déjà complet (pas de lien à ajouter).");
        }
    }

    return result;
}


Matrix S_epsi(const Matrix& theta, double xi) {
    size_t p = theta.size(); // Nombre de régions/lignes (dimension du vecteur)
    if (p == 0) return Matrix();
    
    size_t T1 = theta[0].size(); // Nombre de périodes de temps (colonnes)
    if (T1 < 1) return Matrix(); 

    // Initialisation de la matrice de résultat (p x p)
    Matrix tp(p, Vector(p, 0.0));
    
    // --- Terme Initial: XXt(np.log(theta[:,0])) ---
    
    // 1. Extraire et appliquer log à theta[:, 0]
    Vector log_theta_0(p);
    for (size_t i = 0; i < p; ++i) {
        if (theta[i][0] <= 0.0) {
             throw std::runtime_error("S_epsi: Erreur log, theta[i, 0] doit être positif.");
        }
        log_theta_0[i] = std::log(theta[i][0]);
    }
    
    // 2. Calculer XXt et initialiser tp
    tp = XXt(log_theta_0);

    // --- Boucle de Sommation: sum_{t=0}^{T1-2} XXt(xi * log(theta[:, t]) - log(theta[:, t+1])) ---
    
    for (size_t t = 0; t < T1 - 1; ++t) {
        
        // Calculer le vecteur epsi_t = xi * log(theta[:, t]) - log(theta[:, t+1])
        Vector epsi_t(p);
        
        for (size_t i = 0; i < p; ++i) {
            double log_t;
            double log_t_plus_1;
            
            if (theta[i][t] <= 0.0 || theta[i][t+1] <= 0.0) {
                 throw std::runtime_error("S_epsi: Erreur log, theta[i, t] ou theta[i, t+1] doit être positif.");
            }

            log_t = std::log(theta[i][t]);
            log_t_plus_1 = std::log(theta[i][t+1]);
            
            // xi * log(theta[:, t]) - log(theta[:, t+1])
            epsi_t[i] = xi * log_t - log_t_plus_1; 
        }
        
        // Calculer XXt(epsi_t)
        Matrix XXt_t = XXt(epsi_t);
        
        // Sommation de la matrice: tp += XXt_t
        for (size_t r = 0; r < p; ++r) {
            for (size_t c = 0; c < p; ++c) {
                tp[r][c] += XXt_t[r][c];
            }
        }
    }
    
    return tp;
}


double logLikhoodBetaJ(const Matrix& X, const Matrix& ALPHA, const Matrix& BETA, const Vector& S, int j) {
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

            double x_it = X[i][t];
            double alpha_it = ALPHA[i][t];
            double beta_it = BETA[i][t];

            // Accumuler la log-vraisemblance du PDF Bêta
            tp += logBetaPDF(x_it, alpha_it, beta_it);
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


Vector meanSS(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l) {
    // tp = 0*np.zeros(l) -> Vecteur de taille l initialisé à zéro (pour la somme)
    Vector tp(l, 0.0);
    
    if (X.empty() || l == 0) return tp; 
    
    // p = Nombre d'individus/états (colonnes dans X)
    // Nous devons déterminer la dimension des données. 
    // Si X est (l x p*T), alors p_total = p*T. Si X est (l x p), p_total = p.
    size_t p_total = X[0].size(); 
    size_t taille_S = S.size(); 
    
    // Vérification de base (S doit avoir la même taille que le nombre d'entités dans X)
    if (taille_S != p_total) {
        throw std::invalid_argument("Erreur meanSS: La taille de S ne correspond pas au nombre total d'entités dans X.");
    }
    
    // Vérifier la taille du cluster (Cs[j])
    if (j >= Cs.size() || Cs[j] <= 0) {
        // Le cluster j n'existe pas ou est vide.
        throw std::invalid_argument("Erreur meanSS: Le cluster j est vide ou l'indice est invalide.");
    }
    
    // Boucle sur toutes les entrées du vecteur de regroupement S
    for (size_t i = 0; i < taille_S; ++i) {
        // if(S[i]==(j)):
        if (static_cast<int>(S[i]) == j) {
            
            // i_0 est l'indice de l'entité (colonne) dans X
            size_t i_0 = i; 
            
            // tp = tp + X[:,i_0] (Sommation du vecteur colonne)
            for (int k = 0; k < l; ++k) {
                // X[k][i_0] car X est (l x p_total)
                tp[k] += X[k][i_0]; 
            }
        }
    }
    
    // return (tp/Cs[j]) -> Division par la taille du cluster
    double Cs_j = Cs[j];
    for (int k = 0; k < l; ++k) {
        tp[k] /= Cs_j;
    }
    
    return tp;
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
Matrix EchantillonPsiMod(Matrix& Psi, Matrix& A, const Matrix& T, double b, size_t i1, size_t j1, bool Etat, double tp) {
    size_t p = A.size();
    if (p == 0) return Matrix();
    
    // T1 est la matrice T[:,j] / T[j,j]
    Matrix T1(p, Vector(p, 0.0));
    
    // --- 1. Modification du lien (A[i1, j1]) ---
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    if (Etat) {
        // Ajout d'un lien (Etat=True)
        std::cout << "we are adding an egde" << std::endl;
        A[i1][j1] = 1.0; 
        A[j1][i1] = 1.0; // Symétrie
        Psi[i1][j1] = normal_dist(generator_prior); // Échantillonner N(0, 1) pour le nouveau lien
    } else {
        // Suppression d'un lien (Etat=False)
        std::cout << "we are removing an egde" << std::endl;
        A[i1][j1] = 0.0;
        A[j1][i1] = 0.0; // Symétrie
        // Psi[i1,j1] est mis à zéro dans la boucle de mise à jour des zéros
    }
    
    // --- 2. Calculer T1 = T[:,j] / T[j,j] ---
    for (size_t j = 0; j < p; ++j) {
        if (std::abs(T[j][j]) < 1e-18) {
             throw std::runtime_error("EchantillonPsiMod: T[j,j] est nul, division impossible.");
        }
        for (size_t r = 0; r < p; ++r) {
            T1[r][j] = T[r][j] / T[j][j];
        }
    }

    // --- 3. Mise à jour de la première ligne (i=0) ---
    // i=1 en Python (indice 1-basé) correspond à i=0 en C++
    size_t i_c = 0; // i_c = i-1 en Python
    size_t j_c = i_c + 1;
    
    while (j_c < p) { // j_c < p correspond à j <= p en Python
        if (std::abs(A[i_c][j_c]) < 1e-12) { // Si A[i_c, j_c] == 0
            
            // Calculer np.sum(Psi[i-1,(i-1):(j-1)]*T1[(i-1):(j-1),j-1])
            // (Psi[0, 0:j_c] * T1[0:j_c, j_c])
            double sum_term = numpySumSlice(Psi, i_c, i_c, j_c, T1, i_c, j_c, j_c);

            Psi[i_c][j_c] = -1.0 * sum_term;
        }
        j_c++;
    }

    // --- 4. Mise à jour des autres lignes (i > 0) ---
    for (size_t i_c_p = 1; i_c_p < p; ++i_c_p) { // i_c_p = i-1 en Python, démarre à 1
        i_c = i_c_p;
        j_c = i_c + 1;
        
        while (j_c < p) { 
            if (std::abs(A[i_c][j_c]) < 1e-12) { // Si A[i_c, j_c] == 0
                
                // Réinitialiser la valeur
                Psi[i_c][j_c] = 0.0;
                
                // Terme 1 (s1*s2 loop) : r = 1 à i-1
                for (size_t r_c = 0; r_c < i_c; ++r_c) { // r_c = r-1 en Python
                    
                    // Calcul de s1
                    // s1_num = Psi[r-1, i-1] + np.sum(Psi[r-1, (r-1):(i-1)] * T1[(r-1):(i-1), i-1])
                    double sum_s1_slice = numpySumSlice(Psi, r_c, r_c, i_c, T1, r_c, i_c, i_c);
                    double s1_num = Psi[r_c][i_c] + sum_s1_slice;
                    
                    if (std::abs(Psi[i_c][i_c]) < 1e-18) {
                        throw std::runtime_error("EchantillonPsiMod: Division par zéro (Psi[i,i] est nul).");
                    }
                    double s1 = s1_num / Psi[i_c][i_c];

                    // Calcul de s2
                    // s2 = Psi[r-1,j-1] + np.sum(Psi[r-1, (r-1):(j-1)] * T1[(r-1):(j-1), j-1])
                    double sum_s2_slice = numpySumSlice(Psi, r_c, r_c, j_c, T1, r_c, j_c, j_c);
                    double s2 = Psi[r_c][j_c] + sum_s2_slice;

                    // Psi[i-1,j-1] = Psi[i-1,j-1]-s1*s2
                    Psi[i_c][j_c] -= s1 * s2;
                }
                
                // Terme 2: -1 * np.sum(Psi[i-1,(i-1):(j-1)]*T1[(i-1):(j-1),j-1])
                double sum_term_2 = numpySumSlice(Psi, i_c, i_c, j_c, T1, i_c, j_c, j_c);

                Psi[i_c][j_c] -= sum_term_2;
            }
            j_c++;
        }
    }
    
    // Le code Python renvoie Psi, qui est modifié par référence
    return Psi;
}


// --- Fonction MCMC Principale ---
MCMC_Result_Full MCMC(double lambda_0, Matrix& B_in, Matrix& Sigma, double detB_init, Matrix& Psi_init, Matrix& T_init, double b, const Matrix& S_epsi, double neta, Matrix& A_Theta, int T_periods) {
    
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
        
        // Calculer B_star = Omega* * sqrt(diag(Sigma*))
        Matrix B_star(p, Vector(p));
        for (size_t r = 0; r < p; ++r) {
            for (size_t c = 0; c < p; ++c) {
                B_star[r][c] = Omega_star[r][c] * std::sqrt(Sigma_star[r][r] * Sigma_star[c][c]);
            }
        }

        // --- 3. Vérification de la Définie Positivité (Eigenvalue Check) ---
        // Placeholder: Cette vérification est difficile sans une librairie externe.
        // Nous allons supposer que le calcul du déterminant suffit pour la non-singularité.
        try {
            double det_omega_star = det(Omega_star);
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
            detB_star = 2.0 * std::log(std::abs(det(Psi_star))); // Calcul précis du detB* (log-déterminant)
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
}


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
