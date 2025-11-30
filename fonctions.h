// fonctions.h
#ifndef FONCTIONS_H
#define FONCTIONS_H
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <utility> // Pour std::pair
#include <algorithm>
#include <map> // Nécessaire pour la ré-indexation
// Définition de types pour simplifier la lecture
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;


// Structure pour contenir les résultats de chargesDonnees
struct DonneesChargees {
    Matrix X;
    int l;
    Matrix IDH;
    Vector IDHPre;
    int T1;
};

/*
* Structure de retour pour K-Means (équivalent à la sortie de f.kmeansSS)
 */
struct KmeansResult {
    Vector SS;       // Labels de cluster (p * T1)
    Vector CS;       // Tailles des clusters (CS)
    size_t Ncl;      // Nombre de clusters final (Ncl)
    Matrix ma;       // Matrice d'adjacence (ma)
};


// Structure de retour hypothétique pour la pollution (ajustez si nécessaire)
struct DonneesPollution {
    Matrix PollutionData; // Les données de pollution (par exemple, P x T)
    int p;                // Nombre de régions/stations (P)
    int T;                // Nombre de périodes de temps (T)
};


struct MatricesPrecision {
    Matrix Omega; // Matrice de Précision (Omega = Psi^T * Psi)
    Matrix Sigma; // Matrice de Covariance (Sigma = Omega^-1)
};


struct PropAjoutRetrait {
    Matrix A;
    size_t i0;
    size_t j0;
};


// Structure pour stocker les résultats MCMC
// Structure de retour pour la fonction MCMC
struct MCMC_Result_Full {
    Matrix Psi;
    Matrix T;
    Matrix B;
    Matrix Sigma;
    double detB;
    Matrix A_Theta;
};


// Déclaration de la nouvelle fonction
DonneesPollution chargesDonneesPollution(
    const std::string& chemin_fichier_pollution, 
    bool has_header = true, 
    char separator = ';');
// --- Fonctions Algèbre Linéaire et Utilitaires ---
double euclideanDistanceSq(const Vector& a, const Vector& b);
// Remplace XXt(X) : Calcule X * X^T (Matrice carrée pour un vecteur colonne)
// Note: Le Python calcule la matrice M[i,:] = X[i]*X, ce qui est X * X^T pour un vecteur X de taille l.
Matrix XXt(const Vector& X);
KmeansResult kmeansSS(const Matrix& X, int T1, int K_init);
// Remplace factoriel(n)
long  factoriel(int n);

// Remplace trace(M)
double trace(const Matrix& M);
Vector calculateCs(const Vector& SS);
// Calcule le determinant d'une matrice (l'implémentation complète nécessite des bibliothèques comme Eigen)
// Déclaration simple pour le moment
double det(const Matrix& M);

// Remplace Inverse(Phi)
Matrix Inverse(const Matrix& Phi);

// --- Fonctions Utilitaires Spécifiques ---

// Remplace logLikhood(X,ALPHA,BETA) - nécessite l'implémentation de la fonction Beta PDF.
// Nous allons utiliser std::lgamma pour calculer log(gamma(x))
double logBetaPDF(double x, double alpha, double beta);
double logLikhood(const Matrix& X, const Matrix& ALPHA, const Matrix& BETA);
double logUniformRvs();
Matrix EchantillonPsiMod(Matrix& Psi, Matrix& A, const Matrix& T, double b, size_t i1, size_t j1, bool Etat, double tp = 0.0);

// Remplace secondesVersHMS
std::tuple<int, int, int> secondesVersHMS(long long secondes);
// Déclaration de la fonction chargesDonnees
DonneesChargees chargesDonnees(const std::string& chemin_fichier_covariable,
                                const std::string& chemin_fichier_VaRep,
                                bool State = true);
 // Permet de calculer somme de =s xi t(xi) sur S_j 
Matrix covJ(const std::vector<int>& S, const Matrix& X, int j, int l);
                                // ... (Autres déclarations de fonctions)   
Matrix covJPoll(const Vector& S, const Matrix& IDH, int j, const Vector& Cs, int l);                                                             
// fonctions.h

// ... (Inclusions et définitions de types)
// Remplace mubarj : Calcule le vecteur moyen pour les éléments du cluster j
Vector mubarj(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l);
double multigammaln(double a, int p);
// Remplace numerateur : Calcule le numérateur du ratio de Bayes (log-échelle)
double numerateur(const Vector& Cs, int j,double b_j, int l);
// Remplace denominateur
double denominateur(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l, double b_j);
// Déclaration nécessaire si vous n'avez pas encore implémenté le déterminant
double det(const Matrix& M);
double denominateurPoll(const Vector& S, const Matrix& Data, int j, const Vector& Cs, int l, double b_j);
Vector truncatedInvgammaSample(double alpha, double beta, double a, double b, size_t size);
Vector truncatedNormalSample(double mu, double sigma, double a, double b, size_t size);
double betaFunction(double x, double y);

Matrix EchantillonPsi(const Matrix& A, const Matrix& T, double b);
Matrix echantillonPsiMod(Matrix& Psi, Matrix& A, const Matrix& T, double b, size_t i1, size_t j1, bool Etat, double tp = 0.0);
Matrix Inverse(const Matrix& Phi);
// Transposition de matrice
Matrix Transpose(const Matrix& M);
MatricesPrecision precisionMatrix(const Matrix& Psi);
double constanteNormalisation(const Matrix& A, double b, const Matrix& T, size_t count);
Matrix structureMatrice(const Matrix& Omega);
Matrix retirerUnNoeud(const Matrix& Omega);
PropAjoutRetrait ajoutUnNoeud(const Matrix& aOmega);
Matrix S_epsi(const Matrix& theta, Vector xi);
double logLikhoodBetaJ(const Matrix& X, const Matrix& ALPHA, const Matrix& BETA, const Vector& S, int j);
double poissonGammaLog(int y, double alpha, double beta);
double logLikhoodPoissonGammaJ(const Matrix& X, const Matrix& ALPHA, const Matrix& BETA, const Vector& S, int j);
double logLikhood(const Matrix& X, const Matrix& ALPHA, const Matrix& BETA);
Matrix extractMatrix(const Matrix& B, int j, const Vector& S, int Nj);
Vector extractVect(const Vector& U, int j, const Vector& S, int Nj);
std::pair<int, int> isValueInMatrix(const Vector& SS, int i, int t, int p);
Vector meanSS(const Vector& S, const Matrix& X, int j, const Vector& Cs, int l);
// Fonction utilitaire nécessaire pour l'étape MH
Matrix AddMatrices(const Matrix& A, const Matrix& B);
Matrix DivideMatrixByScalar(const Matrix& M, double C);
// Fonctions utilitaires matricielles nécessaires pour le MCMC
Matrix SubtractMatrices(const Matrix& A, const Matrix& B); // A - B
double SumAbsMatrix(const Matrix& M); // np.sum(np.abs(M))
MCMC_Result_Full MCMC(double lambda_0, Matrix& B_in, Matrix& Sigma, double detB_init, Matrix& Psi_init, Matrix& T_init, double b, const Matrix& S_epsi, double neta, Matrix& A_Theta, int T_periods);
// ... (Autres déclarations de fonctions)
#endif // FONCTIONS_H