



// fonctions.cpp


#include <stdexcept>
#include <vector>
#include <iostream>
#include "matricesCalculations.h"
// ... (Définitions des autres fonctions) ...

/**
 * Calcule le produit matriciel A * B.
 * Le nombre de colonnes de A doit être égal au nombre de lignes de B.
 */
Matrix multiplyMatrices(const Matrix& A, const Matrix& B) {
    size_t rows_A = A.size();
    if (rows_A == 0) return Matrix();
    size_t cols_A = A[0].size();
    
    size_t rows_B = B.size();
    if (rows_B == 0) return Matrix();
    size_t cols_B = B[0].size();
    
    // Vérification de la compatibilité des dimensions
    if (cols_A != rows_B) {
        throw std::invalid_argument("Erreur multiplyMatrices: Les dimensions ne sont pas compatibles (cols_A != rows_B).");
    }
    
    // La matrice résultante C sera de taille rows_A x cols_B
    Matrix C(rows_A, Vector(cols_B, 0.0));
    
    // Algorithme de multiplication matricielle (C[i][j] = Sum_k (A[i][k] * B[k][j]))
    for (size_t i = 0; i < rows_A; ++i) { // Lignes de C (et A)
        for (size_t j = 0; j < cols_B; ++j) { // Colonnes de C (et B)
            double sum = 0.0;
            for (size_t k = 0; k < cols_A; ++k) { // Somme sur la dimension intérieure
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

//------------------------------ cholesky decomposition --------------------------//
Matrix cholesky(const Matrix& A) {
    size_t n = A.size();
    if (n == 0) return Matrix();
    if (n != A[0].size()) {
        throw std::invalid_argument("Erreur cholesky: La matrice A doit être carrée.");
    }
    
    // Création d'une matrice L (triangulaire inférieure) initiale à zéro
    Matrix L(n, Vector(n, 0.0));

    // Calcul de la matrice L
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;

            if (j == i) { 
                // Diagonale: L[j, j] = sqrt(A[j, j] - sum(L[j, k]^2, k=0..j-1))
                for (size_t k = 0; k < j; ++k) {
                    sum += std::pow(L[j][k], 2.0); // L[j, k] ** 2
                }
                
                double value = A[j][j] - sum;
                
                if (value <= 1e-12) {
                    // La matrice n'est pas définie positive (ou singulière), cholesky échoue
                    throw std::runtime_error("Erreur cholesky: La matrice n'est pas définie positive (racine d'un nombre négatif).");
                }
                L[j][j] = std::sqrt(value);
            } else { 
                // Termes non diagonaux: L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k], k=0..j-1)) / L[j, j]
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                
                // Vérification de la division par zéro (bien que cela implique A[j,j]=0 plus haut)
                if (std::abs(L[j][j]) < 1e-18) {
                     throw std::runtime_error("Erreur cholesky: Division par zéro (L[j][j] est nul).");
                }
                
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    
    return L;
}

// ------------ calculation of the scalar product of two vectors ---------
double vectorDotProduct(const Vector& A, const Vector& B) {
    
    // Vérification de la compatibilité des dimensions
    if (A.size() != B.size()) {
        throw std::invalid_argument("Erreur vector_dot_product: Les vecteurs doivent avoir la même taille.");
    }
    
    double result = 0.0;
    
    // Calcul du produit scalaire (somme des produits des éléments correspondants)
    for (size_t i = 0; i < A.size(); ++i) {
        result += A[i] * B[i];
    }
    
    return result;
}