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
// Multiplication de deux matrices
Matrix multiplyMatrices(const Matrix& A, const Matrix& B);
//------------------------------ cholesky decomposition --------------------------//
Matrix cholesky(const Matrix& A);
// ------------ calculation of the scalar product of two vectors ---------
double vectorDotProduct(const Vector& A, const Vector& B);