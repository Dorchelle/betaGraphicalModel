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
#include "usefullFunctions.h"
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
// begin len SS==j

size_t countNonzeroClusterJ(const Vector& SS, int j)
{
    // Nous utilisons une petite tolérance (1e-9) pour comparer le double 'label' à l'entier 'j'.
    size_t count = std::count_if(SS.begin(), SS.end(), 
        [j](double label) {
            // Le label (float) doit être égal à j (converti en float)
            return std::abs(label - (double)j) < 1e-9;
        }
    );
    return count;
}       // len SS==j end

void updateClusterSizeJ(Vector& CS, const Vector& SS, int j) {
    
    if (j < 0 || (size_t)j >= CS.size()) throw std::out_of_range("L'indice de cluster j est hors limites pour le vecteur CS.");
    // Affecter le résultat à CS[j]
    CS[j] = (double)countNonzeroClusterJ(SS,j);
}


  
