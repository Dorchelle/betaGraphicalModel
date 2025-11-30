#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm> 
#include <iostream>
#include <map>
#include "GraphicalFunctions.h"
// Nécessaire pour la ré-indexation
// Définition de types pour simplifier la lecture
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
PropAjoutRetrait extractNoeud(const Matrix& A_Omega)
{
size_t p = A_Omega.size();
    if (p < 2) {
        throw std::invalid_argument("Erreur Retrait_un_Noeud: La dimension p doit être >= 2.");
    }
    
    // Initialisation
    PropAjoutRetrait result;
    // A_OmegaPrime = A_Omega.copy()
    result.A = A_Omega; 
    
    bool valeur = true;
    int it = 20; // Limite d'itération (comme dans le Python)
    
    // Distribution uniforme pour choisir un indice (0 à p-1)
    std::uniform_int_distribution<> distrib_p(0, (int)p - 1);
    
    // Indices initiaux (mis à -1 pour le cas où aucun retrait n'est trouvé)
    size_t i_0_temp = 0, j_0_temp = 0; 
    
    while (valeur && it > 0) {
        it--;
        
        // Echantillonner i_0 et j_0 (0 à p-1)
        i_0_temp = distrib_p(generator_prior); 
        j_0_temp = distrib_p(generator_prior); 

        // Condition d'acceptation: i_0 != j_0 ET A[i_0, j_0] == 1 (lien existant)
        if (i_0_temp != j_0_temp && std::abs(A_Omega[i_0_temp][j_0_temp] - 1.0) < 1e-9) {
            
            std::cout << "###########Essai reussi!!!!!!!#############" << std::endl;
            
            // Retrait du lien (symétrique)
            result.A[i_0_temp][j_0_temp] = 0.0;
            result.A[j_0_temp][i_0_temp] = 0.0;
            
            // S'assurer que i0 est le plus petit des deux pour le retour (si nécessaire)
            if (j_0_temp < i_0_temp) {
                std::swap(i_0_temp, j_0_temp);
            }
            
            result.i0 = i_0_temp;
            result.j0 = j_0_temp;
            
            valeur = false; // Succès
        }
    }
    
    // --- Logique Python de Recherche Séquentielle (après échec du tirage aléatoire) ---
    if (valeur) {
        // Le code Python utilise une boucle while(valeur) avec une logique d'arrêt peu claire
        // et probablement erronée: while (j_0<p and A_OmegaPrime[i_0,j_0]!=0): j_0-=1
        
        // Nous implémentons la recherche séquentielle pour garantir un retrait (méthode la plus simple)
        bool found_sequential = false;
        for (size_t i = 0; i < p; ++i) {
            for (size_t j = i + 1; j < p; ++j) {
                // Recherche du premier lien existant (A[i,j] == 1)
                if (std::abs(A_Omega[i][j] - 1.0) < 1e-9) {
                    result.A[i][j] = 0.0;
                    result.A[j][i] = 0.0;
                    result.i0 = i;
                    result.j0 = j;
                    found_sequential = true;
                    break;
                }
            }
            if (found_sequential) break;
        }

        if (!found_sequential) {
            // Si le graphe est déjà déconnecté (pas de liens hors diagonale)
            // Nous retournons simplement la matrice originale et des indices invalides ou le dernier échantillonné.
            // Pour le MCMC, cela peut être un signe que le graphe a atteint l'état souhaité.
             result.i0 = 0;
             result.j0 = 0;
             result.A = A_Omega; // Retourne l'original sans modification
        }
    }

    return result;

}