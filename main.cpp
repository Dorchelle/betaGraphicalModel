
// main.cpp

// -----------include the packages--------------------
#include <iostream>
#include <iomanip>  // Pour formater l'affichage des doubles
#include <stdexcept> // Pour gérer les exceptions (erreurs)
// -----------include libraries--------------------
#include "usefullFunctions.h"   // some usefull functions
#include "sampling.h"   // some usefull functions
#include "fonctions.h" // Incluez votre fichier d'en-tête
#include "prior.h"   // prior distribustion
#include "fullConditional.h"   // prior distribustion
using namespace std;
// definition des variables 
DonneesChargees donnees;
double x;

// Ajoutez d'autres fonctions de test ici pour XXt, Inverse, logLikhood, etc.

int main() {
    cout << "========================================" << endl;
    cout << "         DÉBUT DES TESTS C++            " << endl;
    cout << "========================================" << endl;

    // Appel de la fonction
   x=logBetaPDF(0.5,2.0,3.0);
   x= multigammaln(6, 5);
    
    // Affichage du résultat (Décommenté)
    std::cout << "Valeur de logBetaPDF(0.5, 2.0, 3.0) : " << std::fixed << std::setprecision(4) << x << std::endl;
    //std::cout << "Valeur de logBetaPDF(0.5, 2.0, 3.0) : " << std::fixed << std::setprecision(4) << x << std::endl;
    donnees = chargesDonnees(std::string("Data/covariableDataMexicoStatesETLL.csv"),std::string("Data/IDHMEXICO.CSV"),true);
    std::cout << "Valeur : " << std::fixed << std::setprecision(2) << donnees.T1 << std::endl;
    
    cout << "----------------------------------------" << endl;
    // ...
    cout << "========================================" << endl;
    cout << "          FIN DES TESTS C++             " << endl;
    cout << "========================================" << endl;
    cout << "=============== bonjour bonjour ===========" << endl;
    return 0;
}
