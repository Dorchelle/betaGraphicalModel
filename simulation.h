#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <cstring>
#include <cerrno>
#include "fonctions.h"
#include "prior.h"
#include "fullConditional.h"
#include "modeleBetaGaussian.h"
#include "sampling.h" // function for sampling
#include "usefullFunctions.h" // usefull functions
#include "matricesCalculations.h"


void simulationBetaGaussian(int T1, int p, int l,
                             const std::string &chemin_dossier_in,
                             const std::string &chemin_dossier_out);

// Alias pour rester cohérent avec ce que tu utilises
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

Matrix eye(int n, double scale = 1.0);

// Matrice de zéros
Matrix zeros(int rows, int cols);

// Vecteur de zéros
Vector zeros(int n);

// Graphe Erdos–Rényi G(n, p) -> matrice d’adjacence 0/1
Matrix erdosRenyiAdjacency(int n, double prob, std::mt19937_64 &gen);


void createDirIfNotExists(const std::string &path); 