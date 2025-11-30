
long factoriel(int n) {
    if (n < 0) {
        throw std::invalid_argument("Le nombre ne peut pas être négatif.");
    }
    if (n == 0 || n == 1) {
        return 1;
    }
    long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}





// --- Fonctions Utilitaires Spécifiques ---

/**
 * Calcule le logarithme de la densité de probabilité de la loi Beta.
 * log(Beta_pdf(x | alpha, beta)) = (alpha - 1)log(x) + (beta - 1)log(1 - x) - log(Beta(alpha, beta))
 * log(Beta(alpha, beta)) = log(Gamma(alpha)) + log(Gamma(beta)) - log(Gamma(alpha + beta))
 */
double logBetaPDF(double x, double alpha, double beta) {
    // Vérification des conditions
    if (x <= 0.0 || x >= 1.0 || alpha <= 0.0 || beta <= 0.0) {
        return -std::numeric_limits<double>::infinity(); // Retourne un log(0)
    }

    // Le C++ standard utilise std::lgamma pour log(Gamma(x))
    double log_beta_func = std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);

    return (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log(1.0 - x) - log_beta_func;
}

