
// main.cpp

// -----------include the packages--------------------
#include <chrono>
#include <iostream>
#include <iomanip>  // Pour formater l'affichage des doubles
#include <stdexcept> // Pour gérer les exceptions (erreurs)
#include <vector>
// -----------include libraries--------------------
#include "usefullFunctions.h"   // some usefull functions
#include "sampling.h"   // some usefull functions
#include "fonctions.h" // Incluez votre fichier d'en-tête
#include "prior.h"   // prior distribustion
#include "fullConditional.h"   // prior distribustion
#include "modeleBetaGaussian.h"
#include "simulation.h"
using namespace std;
using Vector = std::vector<double>; // Votre alias de type
using namespace std::chrono;
// definition des variables 
DonneesChargees donnees;
KmeansResult kmeans;
HyperParameters hp;
BetaGaussienResult result;

double x;





/**
 * Affiche le contenu d'un vecteur à l'écran avec une précision de 6 chiffres après la virgule.
 */
/**/
/*void print_vector(const Vector& vec, const std::string& name = "Vecteur") {
    
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
} */

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
    std::cout << "Valeur  de periode: " << std::fixed << std::setprecision(2) << donnees.T1 << std::endl;
    kmeans=kmeansSS(donnees.X,donnees.T1,20);   // we start with 20 clusters
    size_t taille=(kmeans.SS).size();
    //print_vector(kmeans.ma[0]);
    size_t n_sample = 1000;
    size_t burn_in = 1000;
    bool State = true;
    HyperParameters defaults;
    defaults.b_mu=0.25;
    defaults.a_sigma_1=0.5;
    defaults.a_sigma_2=0.5;
    defaults.b_sigma_1=2;
    defaults.b_sigma_2=2;
    defaults.lambda_0=0.05;
    defaults.b_alpha=0.005;
    BetaGaussienResult result;
    //std::cout << "simulation T petit p petit " << std::fixed << std::setprecision(2) <<T1 << std::endl;
    int T1=5;
    int p=10;
    int l=5;
    std::cout << "simulation T petit p petit " << std::fixed << std::setprecision(2) << T1 <<"p=="<<p<< std::endl;

    /*simulationBetaGaussian(T1, p, l,"DonneesT5P10/simulation1","ResultatsT5P10/simulation1"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P10/simulation2","ResultatsT5P10/simulation2"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P10/simulation3","ResultatsT5P10/simulation3"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P10/simulation4","ResultatsT5P10/simulation4"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P10/simulation5","ResultatsT5P10/simulation5"); 


    T1=5;p=30;l=5;
    std::cout << "simulation T petit p grand " << std::fixed << std::setprecision(2) << T1 <<"p=="<<p<< std::endl;

    simulationBetaGaussian(T1, p, l,"DonneesT5P30/simulation1","ResultatsT5P30/simulation1"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P30/simulation2","ResultatsT5P30/simulation2"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P30/simulation3","ResultatsT5P30/simulation3"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P30/simulation4","ResultatsT5P30/simulation4"); 
    simulationBetaGaussian(T1, p, l,"DonneesT5P30/simulation5","ResultatsT5P30/simulation5"); 

     T1=30;  p=5; l=5;
    std::cout << "simulation T grand p petit " << std::fixed << std::setprecision(2) << T1 <<"p=="<<p<< std::endl;

    simulationBetaGaussian(T1, p, l,"DonneesT30P5/simulation1","ResultatsT30P5/simulation1"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P5/simulation2","ResultatsT30P5/simulation2"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P5/simulation3","ResultatsT30P5/simulation3"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P5/simulation4","ResultatsT30P5/simulation4"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P5/simulation5","ResultatsT30P5/simulation5"); 

    std::cout << "simulation T grand p grand " << std::fixed << std::setprecision(2) <<T1 <<"p=="<<p<< std::endl;


     T1=30;p=20;l=5;
    simulationBetaGaussian(T1, p, l,"DonneesT30P20/simulation1","ResultatsT30P20/simulation1"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P20/simulation2","ResultatsT30P20/simulation2"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P20/simulation3","ResultatsT30P20/simulation3"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P20/simulation4","ResultatsT30P20/simulation4"); 
    simulationBetaGaussian(T1, p, l,"DonneesT30P20/simulation5","ResultatsT30P20/simulation5");*/ 
    //print_vector(kmeans.CS);
    // ######### Resultats
    size_t count=20;   // vector size
    // partition of b_mu
     Vector b=linspace(0.1,1,count);
     Vector crpsCourbe(count,0.0);
     Vector crps;
    double tp=defaults.b_mu;

    for (size_t i = 0; i < count; i++)
    {
        defaults.b_mu=b[i];
        result= betaGaussien(std::string("Data/covariableDataMexicoStatesETLL.csv"),std::string("Data/IDHMEXICO.CSV"),burn_in,n_sample,defaults,true);
        crps=crpsVector(result.crps_sample,result.IDHPre);
        crpsCourbe[i]=mean(crps);    
    }
    defaults.b_mu=tp;
    plotCurveToPNG(b, crpsCourbe,
        "result/crpsMeanParameter.png",
        "Mean parameter vs CRPS",
        " mu parameter",
        "CRPS");
    size_t i =index_of_min(crpsCourbe);
    double min=crpsCourbe[i];
    defaults.b_mu=b[i];
     // for the alpha parameter
    tp=defaults.b_alpha;
    b=linspace(0.1,10,count);
    for (size_t i = 0; i < count; i++)
    {
        defaults.b_alpha=b[i];
        result= betaGaussien(std::string("Data/covariableDataMexicoStatesETLL.csv"),std::string("Data/IDHMEXICO.CSV"),burn_in,n_sample,defaults,true);
        crps=crpsVector(result.crps_sample,result.IDHPre);
        crpsCourbe[i]=mean(crps);    
    }
     plotCurveToPNG(b, crpsCourbe,
        "result/crpsAlphaParameter.png",
        "alpha parameter vs CRPS",
        " alpha parameter",
        "CRPS");
    i =index_of_min(crpsCourbe);
    defaults.b_alpha=tp;
    if (min>crpsCourbe[i] )
    {
        defaults.b_alpha=b[i];
        min=crpsCourbe[i];
    }
    
    
    // sigma 1 parameters
    tp=defaults.a_sigma_1;
    b=linspace(0.1,5,count);
    for (size_t i = 0; i < count; i++)
    {
        defaults.a_sigma_1=b[i];
        defaults.b_sigma_1=b[i]+2;
        result= betaGaussien(std::string("Data/covariableDataMexicoStatesETLL.csv"),std::string("Data/IDHMEXICO.CSV"),burn_in,n_sample,defaults,true);
        crps=crpsVector(result.crps_sample,result.IDHPre);
        crpsCourbe[i]=mean(crps);    
    }
    defaults.a_sigma_1=tp;
    defaults.b_sigma_1=2;
    plotCurveToPNG(b, crpsCourbe,
        "result/crpsSigma1Parameter.png",
        "Sigma 1 parameter vs CRPS",
        " sigma1 parameter",
        "CRPS");
    i =index_of_min(crpsCourbe);
     if (min>crpsCourbe[i] )
    {
        defaults.a_sigma_1=b[i];
        defaults.b_sigma_1=b[i]+2;
        min=crpsCourbe[i];
    }
    
// sigma 2 parameter
    tp=defaults.a_sigma_2;
     for (size_t i = 0; i < count; i++)
    {
        defaults.a_sigma_2=b[i];
        defaults.b_sigma_2=b[i]+2;
        result= betaGaussien(std::string("Data/covariableDataMexicoStatesETLL.csv"),std::string("Data/IDHMEXICO.CSV"),burn_in,n_sample,defaults,true);
        crps=crpsVector(result.crps_sample,result.IDHPre);
        crpsCourbe[i]=mean(crps);    
    }
    defaults.a_sigma_2=tp;
    defaults.b_sigma_2=2;
    plotCurveToPNG(b, crpsCourbe,
        "result/crpsSigma2Parameter.png",
        "Sigma 2 parameter vs CRPS",
        " sigma2 parameter",
        "CRPS");
    i =index_of_min(crpsCourbe);
     if (min>crpsCourbe[i] )
    {
        defaults.a_sigma_1=b[i];
        defaults.b_sigma_1=b[i]+2;
        min=crpsCourbe[i];
    }

    count =5;
    // for xi parameter
    // sigma 2 parameter
    tp=defaults.a_xi;
    Vector b1=linspace(-6,-1,count);
    Vector crpsCourbe1(count,0.0);
    Vector crps11;
    for (size_t i = 0; i < count; i++)
    {
        defaults.a_xi=b1[i];
        result= betaGaussien(std::string("Data/covariableDataMexicoStatesETLL.csv"),std::string("Data/IDHMEXICO.CSV"),burn_in,n_sample,defaults,true);
        crps11=crpsVector(result.crps_sample,result.IDHPre);
        crpsCourbe1[i]=mean(crps11);    
    }
    defaults.a_xi=tp;
    plotCurveToPNG(b1, crpsCourbe1,
        "result/crpsXiParameter.png",
        "Xi parameter vs CRPS",
        " Xi parameter",
        "CRPS");
    i =index_of_min(crpsCourbe1);
     if (min>crpsCourbe1[i] )
    {
        defaults.a_xi=b1[i];
        min=crpsCourbe[i];
    }

    // ----------------------SAVING------------------------------//


    auto start = high_resolution_clock::now();
   /* defaults.b_mu=0.30;
    defaults.a_sigma_1=0.5;  
    defaults.a_sigma_2=0.5;  
    defaults.b_sigma_1=2;  
    defaults.b_sigma_2=2;  
    defaults.lambda_0=0.05;
    defaults.b_alpha=0.25;*/
    result= betaGaussien(std::string("Data/covariableDataMexicoStatesETLL.csv"),std::string("Data/IDHMEXICO.CSV"),burn_in,n_sample,defaults,true);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Durée = " << duration.count() << " ms" << std::endl;
    // print_vector(result.mu_save);
   Vector means =rowMeans(result.mu_save);
   Vector crps1=crpsVector(result.crps_sample,result.IDHPre);
   // exportation des resultats
   print_vector(means);
   print_vector(result.IDHPre);
   Vector std=RowsStd(result.mu_save);
   print_vector(VectorDifference(means,result.IDHPre));
   print_vector(std);
   print_vector(crps1);
   saveMatrixCSV(result.crps_sample,"result/crpsSample.csv");   // saving of the crps Sample
   saveMatrixCSV(result.mu_save,"result/muSave.csv");    // saving of the proposal mu sample
   saveVectorCSV(means,"result/predict.csv");     // saving of the means prediction
   saveVectorCSV(VectorDifference(means,result.IDHPre),"result/diffForeObs.csv");   // saving of the difference
   saveVectorCSV(crps1,"result/crps.csv");    // crps proposal values
   saveVectorCSV(std,"result/standardDeviation.csv");  // standard deviation  /*  */
   // ########### IDH AFRIQUE ##################
    /*auto start = high_resolution_clock::now();
    State=false;
    result= betaGaussien(std::string("Donnees_IDH_model_Dorchelle/covariables_IDH_Afrique.csv"),std::string("Donnees_IDH_model_Dorchelle//IDH_Afrique.CSV"));
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Durée = " << duration.count() << " ms" << std::endl;
    // print_vector(result.mu_save);
   Vector means =rowMeans(result.mu_save);
   Vector crps=crpsVector(result.crps_sample,result.IDHPre);
   // exportation des resultats
   print_vector(means);
   print_vector(result.IDHPre);
   Vector std=RowsStd(result.mu_save);
   print_vector(VectorDifference(means,result.IDHPre));
   print_vector(std);
   print_vector(crps);
   saveMatrixCSV(result.crps_sample,"ResultatIDHAfrique/crpsSample.csv");   // saving of the crps Sample
   saveMatrixCSV(result.mu_save,"ResultatIDHAfrique/muSave.csv");    // saving of the proposal mu sample
   saveVectorCSV(means,"ResultatIDHAfrique/predict.csv");     // saving of the means prediction
   saveVectorCSV(VectorDifference(means,result.IDHPre),"ResultatIDHAfrique/diffForeObs.csv");   // saving of the difference
   saveVectorCSV(crps,"ResultatIDHAfrique/crps.csv");    // crps proposal values
   saveVectorCSV(std,"ResultatIDHAfrique/standardDeviation.csv");  // standard deviation  */
  
   

    
    //print_vector();
    std::cout <<" le nombre de:"<< std::fixed <<std::setprecision(4)<< donnees.l <<std::endl;
    cout << "----------------------------------------" << endl;
    //cout <<donnees.Ville[0] << endl;
    // ...
    cout << "========================================" << endl;
    cout << "          FIN DES TESTS C++             " << endl;
    cout << "========================================" << endl;
    cout << "=============== bonjour bonjour ===========" << endl;
    return 0;
}
