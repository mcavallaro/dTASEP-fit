#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "type_defs.h"
#include "update.h"
#include "print.h"

void usage(char *argv[]){
  printf("*******************************************************\n"
         "**                                                   **\n"
         "**         -*-         dWASEP        -*-             **\n"
         "**       \\p_0 p_1 p_2 p_3 p_4 p_5 p_6 \\p_N         **\n"
         "**                                                   **\n"
         "*******************************************************\n"
         "\n\n"
         " This is Free Software - You can use and distribute it under \n"
         " the terms of the GNU General Public License, version 3 or later\n\n" 
         " (c) Massimo Cavallaro (m.cavallaro@warwick.ac.uk) \n\n"
         );
  printf("Usage: %s [thermalization time] [simulation time] [dim ensemble] [input rate profile] [# rates (length of input)] [L] [output prefix]\n\n" , argv[0]);
}

int main(int argc, char *argv[]) {

    if (argc != 8){
        usage(argv);
        exit(EXIT_FAILURE);
    }

    int n_of_rates;
    int dim_chain;
    double * rates;
//     double * q_rates;
    double R, a, L;
    double therm_t, simul_t;

    FILE* densities_file;
    FILE* rates_file;

    int i; //span the ensamble
    int j; //span the nodes
    double t;

    int DIM_ENSEMBLE;
    int tmp;
    int status = 0;
    gsl_rng * r;

    therm_t = atof(argv[1]);  // > 100
    simul_t = atof(argv[2]);  // 300
    DIM_ENSEMBLE = atoi(argv[3]);
    
    n_of_rates = atoi(argv[5]);
    L = atof(argv[6]); // 100
    
    dim_chain = n_of_rates - 1;
    a = L / dim_chain;
    rates = new double [n_of_rates];
    
// //     double mean_rate = 0;
//     // define the vector q for leftwards particle jumping 
//     q_rates = new double [n_of_rates];

    char densities_file_name[120];

    snprintf(densities_file_name, 120, "%sprofiles_time=%s_time=%s_ensem=%s_N=%d_L=%s.txt", argv[7], argv[1], argv[2], argv[3], dim_chain, argv[6]);
    densities_file = fopen(densities_file_name, "w");

    rates_file = fopen(argv[4], "r");
    if (rates_file == NULL) {
        perror("Error while opening the file.\n");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < n_of_rates; i++){
        status = fscanf(rates_file, "%lf\n", &rates[i]);
        rates[i] = rates[i] / a; // Apply Euler scaling
    }
    fclose(rates_file);
    
//     // the system is weakly driven, q_rates is only half of the mean_rate
//     for (i = 0; i < n_of_rates; i++){
//         // q_rates[i] = mean_rate / 10;
//         q_rates[i] = 0;
//     }

    r = gsl_rng_alloc(gsl_rng_taus);

    vector<Systems> system(DIM_ENSEMBLE);

    // initialise
    for (i=0; i<DIM_ENSEMBLE; i++) {
        system[i].node.reserve(dim_chain);
        system[i].node.resize(dim_chain);
        system[i].N=0;
        for (j=0; j<dim_chain; j++){
            system[i].node[j] = gsl_ran_bernoulli(r, 0.5);
            if (system[i].node[j] > 0){
                system[i].N = system[i].N + 1;
            }
        }
        system[i].part_sum.reserve(n_of_rates);
        system[i].part_sum.resize(n_of_rates);
        system[i].part_sum.assign(n_of_rates, 0);
        system[i].t=0;
    }

    for (i =0; i< DIM_ENSEMBLE; i++) {
        status = assign_part_sum(&system[0], i, rates, n_of_rates);
        if (status != EXIT_SUCCESS) {
            fprintf(stderr,"ERROR IN assign_part_sum\n");
            exit(EXIT_FAILURE);
        }
    }
    
    
    // for(i=0; i<2 * n_of_rates;i++){
    //     cout << i << " "<< system[0].part_sum[i] << endl;
    // }

    // thermalize
    for  (i=0; i<DIM_ENSEMBLE; i++) {
        while (system[i].t < therm_t) {
            R = gsl_rng_uniform(r) * system[i].part_sum.back();
            j = upper_bound(system[i].part_sum.begin(), system[i].part_sum.end(), R) - system[i].part_sum.begin();

            status = update(&system[0], i, j, dim_chain);
            if (status != EXIT_SUCCESS) {
                fprintf(stderr,"ERROR in update\n");
                exit(EXIT_FAILURE); 
            }
            status = assign_part_sum(&system[0], i, rates, n_of_rates);
            if (status != EXIT_SUCCESS) {
                fprintf(stderr,"ERROR IN assign_part_sum\n");
                exit(EXIT_FAILURE);
            }
            t = gsl_ran_exponential(r, 1./system[i].part_sum.back());
            system[i].t = system[i].t + t;
        }
    }
    status = print_profile(densities_file, &system[0], dim_chain, DIM_ENSEMBLE); // print density profile
    fprintf(stdout,"NESS printed\n");
    if (status != EXIT_SUCCESS) {
        fprintf(stderr,"ERROR in print_profile()\n");
        exit(EXIT_FAILURE);
    }
        
    // add perturbation
    rates[0] = 0;
    for (i=0; i<DIM_ENSEMBLE; i++){
        status = assign_part_sum(&system[0], i, rates, n_of_rates);
        if (status != EXIT_SUCCESS) {
            fprintf(stderr,"ERROR IN assign_part_sum\n");
            exit(EXIT_FAILURE);
        }
    }
    
    // Simulate and save
    tmp = 0;
    while (tmp < simul_t){
        tmp = tmp + 10; //Print ever 10 time units.
        for  (i = 0; i < DIM_ENSEMBLE; i++) {
            while (system[i].t < therm_t + tmp) {

                R = gsl_rng_uniform(r) * system[i].part_sum.back();
                j = upper_bound(system[i].part_sum.begin(), system[i].part_sum.end(), R) - system[i].part_sum.begin();
                status = update(&system[0], i, j, dim_chain);
                if (status != EXIT_SUCCESS) {
                    fprintf(stderr,"ERROR in update\n");
                    exit(EXIT_FAILURE); 
                }

                status = assign_part_sum(&system[0], i, rates, n_of_rates);
                if (status != EXIT_SUCCESS) {
                    fprintf(stderr,"ERROR IN assign_part_sum\n");
                    exit(EXIT_FAILURE);
                }
                t = gsl_ran_exponential(r, 1./system[i].part_sum.back());
                system[i].t = system[i].t + t;
            }
        }
        status = print_profile(densities_file, &system[0], dim_chain, DIM_ENSEMBLE);
        fprintf(stdout,"density profile at %d printed.\n", tmp);
        if (status != EXIT_SUCCESS) {
            fprintf(stderr,"ERROR in print_profile()\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(densities_file);
    gsl_rng_free(r);

    delete [] rates;
    return 1;
}
