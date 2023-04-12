#include <cstdlib>
#include <vector>
#include <cstdio>
#include "type_defs.h"

#include <iostream>

using namespace std;


// int update(Systems *system, int i, int j, int dim_chain){
//     if (j == 0){
//     	system[i].node[0]++;
//     	system[i].N++;
//     }
//     else if (j < dim_chain){
//     	system[i].node[j-1]--;
//     	system[i].node[j]++;
//     }
//     else if (j == dim_chain){ //leave the system
//     	system[i].N--;
//         system[i].node[j-1]--;
//     }
//     else {
// 		fprintf(stderr,"error_777\n");
// 		exit(EXIT_FAILURE);
// 	}
// 	return EXIT_SUCCESS;
// }

// int assign_part_sum(Systems *system, int i, double * p, int n_of_rates){

//     int j;
//     if(system[i].node[0] == 1){
//         system[i].part_sum[0] = 0;
//     }
//     else if (system[i].node[0] == 0){
//         system[i].part_sum[0] = p[0];
//     }
//     for (j=1; j<n_of_rates-1; j++) {
//         system[i].part_sum[j] = system[i].part_sum[j-1] + system[i].node[j-1] * (1 - system[i].node[j]) * p[j];
//     }

//     system[i].part_sum[n_of_rates-1] = system[i].part_sum[n_of_rates-2] + system[i].node[n_of_rates-2] * p[n_of_rates-1];

//     return EXIT_SUCCESS;
// }


int update(Systems *system, int i, int j, int dim_chain){
    int pos;
    if (j == 0){// enter the system from the left
        system[i].node[0]++;
        system[i].N++;
    }
    else if (j < dim_chain){
        system[i].node[j-1]--;
        system[i].node[j]++;
    }
    else if (j == dim_chain){ //leave the system rightwards
        system[i].N--;
        system[i].node[j-1]--;
    }
    //
    else if (j == dim_chain + 1){ // enter the system from the right
        system[i].node[dim_chain-1]++;
        system[i].N++;
    }
    else if ((j > dim_chain + 1) && (j <= 2 * dim_chain)){
        pos = 2 * dim_chain + 1 - j;
        system[i].node[pos]--;
        system[i].node[pos-1]++;
    }
    else if (j == 2 * dim_chain + 1){ //leave the system leftwards
        system[i].N--;
        system[i].node[0]--;
    }
    else {
        fprintf(stderr, "error_777  %d %d\n", j, dim_chain);
        for (int J=0; J<dim_chain; J++){
            printf("%d\n",  system[i].node[J] );

        }
        exit(EXIT_FAILURE);
    }
    for(int k=0; k<dim_chain; k++){
        if(system[i].node[k] > 1){
            cerr << "9999 " << k <<" "<< system[i].node[k] << " " << j << endl;
            exit(EXIT_FAILURE);
        }
    }
    return EXIT_SUCCESS;
}


// int assign_part_sum(Systems *system, int i, double * p, double * q, int n_of_rates_2){
//     int n_of_rates = x * 2;
//     int j, jj;
//     if(system[i].node[0] == 1){
//         system[i].part_sum[0] = 0;
//     }
//     else if (system[i].node[0] == 0){
//         system[i].part_sum[0] = p[0];
//     }
//     for (j=1; j<n_of_rates_2-1; j++) {
//         system[i].part_sum[j] = system[i].part_sum[j-1] + system[i].node[j-1] * (1 - system[i].node[j]) * p[j];
//     }
//     system[i].part_sum[n_of_rates_2-1] = system[i].part_sum[n_of_rates_2-2] + system[i].node[n_of_rates_2-2] * p[n_of_rates_2-1];
//     //
//     if(system[i].node[n_of_rates_2-1] == 1){
//         system[i].part_sum[n_of_rates_2] = system[i].part_sum[n_of_rates_2-1];
//     }
//     else if (system[i].node[n_of_rates_2-1] == 0){
//         system[i].part_sum[n_of_rates_2] = system[i].part_sum[n_of_rates_2-1] + q[n_of_rates - 1];
//     }
//     for (j=n_of_rates_2, jj=n_of_rates_2-1; j<n_of_rates-1; j++, jj--) {
//         system[i].part_sum[j] = system[i].part_sum[j-1] + system[i].node[jj] * (1 - system[i].node[jj-1]) * q[jj];
//     }
//     system[i].part_sum[n_of_rates-1] = system[i].part_sum[n_of_rates-2] + system[i].node[0] * q[0];
// 
//     return EXIT_SUCCESS;
// }



int assign_part_sum(Systems *system, int i, double * p, int n_of_rates){
    int j;
    if(system[i].node[0] == 1){
        system[i].part_sum[0] = 0;
    }
    else if (system[i].node[0] == 0){
        system[i].part_sum[0] = p[0];
    }
    for (j=1; j<n_of_rates-1; j++) {
        system[i].part_sum[j] = system[i].part_sum[j-1] + system[i].node[j-1] * (1 - system[i].node[j]) * p[j];
    }
    system[i].part_sum[n_of_rates-1] = system[i].part_sum[n_of_rates-2] + system[i].node[n_of_rates-2] * p[n_of_rates-1];
    
    return EXIT_SUCCESS;
}










