#include <cstdlib>
#include <vector>
#include <cstdio>
#include "type_defs.h"
#include "print.h"
#include <cmath>


int print_profile(FILE *f, Systems* system, int dim_chain, int DIM_ENSEMBLE){ //stampa density profile

    double * profile;
    profile = new double [dim_chain];
    int j,i;

    for (j  = 0; j < dim_chain; j++){
        profile[j] = 0;
    }

    for(i = 0; i < DIM_ENSEMBLE; i++){
        for (j  = 0; j < dim_chain; j++){
            profile[j] = profile[j] + system[i].node[j];
        }
    }

    for (j = 0; j < dim_chain; j++){
        fprintf(f, "%f ", profile[j]/DIM_ENSEMBLE );
    }
    fprintf(f, "\n" );

    delete [] profile; 

    return EXIT_SUCCESS;
}
