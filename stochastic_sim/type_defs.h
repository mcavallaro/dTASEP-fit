#include <vector>

using namespace std;

#ifndef type_def
#define type_def
typedef struct Systems {
	vector<int> node;
	vector <double> part_sum;
    double t;
	int N;
} System;
#endif
