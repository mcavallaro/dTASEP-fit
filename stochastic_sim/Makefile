.PHONY: clean all debug

all: dTASEP.exe dTASEP_ic.exe

debug: dTASEP-g.exe dTASEP_ic-g.exe

clean:
	rm -rf *exe

dTASEP.exe: main.cpp update.cpp print.cpp
	g++ -O3 -Wall -o dTASEP.exe  main.cpp print.cpp update.cpp  -lm -lgsl -lgslcblas

dTASEP_ic.exe: main_ic.cpp update.cpp print.cpp
	g++ -O3 -Wall -o dTASEP_ic.exe  main_ic.cpp print.cpp update.cpp  -lm -lgsl -lgslcblas

dTASEP-g.exe: main.cpp update.cpp print.cpp
	g++ -O0 -Wall -g -o dTASEP-g.exe  main.cpp print.cpp update.cpp -lm -lgsl -lgslcblas

dTASEP_ic-g.exe: main_ic.cpp update.cpp print.cpp
	g++ -O0 -Wall -g -o dTASEP_ic-g.exe  main_ic.cpp print.cpp update.cpp -lm -lgsl -lgslcblas
