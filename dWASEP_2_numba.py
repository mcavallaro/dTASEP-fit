#!/usr/bin/env python
"""
Defines a class to integrate the density of the mean-field limit of the dWASEP
with numba JIT compiler.

Test as stand-alone program:
> python dWASEP_2_numba.py
"""
import numpy as np
from scipy import special
from scipy.integrate import simps
from numba import jit, int64, float64, boolean
# import warnings
# warnings.filterwarnings('error')

__author__ = "Massimo Cavallaro"
__license__ = "GPLv3"
__version__ = "0.99"
__email__ = ["m.cavallaro@warwick.ac.uk", "cavallaromass@gmail.com"]

# spec = [
#     ('T', int64),
#     ('deltat', float64),
#     ('deltax', float64),
#     ('N', int64),
#     ('deltax2', float64),
#     ('rate_profile_max', float64),
#     ('totalt', float64),
#     ('totalx', float64),
#     ('a', float64),
#     ('a2', float64),
#     ('macro_length', float64),
#     ('no_influx', boolean),    
#     ('u', float64[:,:]),
#     ('h', float64[:,:]),
#     ('rho', float64[:,:]),
#     ('rho0', float64[:]),
#     ('rhoL', float64[:]),
#     ('rhoR', float64[:]),
#     ('L', float64[:]),
#     ('R', float64[:]),
#     ('xm1', int64[:])
#     ('xp1', int64[:])
#     ('x', int64[:])
# ]

   
@jit(int64(float64[:,::1], float64[:,::1], float64[:,::1], int64, int64, float64, float64, float64,
           float64, float64,
#           int64[::1], int64[::1], int64[::1],
           float64[::1], float64[::1], float64[::1], float64[::1], boolean), nopython=True)
def integrate(u, h, rho, T, N, deltat, deltax, deltax2,
              a, a2,
#               xm1, xp1, x,
              L, R, D, rate_profile, no_influx):
        for t in range(1, T):
            # 2) bulk for u:
            for i in range(1,N-1):
                u[t, i] = u[t - 1, i] + deltat * \
                                      D[i] * (a2 * (u[t - 1, i-1] + u[t - 1, i+1] - 2 * u[t - 1, i]) / deltax2 - u[t - 1, i] )
            
            u[t, 0] = u[t - 1, 0] + deltat * \
                                      D[0] * (a2 * (L[t - 1] * u[t - 1, 0] + 2 * u[t - 1, 1]) / deltax2 - u[t - 1, 0] )
            # self.u[t, 0] = self.u[t - 1, 0] + self.deltat * \
            #                           D[1] * ((self.L[t - 1] * self.u[t - 1, 0] + 2 * self.u[t - 1, 1]) / self.deltax ** 2 - self.u[t - 1, 0]) ##last term was self.u[t - 1, 1])
            u[t, N - 1] = u[t - 1, N - 1] + deltat * \
                                      D[-1] * (a2 * (2 * u[t - 1, N - 2] + R[t - 1] * u[t - 1, N - 1]) / deltax2 - u[t - 1, N - 1] )

          # # compute current profile (optional)
          # self.J[t, :] = D[self.x] .* (0.5 - 0.5 * (self.u([t, self.xm1] + self.u[t, self.xp1] - 2 * self.u[t, self.x]) / (2 * self.deltax) / self.u[t, self.x]))

          # 4) bulk for rho (inverse Cole-Hopf transform):
            h[t, :] = np.log(u[t, :]) * 0.5 * a
            for i in range(1, N-1):
                rho[t, i] = (h[t, i+1] - h[t, i-1]) / (2 * deltax) + 0.5

          # 1) left boundaries for rho:
            if no_influx:
                rho[t, 0] = rho[t - 1, 0] - deltat * \
                             rate_profile[0] * rho[t - 1, 0] * (1 - rho[t - 1, 0] - a / deltax * (rho[t - 1, 0] - rho[t - 1, 1]))
            # else:
            #   self.rho[t, 0] = self.rhoL[t]
            # self.rho[t, self.N - 1] = self.rhoR[t]
            # 3) boundary for u:
            # in the no_influx boundary conditions must be updated recursivery
            # self.somma = self.somma + 0.5 - (self.u[t - 1, 2] - self.u[t - 1, 0]) / (4 * self.deltax * self.u[t - 1, 1])
                L[t] = - 2 - (4 * rho[t, 0] - 2) * deltax / a
            # L = - 2 - (4 * self.rho[t, 0] - 2) * self.deltax
            
        return 1
        

class DWASEP__(object):
    def __init__(self, rho0, rhoR, rhoL=False, totalt=50, N=None, T=None, a=1., totalx=1000):

        # physical and numerical parameters:
        if T == None:
            self.deltat = 0.002
            self.T = int(totalt / self.deltat)
        else:
            self.T = T + 1
            self.deltat = totalt / T
            
        if N == None:
            self.deltax = 0.05
            self.N = int(totalx / self.deltax)
        else:
            self.N = N
            self.deltax = totalx / N 

        self.deltax2 = self.deltax ** 2
        
        # warning the rate profile max according to stability condition
        # of Euler methods has dx^2 at numerator, but here we have only dx beacuse the true rate
        # profiled is rescaled by deltax
        self.rate_profile_max = self.deltax2 / self.deltat
        
        self.totalt = totalt + self.deltat    
        self.totalx = totalx
        
        self.a = a
        self.a2 = self.a ** 2

        self.macro_length = self.N * self.deltax / self.a # it's in the same units as rho0.
        
        print('memory usage: %.0f Mbs circa'%((3 * 8 * self.T * self.N) / 1000000))

        self.u = np.zeros([self.T, self.N])
        self.h = np.zeros([self.T, self.N])
        self.rho = np.zeros([self.T, self.N])

        self.rhoR = rhoR
        self.rho0 = rho0
        self.rho[:, -1] = rhoR
        
        if rhoL is not False:
            self.rho[:, 0] = rhoL
            self.rhoL = rhoL
            self.no_influx = False
        else:
            self.no_influx = True
            print('Using no-influx left boundary')

        # initial conditions
        self.rho[0, :] = rho0
        
        # Cole-Hopf transform initial conditions
        tmp = rho0 - 0.5
        h0 = np.array([simps(tmp[:i]) for i in range(1,len(tmp)+1)]) * self.deltax / self.a
        self.u[0, :] = np.exp(2 * h0)
        
        if np.isnan(np.log(self.u[0, :])).any():
            raise ValueError("Bad rate profile initialisation or the value of self.deltax / self.a could be too large")

        # if rhoL == False:
        # self.rhoL = rho0[0]
        #   # self.somma = 0.5 - (self.u[0, 2] - self.u[0, 0]) / (4 * self.deltax * self.u[0, 1])
        #   self.L = np.empty(T - 1, dtype=float)
        self.L = np.empty(self.T, dtype=float)
        self.R = np.empty(self.T, dtype=float)
        
        if self.no_influx:
            self.L[0] = - 2 - (4 * self.rho[0, 0] - 2) * self.deltax / self.a
        else:
            self.L[:] = - 2 - 2 * (2 * self.rho[:, 0] - 1) * self.deltax / self.a

        self.R[:] = - 2 + 2 * (2 * self.rho[:, -1] - 1) * self.deltax / self.a
     
        # # calculate current only in the bulk for convenience (optional)
        # self.J = np.zeros([self.T, self.N - 2])

        # define bulk and ghost grid points
        self.x = np.arange(1, (self.N - 1))
        self.xm1 = np.arange(0, (self.N - 2))
        self.xp1 = np.arange(2, self.N)

    def transform_rate(self, rate):
        """
        Simple utility method to transform the rate
        to match the grid
        """
        from scipy.interpolate import interp1d
        x = np.linspace(0, self.totalx, self.N)
#         rho0_  = rho0[rho0_x]
        f = interp1d(np.linspace(0, self.totalx, len(rate)), rate, kind='linear')
        return f(x) #* self.N / len(rate)
    def scale_initial_profile(self, kappa):
        # initial conditions
        self.rho[0, :] = self.rho0 * kappa
        if not self.no_influx:
            self.rho[:, 0] = self.rhoL * kappa
        self.rho[:, -1] = self.rhoR * kappa
        
        # Cole-Hopf transform initial conditions
        tmp = self.rho[0, :] - 0.5
        h0 = np.array([simps(tmp[:i]) for i in range(1,len(tmp)+1)]) * self.deltax / self.a
        self.u[0, :] = np.exp(2 * h0)
#         self.u[0, :] = np.exp(np.cumsum(2 * rho0 - 1) * self.deltax / self.a)

        if (self.u[0, :] < np.finfo(float).tiny).any():
            print("The value of self.deltax / self.a could be too large. Try decreasing deltax.")

        # if rhoL == False:
        # self.rhoL = rho0[0]
        #   # self.somma = 0.5 - (self.u[0, 2] - self.u[0, 0]) / (4 * self.deltax * self.u[0, 1])
        #   self.L = np.empty(T - 1, dtype=float)
        if self.no_influx:
            self.L = np.empty(self.T, dtype=float)
            self.L[0] = - 2 - (4 * self.rho[0, 0] - 2) * self.deltax / self.a
        else:
            self.L = - 2 - 2 * (2 * self.rho[:, 0] - 1) * self.deltax / self.a

        self.R = - 2 + 2 * (2 * self.rho[:, -1] - 1) * self.deltax / self.a
        # # calculate current only in the bulk for convenience (optional)
        # self.J = np.zeros([self.T, self.N - 2])

    def integrate_pde(self, rate_profile):
        """
        bla bla
        """
#         if len(rate_profile) != len(self.rho[0, :]):
#             raise ValueError("the rate profile must have the same lenght as the the initial density profile")
                                
        # Transforming the GP generated function by sigmoid transformation to make it positive valued and upper bounded
        # by rate_profile_max
#         rate_profile = self.rate_profile_max * special.expit(self.a * rate_profile)
        D = 0.5 * rate_profile
        r = max(D) / self.rate_profile_max
                             
        if r > 0.5:
#            raise ValueError("The integration scheme is unstable, with r=%f"%r)
            print("The integration scheme is unstable, with r=%f"%r)
    
        
        tmp = integrate(self.u, self.h, self.rho,
                  self.T, self.N, self.deltat, self.deltax, self.deltax2,
                  self.a, self.a2,
#                   self.xm1, self.xp1, self.x,
                  self.L, self.R, D, rate_profile, self.no_influx)
        if tmp != 1:
            print(tmp)
     

    
if __name__ == '__main__':
    
    import gc
    import time
    K = 100
    rho0 = np.hstack([np.repeat(0.9,K * 50), np.repeat(1,K * 50), np.repeat(0.1,K * 100)])
    # rhoL = rho0[0]
    rhoR = rho0[-1]

    dWASEP = DWASEP__(rho0, rhoR, T=500000, totalt=100)
        
    rate_profile = np.hstack([
                      np.repeat(np.exp(0.1 * np.sin( - np.pi / 2 + np.pi)), K * 50),
                      np.exp(0.1* np.sin(np.linspace(- np.pi / 2 + np.pi, - np.pi / 2 + 5 * np.pi, K * 50))),
                      np.repeat(np.exp(0.1 * np.sin( - np.pi / 2 + 5 * np.pi)), K * 100)]) * 4 - 3.3
    
    rate_profile = np.repeat(0.1, len(rho0))

    # rate_profile[(K * 150):(K * 155)] = 1
    # rate_profile = rate_profile - min(rate_profile)
    start = time.time()
    dWASEP.integrate_pde(rate_profile)
    end = time.time()
    
    print("Elapsed (after compilation) = %s" % (end - start))
#     start = time.time()
    
#     for i in range(5):
#         dWASEP = DWASEP__(dWASEP.rho[-1, :], rhoR, T=20000, totalt=50)
#         dWASEP.integrate_pde(rate_profile)

#         if (i % 10) == 0:
#             gc.collect()
#       #   fig, ax =  dWASEP.plot_rho_2d(4)
#       #   ax.plot(rate_profile / 4)
#       #   fig.savefig('rho' + str(i) + '.png')
#       #   plt.close()


#     # after calling the integrator,
#     # to get the density profile just use:
#     # dWASEP.rho

#     # print("the final density profile is:")
#     # print(dWASEP.rho[-1,:])

#     # print(dWASEP.rho[:,0])
#     # print(dWASEP.rho[:,1])

#     # fig, ax = dWASEP.plot_rho()
#     # fig.show()
# #     fig, ax =  dWASEP.plot_rho_2d(7)
    
# #     fig.savefig('rho3.pdf')
#     end = time.time()
    print("Elapsed (after compilation) * 5 = %s" % (end - start))
