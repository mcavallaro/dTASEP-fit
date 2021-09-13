#!/usr/bin/env python
"""
Defines a class to integrate the density of the mean-field limit of the dWASEP
with arbitrary (variable) rate profile and  (fixed) Dirichlet boundary conditions.

Test as stand-alone program:
> python dWASEP_2.py
"""

import numpy as np
from scipy import special
from scipy.integrate import simps

# import warnings
# warnings.filterwarnings('error')


__author__ = "Massimo Cavallaro"

__license__ = "GPLv3"

__version__ = "0.99"

__email__ = ["m.cavallaro@warwick.ac.uk", "cavallaromass@gmail.com"]


class DWASEP:

    def __init__(self, rho0, rhoR, rhoL=False, totalt=50, a=1, deltax=None, deltat=None):

        # physical and numerical parameters:
        self.N = len(rho0)
        if deltat == None:
            self.deltat = 0.002
        else:
            self.deltat = deltat
        if deltax == None:
            self.deltax = 0.05
        else:
            self.deltax = deltax

        self.deltax2 = self.deltax ** 2
        self.rate_profile_max =  (self.deltax ** 2 / self.deltat)
        
        
        self.totalt = totalt
        self.T = int(totalt / self.deltat) + 1
        
        self.a = a # 0.05
        self.a2 = self.a ** 2

        self.macro_length = self.totalx = self.N * self.deltax / self.a # it's in the same units as rho0.
        
        print('memory usage: %.0f Mbs circa'%((3 * 8 * self.T * self.N) / 1000000))


        self.u = np.zeros([self.T, self.N])
        self.h = np.zeros([self.T, self.N])
        self.rho = np.zeros([self.T, self.N])

      # boundary conditions
        self.rho[:, -1] = rhoR
        self.rhoR = rhoR
        if rhoL is not False:
            self.rho[:, 0] = rhoL
            self.no_influx = False
            self.rhoL = rhoL
        else:
            self.no_influx = True
            print('Using no-influx left boundary')


        # initial conditions
        self.rho0 = rho0
        self.rho[0, :] = self.rho0

        # Cole-Hopf transform initial conditions
        tmp = rho0 - 0.5
        h0 = np.array([simps(tmp[:i]) for i in range(1,len(tmp)+1)]) * self.deltax / self.a
        self.u[0, :] = np.exp(2 * h0)
#         self.u[0, :] = np.exp(np.cumsum(2 * rho0 - 1) * self.deltax / self.a)

        if (self.u[0, :] < np.finfo(float).tiny).any():
            raise ValueError("The value of self.deltax / self.a could be too large. Try decreasing deltax.")

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

        # define bulk and ghost grid points
        self.x = np.arange(1, (self.N - 1))
        self.xm1 = np.arange(0, (self.N - 2))
        self.xp1 = np.arange(2, self.N)
        
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
        if len(rate_profile) != len(self.rho[0, :]):
            raise ValueError("the rate profile must have the same lenght as the the initial density profile")
                                
        # Transforming the GP generated function by sigmoid transformation to make it positive valued and upper bounded
        # by rate_profile_max
#         rate_profile = self.rate_profile_max * special.expit(self.a * rate_profile)
        D = 0.5 * rate_profile
        r = max(D) / self.rate_profile_max
                             
        if r > 0.5:
            raise ValueError("The integration scheme is unstable, with r=%f"%r)

        for t in range(1, self.T):
            # 2) bulk for u:
            self.u[t, self.x] = self.u[t - 1, self.x] + self.deltat * \
                                      D[self.x] * (self.a2 * ( self.u[t - 1, self.xm1] + self.u[t - 1, self.xp1] - 2 * self.u[t - 1, self.x]) / self.deltax2 - self.u[t - 1, self.x] )
            self.u[t, 0] = self.u[t - 1, 0] + self.deltat * \
                                      D[0] * (self.a2 * (self.L[t - 1] * self.u[t - 1, 0] + 2 * self.u[t - 1, 1]) / self.deltax2 - self.u[t - 1, 0] )
            # self.u[t, 0] = self.u[t - 1, 0] + self.deltat * \
            #                           D[1] * ((self.L[t - 1] * self.u[t - 1, 0] + 2 * self.u[t - 1, 1]) / self.deltax ** 2 - self.u[t - 1, 0]) ##last term was self.u[t - 1, 1])
            self.u[t, self.N - 1] = self.u[t - 1, self.N - 1] + self.deltat * \
                                      D[-1] * (self.a2 * (2 * self.u[t - 1, self.N - 2] + self.R[t - 1] * self.u[t - 1, self.N - 1]) / self.deltax2 - self.u[t - 1, self.N - 1] )

          # # compute current profile (optional)
          # self.J[t, :] = D[self.x] .* (0.5 - 0.5 * (self.u([t, self.xm1] + self.u[t, self.xp1] - 2 * self.u[t, self.x]) / (2 * self.deltax) / self.u[t, self.x]))

          # 4) bulk for rho (inverse Cole-Hopf transform):
            self.h[t, :] = np.log(self.u[t, :]) * 0.5 * self.a 
            self.rho[t, self.x] = (self.h[t, self.xp1] - self.h[t, self.xm1]) / (2 * self.deltax) + 0.5

          # 1) left boundaries for rho:
            if self.no_influx:
                self.rho[t, 0] = self.rho[t - 1, 0] - self.deltat * \
                             rate_profile[0] * self.rho[t - 1, 0] * (1 - self.rho[t - 1, 0] - self.a / self.deltax * (self.rho[t - 1, 0] - self.rho[t - 1, 1]))
            # else:
            #   self.rho[t, 0] = self.rhoL[t]
            # self.rho[t, self.N - 1] = self.rhoR[t]
            # 3) boundary for u:
            # in the no_influx boundary conditions must be updated recursivery
            # self.somma = self.somma + 0.5 - (self.u[t - 1, 2] - self.u[t - 1, 0]) / (4 * self.deltax * self.u[t - 1, 1])
                self.L[t] = - 2 - (4 * self.rho[t, 0] - 2) * self.deltax / self.a
            # L = - 2 - (4 * self.rho[t, 0] - 2) * self.deltax




    def plot_rho(self):
        "create axis, plot rho, return the axis"
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, self.N)
        T = np.arange(0, self.T)
        X, T = np.meshgrid(X, T)
        ax.plot_surface(X, T, self.rho, linewidth=0, antialiased=False, alpha=0.5,  cmap=plt.cm.seismic)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$')
        ax.set_zlabel(r'$\rho$')
        ax.view_init(30, 30)
        return fig, ax


    def plot_rho_2d(self, sl_par):
        "create axis, plot rho, return the axis"
        if sl_par > self.totalt:
            print('sl_par must be smaller than totalt')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        X = np.linspace(0, self.totalx, self.N)
        sl = np.r_[0:self.T:int(sl_par / self.deltat)]
        for s in sl:
            ax.plot(X, self.rho[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\rho$')
        ax.legend()
        return fig, ax

    
    def plot_u_2d(self, sl_par):
        "create axis, plot u, return the axis"
        if sl_par > self.totalt:
            print('sl_par must be smaller than totalt')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        X = np.linspace(0, self.totalx, self.N)
#         sl = np.r_[0:self.T:int(self.T / sl_par)]
        sl = np.r_[0:self.T:int(sl_par /self.deltat)]
        for s in sl:
            ax.plot(X, self.u[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.legend()
        return fig, ax
    

    def plot_u(self):
        "create axis, plot the height, return the axis"
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, self.N)
        T = np.arange(0, self.T)
        X, T = np.meshgrid(X, T)
        ax.plot_surface(T, X, self.u, linewidth=0, antialiased=False, alpha=0.5,  cmap=plt.cm.seismic)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$')
        ax.set_zlabel(r'$u$')
        ax.view_init(30, 30)
        return fig, ax

    def plot_J(self):
        "create axis, plot the current, return the axis"
        pass


class DWASEP_open(DWASEP):

    def __init__(self, rho0, alpha, beta, totalt=50, deltax=None, deltat=None):

        # physical and numerical parameters:
        self.N = len(rho0)
        if deltat == None:
            self.deltat = 0.002
        else:
            self.deltat = deltat
        if deltax == None:
            self.deltax = 0.05
        else:
            self.deltax = deltax
            
        self.alpha = alpha
        self.beta = beta

        self.deltax2 = self.deltax ** 2
        self.rate_profile_max = (self.deltax ** 2 / self.deltat)
        
        self.totalt = totalt
        self.T = int(totalt / self.deltat)
        
        self.macro_length = self.N * self.deltax # it's in the same units as rho0.

        self.a = 0.1 # 0.05
        self.a2 = self.a ** 2

        print('memory usage: %.0f Mbs circa'%((3 * 8 * self.T * self.N) / 1000000))

        self.u = np.zeros([self.T, self.N])
        self.h = np.zeros([self.T, self.N])
        self.rho = np.zeros([self.T, self.N])


        # initial conditions
        self.rho[0, :] = rho0

        # Cole-Hopf transform initial conditions
        tmp = rho0 - 0.5
        h0 = np.array([simps(tmp[:i]) for i in range(1,len(tmp)+1)]) * self.deltax / self.a
        self.u[0, :] = np.exp(2 * h0)
#         self.u[0, :] = np.exp(np.cumsum(2 * rho0 - 1) * self.deltax / self.a)

        if np.isnan(np.log(self.u[0, :])).any():
            raise ValueError("Bad rate profile initialisation or the value of self.deltax / self.a could be too large")

        self.L = np.empty(self.T, dtype=float)
        self.L[0] = - 2 - (4 * self.rho[0, 0] - 2) * self.deltax / self.a

        self.R = np.empty(self.T, dtype=float)
        self.R[0] = - 2 - (4 * self.rho[0, -1] - 2) * self.deltax / self.a
        
        # # calculate current only in the bulk for convenience (optional)
        # self.J = np.zeros([self.T, self.N - 2])

        # define bulk and ghost grid points
        self.x = np.arange(1, (self.N - 1))
        self.xm1 = np.arange(0, (self.N - 2))
        self.xp1 = np.arange(2, self.N)


    
    def integrate_pde(self, rate_profile):
        """
        bla bla
        """
        if len(rate_profile) != len(self.rho[0, :]):
            raise ValueError("the rate profile must have the same lenght as the the initial density profile")
                                
        # Transforming the GP generated function by sigmoid transformation to make it positive valued and upper bounded
        # by rate_profile_max
        # rate_profile = self.rate_profile_max * special.expit(self.a * rate_profile)
        rate_profile = rate_profile / self.a
        D = 0.5 * rate_profile
        r = max(D) / self.rate_profile_max
                             
        if r > 0.5:
            raise ValueError("The integration scheme is unstable, with r=%f"%r)

        for t in range(1, self.T):
            self.u[t, self.x] = self.u[t - 1, self.x] + self.deltat * \
                                      D[self.x] * (self.a2 * ( self.u[t - 1, self.xm1] + self.u[t - 1, self.xp1] - 2 * self.u[t - 1, self.x]) / self.deltax2 - self.u[t - 1, self.x])
            self.u[t, 0] = self.u[t - 1, 0] + self.deltat * \
                                      D[0] * (self.a2 * (self.L[t - 1] * self.u[t - 1, 0] + 2 * self.u[t - 1, 1]) / self.deltax2 - self.u[t - 1, 0])
            self.u[t, self.N - 1] = self.u[t - 1, self.N - 1] + self.deltat * \
                                      D[-1] * (self.a2 * (2 * self.u[t - 1, self.N - 2] + self.R[t - 1] * self.u[t - 1, self.N - 1]) / self.deltax2 - self.u[t - 1, self.N - 1])

            self.h[t, :] = np.log(self.u[t, :]) * 0.5 * self.a           
            
            self.rho[t, self.x] = (self.h[t, self.xp1] - self.h[t, self.xm1]) / (2 * self.deltax) + 0.5
            
            ######
            self.rho[t, 0] = self.rho[t - 1, 0] + self.deltat * \
                             (- rate_profile[0] * self.rho[t - 1, 0] * (1 - self.rho[t - 1, 0] - self.a / self.deltax * (self.rho[t - 1, 0] - self.rho[t - 1, 1])) + self.alpha * self.deltax * (1 - self.rho[t - 1, 0]))
            self.L[t] = - 2 - (4 * self.rho[t, 0] - 2) * self.deltax / self.a
#             print(t, self.L[t], self.rho[t, 0], self.u[t, 0])

            ####
            
            self.rho[t, -1] = self.rho[t - 1, -1] + self.deltat * \
                            (rate_profile[-1] * (self.rho[t - 1, -1] + self.a / self.deltax * (self.rho[t - 1, -2] - self.rho[t - 1, -1])) * (1 - self.rho[t - 1, -1]) - self.beta * self.rho[t - 1, -1])                       
            self.R[t] = - 2 + (4 * self.rho[t, -1] - 2) * self.deltax / self.a



class DWASEP_(DWASEP):
    def __init__(self, rho0, rhoR, rhoL=False, totalt=50, deltax=None, deltat=None, totalx=1000):
        from scipy.interpolate import interp1d
        import bisect

        # physical and numerical parameters:
        if deltat == None:
            self.deltat = 0.002
        else:
            self.deltat = deltat
        if deltax == None:
            self.deltax = 0.05
        else:
            self.deltax = deltax

        self.deltax2 = self.deltax ** 2
        self.rate_profile_max = self.deltax2 / self.deltat
        
        self.totalt = totalt
        self.T = int(totalt / self.deltat)
        
        self.totalx = totalx
        self.N = int(totalx / self.deltax)
       
        self.X = np.linspace(0, self.totalx, self.N)
        xdata = np.linspace(0, self.totalx, len(rho0))
        f_rho0 = interp1d(xdata, rho0, kind='linear')
        rho0_ = f_rho0(self.X)
        
        
        # the initial data rho0 can be retrieved from
        # the augmented vector rho0_ using the following indexes
#         self.k = np.r_[0:(totalx+self.deltax):self.deltax]
        self.index_xdata = np.array([bisect.bisect_left(self.X, i) for i in xdata])

        
#         # from scipy.interpolate import interp1d
# rho0 = [0,1,2,3,4,10,11,1,2,3,5]
# x = np.linspace(0, 10, 100)
# f_rho0 = interp1d(np.linspace(0, 10, len(rho0)), rho0, kind='linear')
# rho0_ = f_rho0(x)
# plt.plot(x, rho0_)
# plt.plot(rho0)
        
        
        self.a = self.totalx / len(rho0)
        self.a2 = self.a ** 2

        self.macro_length = self.N * self.deltax / self.a # it's in the same units as rho0.
        
        print('memory usage: %.0f Mbs circa'%((3 * 8 * self.T * self.N) / 1000000))

        self.u = np.zeros([self.T, self.N])
        self.h = np.zeros([self.T, self.N])
        self.rho = np.zeros([self.T, self.N])

        # boundary conditions
        self.rho[:, -1] = rhoR
        if rhoL is not False:
            self.rho[:, 0] = rhoL
            self.no_influx = False
        else:
            self.no_influx = True
            print('Using no-influx left boundary')


        # initial conditions
        self.rho[0, :] = rho0_

        # Cole-Hopf transform initial conditions
        tmp = rho0_ - 0.5
        h0 = np.array([simps(tmp[:i]) for i in range(1,len(tmp)+1)]) * self.deltax / self.a
        self.u[0, :] = np.exp(2 * h0)
        
        if np.isnan(np.log(self.u[0, :])).any():
            raise ValueError("Bad rate profile initialisation or the value of self.deltax / self.a could be too large")

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
        

    def plot_rho(self):
        "create axis, plot rho, return the axis"
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, self.N)
        T = np.arange(0, self.T)
        X, T = np.meshgrid(X, T)
        ax.plot_surface(X, T, self.rho, linewidth=0, antialiased=False, alpha=0.5,  cmap=plt.cm.seismic)

        xtickslocs = ax.get_xticks()
        ax.set_xticklabels(xtickslocs / self.N * self.totalx)

        ttickslocs = ax.get_yticks()
        ax.set_yticklabels(ytickslocs / self.N * self.totalt)
        
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$')
        ax.set_zlabel(r'$\rho$')
        ax.view_init(30, 30)
        return fig, ax


    def plot_rho_2d(self, sl_par):
        "create axis, plot rho, return the axis"
        if sl_par > self.totalt:
            print('sl_par must be smaller than totalt')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        X = np.linspace(0, self.totalx, self.N)
        sl = np.r_[0:self.T:int(sl_par / self.deltat)]
        for s in sl:
            ax.plot(X, self.rho[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\rho$')
        
        xtickslocs = ax.get_xticks()
        ax.set_xticklabels(xtickslocs / self.N * self.totalx)

        ax.legend()
        return fig, ax

    
    def plot_u_2d(self, sl_par):
        "create axis, plot u, return the axis"
        if sl_par > self.totalt:
            print('sl_par must be smaller than totalt')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        X = np.linspace(0, self.totalx, self.N)
#         sl = np.r_[0:self.T:int(self.T / sl_par)]
        sl = np.r_[0:self.T:int(sl_par /self.deltat)]
        for s in sl:
            ax.plot(X, self.u[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')

        xtickslocs = ax.get_xticks()
        ax.set_xticklabels(xtickslocs / self.N * self.totalx)
        
        ax.legend()
        return fig, ax
    

    def plot_u(self):
        "create axis, plot the height, return the axis"
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, self.N)
        T = np.arange(0, self.T)
        X, T = np.meshgrid(X, T)
        ax.plot_surface(T, X, self.u, linewidth=0, antialiased=False, alpha=0.5,  cmap=plt.cm.seismic)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$t$')
        ax.set_zlabel(r'$u$')
        
        xtickslocs = ax.get_xticks()
        ax.set_xticklabels(xtickslocs / self.N * self.totalx)

        ttickslocs = ax.get_yticks()
        ax.set_yticklabels(ytickslocs / self.N * self.totalt)        
        
        ax.view_init(30, 30)
        return fig, ax

    def plot_J(self):
        "create axis, plot the current, return the axis"
        pass
    
    

class DWASEP__(DWASEP_):
    def __init__(self, rho0, rhoR, rhoL=False, totalt=50, N=None, T=None, a=1, totalx=1000):

        # physical and numerical parameters:
        if T == None:
            self.deltat = 0.002
            self.T = totalt / self.deltat
        else:
            self.T = T + 1
            self.deltat = totalt / T
            
        if N == None:
            self.deltax = 0.05
            self.N = totalx / self.deltax
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
        if self.no_influx:
            self.L = np.empty(self.T, dtype=float)
            self.L[0] = - 2 - (4 * self.rho[0, 0] - 2) * self.deltax / self.a
        else:
            self.L = - 2 - 2 * (2 * self.rho[:, 0] - 1) * self.deltax / self.a

        self.R = - 2 + 2 * (2 * self.rho[:, -1] - 1) * self.deltax / self.a

        # # calculate current only in the bulk for convenience (optional)
        # self.J = np.zeros([self.T, self.N - 2])

        # define bulk and ghost grid points
        self.x = np.arange(1, (self.N - 1))
        self.xm1 = np.arange(0, (self.N - 2))
        self.xp1 = np.arange(2, self.N)


    def plot_rho_2d2(self, sl_par):
        "create axis, plot rho, return the axis"
        if max(sl_par) > self.T:
            print('sl_par.max() must be smaller than T')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        X = np.linspace(0, self.totalx, self.N)
        print(self.totalx, self.N)
#         sl = np.r_[0:self.T:int(sl_par / self.deltat)]
        for s in sl_par:
            ax.plot(X, self.rho[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\rho$')
        
#         xtickslocs = ax.get_xticks()
#         ax.set_xticklabels(np.round(xtickslocs / self.N * self.totalx,1))

        ax.legend()
        return fig, ax
    
    
#     def integrate_pde(self, rate_profile):
#         """
#         bla bla
#         """
#         if len(rate_profile) != len(self.rho[0, :]):
#             raise ValueError("the rate profile must have the same lenght as the the initial density profile")
                                
#         # Transforming the GP generated function by sigmoid transformation to make it positive valued and upper bounded
#         # by rate_profile_max
# #         rate_profile = self.rate_profile_max * special.expit(self.a * rate_profile)
#         D = 0.5 * rate_profile
#         r = max(D) / self.rate_profile_max
                             
#         if r > 0.5:
#             raise ValueError("The integration scheme is unstable, with r=%f"%r)

#         for t in range(1, self.T):
#             # 2) bulk for u:
#             self.u[t, self.x] = self.u[t - 1, self.x] + self.deltat * \
#                                       D[self.x] * (self.a2 * ( self.u[t - 1, self.xm1] + self.u[t - 1, self.xp1] - 2 * self.u[t - 1, self.x]) / self.deltax2 - self.u[t - 1, self.x] )
#             self.u[t, 0] = self.u[t - 1, 0] + self.deltat * \
#                                       D[0] * (self.a2 * (self.L[t - 1] * self.u[t - 1, 0] + 2 * self.u[t - 1, 1]) / self.deltax2 - self.u[t - 1, 0] )
#             # self.u[t, 0] = self.u[t - 1, 0] + self.deltat * \
#             #                           D[1] * ((self.L[t - 1] * self.u[t - 1, 0] + 2 * self.u[t - 1, 1]) / self.deltax ** 2 - self.u[t - 1, 0]) ##last term was self.u[t - 1, 1])
#             self.u[t, self.N - 1] = self.u[t - 1, self.N - 1] + self.deltat * \
#                                       D[-1] * (self.a2 * (2 * self.u[t - 1, self.N - 2] + self.R[t - 1] * self.u[t - 1, self.N - 1]) / self.deltax2 - self.u[t - 1, self.N - 1] )

#           # # compute current profile (optional)
#           # self.J[t, :] = D[self.x] .* (0.5 - 0.5 * (self.u([t, self.xm1] + self.u[t, self.xp1] - 2 * self.u[t, self.x]) / (2 * self.deltax) / self.u[t, self.x]))

#           # 4) bulk for rho (inverse Cole-Hopf transform):
#             self.h[t, :] = np.log(self.u[t, :]) * 0.5 * self.a 
#             self.rho[t, self.x] = (self.h[t, self.xp1] - self.h[t, self.xm1]) / (2 * self.deltax) + 0.5

#           # 1) left boundaries for rho:
#             if self.no_influx:
#                 self.rho[t, 0] = self.rho[t - 1, 0] - self.deltat * \
#                              rate_profile[0] * self.rho[t - 1, 0] * (1 - self.rho[t - 1, 0] - self.a / self.deltax * (self.rho[t - 1, 0] - self.rho[t - 1, 1]))
#             # else:
#             #   self.rho[t, 0] = self.rhoL[t]
#             # self.rho[t, self.N - 1] = self.rhoR[t]
#             # 3) boundary for u:
#             # in the no_influx boundary conditions must be updated recursivery
#             # self.somma = self.somma + 0.5 - (self.u[t - 1, 2] - self.u[t - 1, 0]) / (4 * self.deltax * self.u[t - 1, 1])
#                 self.L[t] = - 2 - (4 * self.rho[t, 0] - 2) * self.deltax / self.a
#             # L = - 2 - (4 * self.rho[t, 0] - 2) * self.deltax

#     def plot_rho(self):
#         "create axis, plot rho, return the axis"
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         X = np.arange(0, self.N)
#         T = np.arange(0, self.T)
#         X, T = np.meshgrid(X, T)
#         ax.plot_surface(X, T, self.rho, linewidth=0, antialiased=False, alpha=0.5,  cmap=plt.cm.seismic)
#         ax.set_xlabel(r'$x$')
#         ax.set_ylabel(r'$t$')
#         ax.set_zlabel(r'$\rho$')
#         ax.view_init(30, 30)
#         return fig, ax


#     def plot_rho_2d(self, sl_par):
#         "create axis, plot rho, return the axis"
#         if sl_par > self.totalt:
#             print('sl_par must be smaller than totalt')
#             return
#         import matplotlib.pyplot as plt
#         fig = plt.figure()
#         ax = fig.gca()
#         X = np.linspace(0, self.totalx, self.N)
#         sl = np.r_[0:self.T:int(sl_par / self.deltat)]
#         for s in sl:
#             ax.plot(X, self.rho[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
#         ax.set_xlabel(r'$x$')
#         ax.set_ylabel(r'$\rho$')
#         ax.legend()
#         return fig, ax

    
#     def plot_u_2d(self, sl_par):
#         "create axis, plot u, return the axis"
#         if sl_par > self.totalt:
#             print('sl_par must be smaller than totalt')
#             return
#         import matplotlib.pyplot as plt
#         fig = plt.figure()
#         ax = fig.gca()
#         X = np.linspace(0, self.totalx, self.N)
# #         sl = np.r_[0:self.T:int(self.T / sl_par)]
#         sl = np.r_[0:self.T:int(sl_par /self.deltat)]
#         for s in sl:
#             ax.plot(X, self.u[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
#         ax.set_xlabel(r'$x$')
#         ax.set_ylabel(r'$u$')
#         ax.legend()
#         return fig, ax


#     def plot_u(self):
#         "create axis, plot the height, return the axis"
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         X = np.arange(0, self.N)
#         T = np.arange(0, self.T)
#         X, T = np.meshgrid(X, T)
#         ax.plot_surface(T, X, self.u, linewidth=0, antialiased=False, alpha=0.5,  cmap=plt.cm.seismic)
#         ax.set_xlabel(r'$x$')
#         ax.set_ylabel(r'$t$')
#         ax.set_zlabel(r'$u$')
#         ax.view_init(30, 30)
#         return fig, ax

#     def plot_J(self):
#         "create axis, plot the current, return the axis"
#         pass


class DWASEP___(DWASEP_):
    def __init__(self, rho0, rhoR, rhoL=False, totalt=50, N=None, a=1, T=None, totalx=1000):
        from scipy.interpolate import interp1d
        import bisect

        # physical and numerical parameters:
        if T == None:
            self.deltat = 0.002
            self.T = totalt / self.deltat
        else:
            self.T = T
            self.deltat = totalt / T
            
        if N == None:
            self.deltax = 0.05
            self.N = totalx / self.deltax
        else:
            self.N = N
            self.deltax = totalx / N 

        self.deltax2 = self.deltax ** 2
        self.rate_profile_max = (self.deltax / self.deltat)
        
        print(self.rate_profile_max)
        
        self.totalt = totalt
        self.totalx = totalx

        self.X = np.linspace(0, self.totalx, self.N)
        xdata = np.linspace(0, self.totalx, len(rho0))
        rho0_augmented = interp1d(xdata, rho0, kind='linear')(self.X)
        
        # the initial data rho0 can be retrieved from
        # the augmented vector `rho0_augmented` using the following indexes
        self.index_xdata = np.array([bisect.bisect_left(self.X, i) for i in xdata])
        
        self.a = a
        self.a2 = self.a ** 2

        self.macro_length = self.N * self.deltax / self.a # it's in the same units as rho0.
        
        print('memory usage: %.0f Mbs circa'%((3 * 8 * self.T * self.N) / 1000000))

        self.u = np.zeros([self.T, self.N])
        self.h = np.zeros([self.T, self.N])
        self.rho = np.zeros([self.T, self.N])

        # boundary conditions
        self.rho[:, -1] = rhoR
        if rhoL is not False:
            self.rho[:, 0] = rhoL
            self.no_influx = False
        else:
            self.no_influx = True
            print('Using no-influx left boundary')


        # initial conditions
        self.rho[0, :] = rho0_augmented

        # Cole-Hopf transform initial conditions
        tmp = rho0_augmented - 0.5
        h0 = np.array([simps(tmp[:i]) for i in range(1,len(tmp)+1)]) * self.deltax / self.a
        self.u[0, :] = np.exp(2 * h0)
        
        if np.isnan(np.log(self.u[0, :])).any():
            raise ValueError("Bad rate profile initialisation or the value of self.deltax / self.a could be too large")

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

        # define bulk and ghost grid points
        self.x = np.arange(1, (self.N - 1))
        self.xm1 = np.arange(0, (self.N - 2))
        self.xp1 = np.arange(2, self.N)



    def plot_rho_2d2(self, sl_par):
        "create axis, plot rho, return the axis"
        if max(sl_par) > self.T:
            print('sl_par.max() must be smaller than T')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        X = np.linspace(0, self.totalx, self.N)
        print(self.totalx, self.N)
#         sl = np.r_[0:self.T:int(sl_par / self.deltat)]
        for s in sl_par:
            ax.plot(X, self.rho[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\rho$')
        
#         xtickslocs = ax.get_xticks()
#         ax.set_xticklabels(np.round(xtickslocs / self.N * self.totalx,1))

        ax.legend()
        return fig, ax

class DWASEP_open2(DWASEP_open):
    def __init__(self, rho0, alpha, beta, totalt=50, N=None, T=None, a=1, totalx=1000):

        # physical and numerical parameters:
        if T == None:
            self.deltat = 0.002
            self.T = totalt / self.deltat
        else:
            self.T = T + 1
            self.deltat = totalt / T
            
        if N == None:
            self.deltax = 0.05
            self.N = totalx / self.deltax
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
            
        self.alpha = alpha
        self.beta = beta

        self.deltax2 = self.deltax ** 2
        self.rate_profile_max = (self.deltax ** 2 / self.deltat)
        
        self.totalt = totalt
        self.T = int(totalt / self.deltat)
        
        self.macro_length = self.N * self.deltax # it's in the same units as rho0.

        self.a = a
        self.a2 = self.a ** 2

        self.u = np.zeros([self.T, self.N])
        self.h = np.zeros([self.T, self.N])
        self.rho = np.zeros([self.T, self.N])


        # initial conditions
        self.rho[0, :] = rho0

        # Cole-Hopf transform initial conditions
        tmp = rho0 - 0.5
        h0 = np.array([simps(tmp[:i]) for i in range(1,len(tmp)+1)]) * self.deltax / self.a
        self.u[0, :] = np.exp(2 * h0)
#         self.u[0, :] = np.exp(np.cumsum(2 * rho0 - 1) * self.deltax / self.a)

        if np.isnan(np.log(self.u[0, :])).any():
            raise ValueError("Bad rate profile initialisation or the value of self.deltax / self.a could be too large")

        self.L = np.empty(self.T, dtype=float)
        self.L[0] = - 2 - (4 * self.rho[0, 0] - 2) * self.deltax / self.a

        self.R = np.empty(self.T, dtype=float)
        self.R[0] = - 2 - (4 * self.rho[0, -1] - 2) * self.deltax / self.a
        
        # # calculate current only in the bulk for convenience (optional)
        # self.J = np.zeros([self.T, self.N - 2])

        # define bulk and ghost grid points
        self.x = np.arange(1, (self.N - 1))
        self.xm1 = np.arange(0, (self.N - 2))
        self.xp1 = np.arange(2, self.N)

    def plot_rho_2d2(self, sl_par):
        "create axis, plot rho, return the axis"
        if max(sl_par) > self.T:
            print('sl_par.max() must be smaller than T')
            return
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        X = np.linspace(0, self.totalx, self.N)
        print(self.totalx, self.N)
#         sl = np.r_[0:self.T:int(sl_par / self.deltat)]
        for s in sl_par:
            ax.plot(X, self.rho[s,:].T, alpha=0.5, label=r'$t=%.0f$'%(s * self.deltat))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$\rho$')
        
#         xtickslocs = ax.get_xticks()
#         ax.set_xticklabels(np.round(xtickslocs / self.N * self.totalx,1))

        ax.legend()
        return fig, ax
    
if __name__ == '__main__':
    import gc
    K = 5
    rho0 = np.hstack([np.repeat(0.9,K * 50), np.repeat(1,K * 50), np.repeat(0.1,K * 125)])
    # rhoL = rho0[0]
    rhoR = rho0[-1]

    dWASEP = DWASEP(rho0, rhoR, T=2000)

    rate_profile = np.hstack([
                      np.repeat(np.exp(0.1 * np.sin( - np.pi / 2 + np.pi)), K * 25),
                      np.exp(0.1* np.sin(np.linspace(- np.pi / 2 + np.pi, - np.pi / 2 + 5 * np.pi, K * 100))),
                      np.repeat(np.exp(0.1 * np.sin( - np.pi / 2 + 5 * np.pi)), K * 100)]) * 4 - 3.3

    rate_profile[(K * 150):(K * 155)] = 1
    # rate_profile = rate_profile - min(rate_profile)

    dWASEP.integrate_pde(rate_profile)


    for i in range(5):
      dWASEP = DWASEP(dWASEP.rho[-1, :], rhoR, T=20000)
      dWASEP.integrate_pde(rate_profile)

      if (i % 10) == 0:
        gc.collect()
      #   fig, ax =  dWASEP.plot_rho_2d(4)
      #   ax.plot(rate_profile / 4)
      #   fig.savefig('rho' + str(i) + '.png')
      #   plt.close()


    # after calling the integrator,
    # to get the density profile just use:
    # dWASEP.rho

    # print("the final density profile is:")
    # print(dWASEP.rho[-1,:])

    # print(dWASEP.rho[:,0])
    # print(dWASEP.rho[:,1])

    # fig, ax = dWASEP.plot_rho()
    # fig.show()
    fig, ax =  dWASEP.plot_rho_2d(7)
    fig.savefig('rho3.pdf')
