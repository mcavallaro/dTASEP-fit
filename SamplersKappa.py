# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:53:58 2019

@author: Yuexuan Wang (U1864389)
"""
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
#from scikits.sparse.cholmod import cholesky as scichol
from scipy.linalg import cholesky as scichol
import scipy 
from scipy.special import expit, logit
import copy

class Elliptical_Slice_Sampler:#Comes from Elliptical slice sampling [Murray et al., 2010] Figure 2 
    def __init__(self, mean, covariance, log_likelihood_function, re_parametrised):
        #Define  current state f, ellipse and log-likelihood function
        self.mean = mean
        self.covariance = covariance
        self.log_likelihood_function = log_likelihood_function
        self.re_parametrised = re_parametrised
    def __sample(self, f):
        # Choose ellipitical for sampling: upsilon~ N(0,K)
        up = np.random.multivariate_normal(np.zeros(self.mean.shape), self.covariance)
        # Set log-likelihood threshold : log(y)=logL(f)+log(u) where u~U[0,1]
        log_y = self.log_likelihood_function(f, self.mean) + np.log(np.random.uniform(low=0, high=1))
        # Define a bracket and draw an initial proposal
        theta = np.random.uniform(low=0.0, high=2.0 * np.pi)
        theta_min, theta_max = theta - 2.0*np.pi, theta
        
        #Interation
        while True:
            #transform f to f'
            if self .re_parametrised == True:
                fd = f * np.cos(theta) + up * np.sin(theta)
            else:
                fd = (f - self.mean) * np.cos(theta) + up * np.sin(theta) +self.mean
            log_fd = self.log_likelihood_function(fd, self.mean)
            #If logL(f')>log(y) then accept this f' and return it.
            if log_fd > log_y:
                return fd
            #Else shrink the bracket
            else:
                if theta < 0.0:
                    theta_min = theta
                else:
                    theta_max = theta
            #And try a new point from [theta_min, theta_max]
                theta = np.random.uniform(low=theta_min, high=theta_max)
                
    def sample(self, f, n_samples, burnin = 1):
        #Sample f' by input f and 1 burnin
        total_samples = n_samples + burnin
        samples = np.zeros((total_samples, self.covariance.shape[0]))
        samples[0] = f
        for i in range(1,total_samples):
            samples[i] = self.__sample(samples[i-1])
        return samples[burnin:]

#############################GPMCMC for dTASEP case##############################################
class GPMCMCdWASEP(object):
    def __init__(self, Data, t, dWASEP):
        self.data = Data
        self.X = Data[0, :].reshape(1, -1)
        self.y = Data[1:, :]
        # flattern the data
        self.y_flatten = self.y.flatten()
        self.t = t
        self.D_X = self._dis_matrix(self.X, self.X)
        self.dWASEP = dWASEP
    def sample(self, N_samples, N_burnin,
               l_min=0.1, l_max =10, kappa_min=0.01, kappa_max=1,
               filename='PosteriorSamples_X'):
        ################ Values of hyperparameters for Gaussian process sampling #########################
        # threshold of simga_epsilon
        sigma_e_min, sigma_e_max = 0, 10
        # threshold of mean
        m_min, m_max = 0, 2
        # threshold of l_x
        l_x_min, l_x_max = l_min, l_max #l_x_min: If too small the functions becomes non-smooth, l_x_max: If too big, Cholesky decomposition fails
        # threshold of sigma_f
        sigma_f_min, sigma_f_max = 0, 1
        
        input_parameters = {'N_samples':N_samples, 'N_burnin':N_burnin,
                  'l_min':l_min,'l_max':l_max,
                  'sigma_e_min':sigma_e_min, 'sigma_e_max':sigma_e_max,
                  'm_min':m_min, 'm_max':m_max,
                  'sigma_f_min':sigma_f_min, 'sigma_f_max':sigma_f_max,
                  'kappa_min':kappa_min, 'kappa_max':kappa_max}

        ############### Initializa the block of parameters depending on Data #####################################
        # 1. Initialize likelihood parameters
        # Initial value of m should be the mean of the data, sigm_e as the sd of data [Tegn ́er and Roberts, 2019] (17) sigmoid Gaussian priors
        likelihood_sample_z = np.random.multivariate_normal(np.zeros(3), np.diag(np.ones(3)))  # z for likelihood parameters
        # Alternative way of initializing using mean of init function and stdev of data and then doing inverse Scaled sigmoid transformation
        #m_z = -np.log(((np.mean(self.inverse_density(self.y[0, :]))-m_min)/(m_max - m_min))-1)
        #sigma_e_z = -np.log(((np.std(self.y[0, :])-sigma_e_min)/(sigma_e_max - sigma_e_min))-1)
        #likelihood_sample_z = np.array([m_z, sigma_e_z])
        # Using the likelihood_sample_z, compute the present value of likelihood_parameters
        
        m = m_min + (m_max - m_min) / (1 + np.exp(-likelihood_sample_z[0]))
        sigma_e = sigma_e_min + (sigma_e_max - sigma_e_min) / (1 + np.exp(-likelihood_sample_z[1]))
        kappa = kappa_min + (kappa_max - kappa_min) / (1 + np.exp(-likelihood_sample_z[2]))
        likelihood_parameters = np.array([m, sigma_e, kappa])

        # 2. Initialize the function value
        #`function_value` depends on `mi, ma` set to `0.45, 0.55`. this choice
        # makes the std of function value similar to the std of the data.
        function_value = np.zeros(shape=(len(self.y[0, :]),)) #self.inverse_density(self.y[0, :]) - likelihood_parameters[0] #re-parametrised

        # 3. Intialize the covariance parameters
        covariance_sample_z = np.random.multivariate_normal(np.zeros(2), np.eye(2))  # z for covariance parameters
        l_x = l_x_min + (l_x_max - l_x_min) / (1. + np.exp(-covariance_sample_z[0]))
        sigma_f = sigma_f_min + (sigma_f_max - sigma_f_min) / (1. + np.exp(-covariance_sample_z[1]))
        covariance_parameters = np.array([l_x, sigma_f])

        ## Following the initialization of covariance parameters, compute nu [Tegńer and Roberts, 2019] equation (28)
        SIGMA = self._kernel_function(self.D_X, covariance_parameters[0], covariance_parameters[1])
        L = np.linalg.cholesky(SIGMA)
        nu = np.dot(np.linalg.inv(L), function_value)
        #function_value = np.dot(L, up_c)

        ############# To save the intermediate samples ###########################
        Li_samples, F_samples, K_samples = [], [], []
        N_tot = N_burnin + N_samples
        for steps in range(0, N_tot):
            # step 1 : Block-Gibbs sampling of function value f
            # define log_likelihood function
            def LHD_function_values(f, mean):#[Tegn ́er and Roberts, 2019] Eq(18)
                # get PDE solver. Expit is in [0,1]. multiply f by rate_profile_max to cover all the allowed rage.
                self.dWASEP.scale_initial_profile(likelihood_parameters[2])
                self.dWASEP.integrate_pde(self.dWASEP.rate_profile_max * expit(self.dWASEP.a * (f + likelihood_parameters[0])))
                # self.dWASEP.integrate_pde(f + mean)
                rho_xt = np.take(self.dWASEP.rho, self.t, axis=0).flatten()
                return -1. / (2. * likelihood_parameters[1] ** 2) * (np.linalg.norm(likelihood_parameters[2] * self.y_flatten - rho_xt)) ** 2 - \
                       len(self.y_flatten) / 2. * np.log(2. * np.pi * likelihood_parameters[1] ** 2)
                       
            SIGMA = self._kernel_function(self.D_X, covariance_parameters[0], covariance_parameters[1])
            # Initialize elliptic sampler
            function_value_sampler = Elliptical_Slice_Sampler(likelihood_parameters[0] * np.ones(self.y.shape[1]),
                                                              SIGMA, LHD_function_values, True)
            # Sample using elliptic sampler
            # We perform 5 steps of elliptic slice sampler updating function value
            function_value_samples = function_value_sampler.sample(function_value, 5)
            function_value = function_value_samples[-1,]
            # Saving present value of function value to F_samples
            if steps > N_burnin:
                F_samples.append(copy.deepcopy(function_value))
            ####### step 2: Block-Gibbs sampling of covariance parameters (l_X, sigma_f) ###########
            ### define log_likelihood function
            def LHD_cov_parameters(vector_input, mean):
                # Scaled Sigmoid Transformation [Tegńer and Roberts, 2019] Eq(18)
                lx = l_x_min + (l_x_max - l_x_min) / (1. + np.exp(-vector_input[0]))
                sigmaf = sigma_f_min + (sigma_f_max - sigma_f_min) / (1. + np.exp(-vector_input[1]))
                # Computes function value using new covariance parameters #[Tegńer and Roberts, 2019] Eq(28)
                p_x = np.dot(np.linalg.cholesky(self._kernel_function(self.D_X, lx, sigmaf)), nu) + likelihood_parameters[0]
                self.dWASEP.scale_initial_profile(likelihood_parameters[2])
                self.dWASEP.integrate_pde(self.dWASEP.rate_profile_max * expit(self.dWASEP.a * p_x))
#               self.dWASEP.integrate_pde(p_x)
                rho_xt = np.take(self.dWASEP.rho, self.t, axis=0).flatten()
                return -1. / (2. * likelihood_parameters[1] ** 2) * (np.linalg.norm(likelihood_parameters[2] * self.y_flatten - rho_xt)) ** 2 - \
                       len(self.y_flatten) / 2. * np.log(2. * np.pi * likelihood_parameters[1] ** 2)

            ### Initialize and sample using elliptic sampler
            covariance_sampler = Elliptical_Slice_Sampler(np.zeros(2), np.diag(np.ones(2)), LHD_cov_parameters, False)
            ## Sample using elliptic sampler we get updated sample covariance_sample_z
            covariance_sample_z = covariance_sampler.sample(covariance_sample_z, 1)[0,]
            covariance_parameters[0] = l_x_min + (l_x_max - l_x_min) / (1. + np.exp(-covariance_sample_z[0]))
            covariance_parameters[1] = sigma_f_min + (sigma_f_max - sigma_f_min) / (1. + np.exp(-covariance_sample_z[1]))

            ### Recompute the value of nu # [Tegn ́er and Roberts, 2019] Eq(28)
            SIGMA = self._kernel_function(self.D_X, covariance_parameters[0], covariance_parameters[1])
            L = np.linalg.cholesky(SIGMA)
            nu = np.dot(np.linalg.inv(L), function_value)
            #function_value = np.dot(L, up_c)
            ### Saving present value of covariance parameters to K_samples
            if steps > N_burnin:
                K_samples.append(copy.deepcopy(covariance_parameters))
            ######### step 3: Block-Gibbs sampling of log-likelihood parameters ########
            ### define log_likelihood function
            def LHD_likelihood_parameters(vector_input, mean):#[Tegn ́er and Roberts, 2019] Eq(18)
                m_1 = m_min + (m_max - m_min) / (1. + np.exp(-vector_input[0]))
                sigma_e_1 = sigma_e_min + (sigma_e_max - sigma_e_min) / (1. + np.exp(-vector_input[1]))
                # prior:
                kappa = kappa_min + (kappa_max - kappa_min) / (1. + np.exp(-vector_input[2]))
                #
                p_x = function_value + m_1
                # self.dWASEP.integrate_pde(p_x)
                self.dWASEP.scale_initial_profile(kappa)
                self.dWASEP.integrate_pde(self.dWASEP.rate_profile_max * expit(self.dWASEP.a * p_x))
                rho_xt = np.take(self.dWASEP.rho, self.t, axis=0).flatten()
                return -1. / (2. * sigma_e_1 ** 2) * (np.linalg.norm(kappa * self.y_flatten - rho_xt)) ** 2 - \
                       len(self.y_flatten) / 2. * np.log(2. * np.pi * sigma_e_1 ** 2)

            ### Initialize and sample using elliptic sampler
            likelihood_sampler = Elliptical_Slice_Sampler(np.zeros(3), np.diag(np.ones(3)), LHD_likelihood_parameters, False)            
            
            # Sample using elliptic sampler we get updated sample likelihood_sample_z
            likelihood_sample_z = likelihood_sampler.sample(likelihood_sample_z, 1)[0,]
            # update likelihood parameters
            likelihood_parameters[0] = m_min + (m_max - m_min) / (1 + np.exp(-likelihood_sample_z[0]))
            likelihood_parameters[1] = sigma_e_min + (sigma_e_max - sigma_e_min) / (1 + np.exp(-likelihood_sample_z[1]))
            likelihood_parameters[2] = kappa_min + (kappa_max - kappa_min) / (1 + np.exp(-likelihood_sample_z[2])) 

            ### Saving present ualue of likelihood parameters to the Li_samples
            if steps > N_burnin:
                Li_samples.append(copy.deepcopy(likelihood_parameters))

            if steps % 10 == 0:
                print("Iteration %d of %d"%(steps, N_tot), end='\r')
        
            if (steps > N_burnin) and (steps % 20 == 0):
                tmp_filename = filename.split('/')
                tmp_filename[-1] = str(steps) + tmp_filename[-1]
                tmp_filename = '/'.join(tmp_filename)
                np.savez(tmp_filename, parameters=input_parameters,
                         Data=self.data, F_samples=F_samples, K_samples=K_samples, Li_samples=Li_samples)
        
        np.savez(filename, parameters=input_parameters,
                 Data=self.data, F_samples=F_samples, K_samples=K_samples, Li_samples=Li_samples)
        return self.data, F_samples, K_samples, Li_samples

    def _dis_matrix(self, data1, data2):
        D = np.zeros(shape=(data1.shape[1], data2.shape[1]))
        for i in range(0, data1.shape[1]):
            for j in range(0, data2.shape[1]):
                D[i, j] = pow((data1[0, i] - data2[0, j]), 2)
        return D

#     def _kernel_function(self, D, l, sigma_f):
#         return (sigma_f ** 2) * np.exp(-D / (2 * pow(l, 2)))
    def _kernel_function(self, D, l, sigma_f):
        return (sigma_f ** 2) * np.exp(-D / (2 * pow(l, 2))) + 1e-13 * np.eye(D.shape[0])


    def prediction(self, X_star, confidence = 0.95, filename_read='PosteriorSamples_X.npz', filename_write='PredictionSamples_X.npz'):
        Samples = np.load(filename_read)
        Data, F_samples, K_samples, Li_samples = Samples['Data'], Samples['F_samples'], Samples['K_samples'], Samples[
            'Li_samples']
        M = len(F_samples)
        Li = np.array(Li_samples)
        K = np.array(K_samples)
        F = np.array(F_samples)
        X = Data[0, :].reshape(1,-1)

        # Figure out KXX, KXX* and KX*X*
        XX_D = self._dis_matrix(X, X)
        X_starX_D = self._dis_matrix(X_star, X)
        X_starX_star_D = self._dis_matrix(X_star, X_star)

#         iters = 0
        Mean, Covariance = 0, 0
        for steps in range(M):
            function_value = F[steps]
            l_x = K[steps, 0]
            sigma_f = K[steps, 1]
            m = Li[steps, 0]
            sigma_ep = Li[steps, 1]
            KXX = self._kernel_function(XX_D, l_x, sigma_f)
            KX_starX = self._kernel_function(X_starX_D, l_x, sigma_f)
            KXX_star = KX_starX.transpose()
            KX_starX_star = self._kernel_function(X_starX_star_D, l_x, sigma_f)
            #prediction [Rasmussen and Williams, 2006] Eq(2.23, 2.24)
            m_f_star = m + KX_starX.dot(inv(KXX + sigma_ep**2 * np.eye(KXX.shape[0]))).dot(function_value)
            cov_f_star = KX_starX_star - KX_starX.dot(inv(KXX + sigma_ep**2 * np.eye(KXX.shape[0]))).dot(KXX_star)
            Mean += m_f_star
            Covariance +=cov_f_star
#             iters += 1
#             if iters % 100 == 0:
#                 print("iteration =", iters / 100)
        #predicted mean and confidence interval
        Mean, Covariance = Mean/M , Covariance / M
        np.savez(filename_write, Mean=Mean, Covariance=Covariance)
        # Print predicted data
        var = Covariance.diagonal()
        N = len(Mean)
        h = var * scipy.stats.t.ppf((1 + confidence) / 2., N-1)
        #Define confidence interval
        C1 = Mean + h
        C2 = Mean - h
        X_star = X_star.reshape(-1)
        fig = plt.figure()
        ax = plt.fill_between(X_star, C1, C2, alpha=0.2,label = 'Confidence Interval')
        ax = plt.plot(X_star, Mean)
        ax = plt.plot(self.X.reshape(-1), self.y, '.', label = 'data')
        plt.legend()
        return fig, ax

    def inverse_density(self, rho0):
        # BOOKMARK
        tmp = -rho0 - min(-rho0)
        #     print(min(tmp), max(tmp))
        mi, ma = 0.45, 0.55
        #mi, ma = 0.1, 0.9
        rescaled_tmp = (ma - mi) * (tmp - min(tmp)) / (max(tmp) - min(tmp)) + mi
        #print(min(rescaled_tmp), max(rescaled_tmp))
        r = logit(rescaled_tmp)
        return r  # - np.mean(r)
