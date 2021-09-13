import numpy as np
import dWASEP_2_numba as dw
import SamplersKappa as ss
from scipy.interpolate import interp1d
import bisect
import sys


def deltat_max(R, deltax):
    return deltax ** 2 / R


def main(args):
    filename='.'.join(args.filename.split('.')[:-1]) + '_' + str(args.N_samples) + '_' + str(args.N_burnin)
    print(filename)
    data = np.genfromtxt(args.filename, skip_header=True)
    
    if args.times:
        times = args.times
    else:
        times = [0, 2, 5, 10, 20]

    if args.scale:
        data = data / args.scale
    else:
        data = data / np.max(data)

    if args.totalx:
        totalx = args.totalx
    else:
        totalx = int(args.filename.split('_')[-1].strip('.dat'))
        
    if args.deltax:
        N = int(totalx / args.deltax)
    else:
        if args.N:
            N = args.N
        else:
            sys.exit("Either `N` or `deltax` has to be provided.")
            
    if args.max_rate:
        max_rate = args.max_rate
    else:
        max_rate = 200

    if args.T:
        T = args.T
    else:
        T = int(times[-1] / deltat_max(max_rate, args.deltax))

    x = np.linspace(0, totalx, N).reshape(1, N)
    # interpolation is necessary as in oder to integrate the pde a finer grid is typically necessary
    y = interp1d(np.linspace(0, totalx, data.shape[0]), data, axis=0, kind='linear')(x[0]).transpose()
    
    dt = times[-1] / T
    print("deltat",  dt)
    dx = totalx / N
    print("deltax",  dx)
    print("max rate",  dx ** 2 / dt)
    
    Data = np.concatenate((x, y), axis=0)
    if args.noinflux:
        dWASEP = dw.DWASEP__(y[0,:], y[0,-1], rhoL = False,
                         T=T,
                         N=N,
                         totalt=times[-1],
                         totalx=totalx)
    else:
        dWASEP = dw.DWASEP__(y[0,:], y[0,-1], rhoL = y[0,0],
                         T=T,
                         N=N,
                         totalt=times[-1],
                         totalx=totalx)            
    # The true times per grid points are:
    k = np.r_[0:(times[-1]+dWASEP.deltat):dWASEP.deltat]
    # Find the indexes corresponding to the run-on times over the grid:
    t = [bisect.bisect_left(k, i) for i in times]
        
    gpmcmc = ss.GPMCMCdWASEP(Data, t, dWASEP)
    
    
    print("rate_profile_max", dWASEP.rate_profile_max)
    gpmcmc.sample(N_samples=args.N_samples,
                  N_burnin=args.N_burnin,
                  kappa_max=1.2, kappa_min=0.8,
                  filename=filename + 'Kappa')
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
        help="Name of file with data",
        type=str)
    parser.add_argument("N_burnin",  
        help="",
        type=int)
    parser.add_argument("N_samples",
        help="",
        type=int)
    parser.add_argument("-N", "--N",  
        help="No. of x-axis grid points",
        type=int)
    parser.add_argument("-T", "--T",  
        help="No. of y-axis grid points",
        type=int)
    parser.add_argument("-dx", "--deltax",  
        help="Bin size",
        type=float)
    parser.add_argument("-r", "--max_rate",  
        help="Maximum elongation rate allowed by the PDE solver.",
        type=float)
    parser.add_argument('-tx', '--totalx',
                        help='This is the real length of system (in the TASEP it is the number of sites).',
                        type=float)
    parser.add_argument('-t', '--times', action='append',
                        help='List of run-on times. Inputs as, e.g., python arg.py -t 0 -t 2 -t 5 -t 10',
                        type=int)
    parser.add_argument('-s', '--scale', type=float)
    parser.add_argument('-nif', '--noinflux', type=bool)
    
    args = parser.parse_args()
    print(args)
    main(args)
    print("Success.\n")

