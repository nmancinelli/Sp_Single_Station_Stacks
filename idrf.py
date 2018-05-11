from numpy import sqrt, argmax, flipud, fliplr
from numpy import zeros, correlate, convolve, linspace, exp, arange

def IDRF(P,D,dt,t_for=5,t_trun=5,gauss_t=0.7,accept_mis=-0.1):
    """Iterative Deconvolution and Receiver-Function Estimation in time domain
    
    Modified from Matlab code from Junlin Hua -- May 2018.
    
    Note: accept_mis feature does not work properly in this version.
    
    """

    t_max = t_for;

    misfit = 1.;
    misfit_old = 99999.;
    misfit_ref = sqrt(sum(D**2));

    RF = zeros(len(P));

    D_cur = D;

    itmax = 400;

    itnum = 0;
    for itnum in range(itmax):
        amp_corr  = correlate(D_cur,P,'same');
        auto_corr = correlate(P,P);
        ind = argmax(abs(amp_corr));
        amp_rf = amp_corr[ind]/auto_corr;
        RF[ind] = RF[ind]+amp_rf;
        D_sub = convolve(P,RF,'same');
        D_cur = D - D_sub;
        misfit_old = misfit;
        misfit = sqrt(sum(D_cur**2))/misfit_ref;
        itnum = itnum+1;
        
        #from matplotlib import pylab as plt
        #plt.plot(RF)
        #plt.plot(D_sub)
        #plt.show()
        #break
        #print(misfit, accept_mis)
        
        if itnum == itmax or (misfit_old - misfit) < accept_mis :
            break

    RF_Time = (arange(len(P)) - len(P)/2.)*dt;

    RF[RF_Time>t_trun]=0;
    RF = RF[RF_Time<=t_max];
    RF_Time = RF_Time[RF_Time<=t_max];

    if gauss_t != 0:
        gauss_sig = gauss_t/dt;
        x = linspace(-gauss_sig*4,gauss_sig*4,gauss_sig*8);
        Gauss_win = exp(-x**2/(2*gauss_sig**2));
        RF = convolve(RF,Gauss_win,'same');


    RF = RF[::-1];
    RF_Time = RF_Time[::-1];
    
    return RF_Time, RF