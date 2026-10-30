import numpy as np


def trunc_range(r,T,kappa,lambd,theta,sig_0,sig_sig,rho,L,x):
    """
    Calcule les cumulants  puis les bornes de l'intervalle de troncature.

      r: taux sans risque
      T: maturité
      kappa: vitesse de retour à la moyenne
      lambd: coefficient multiplicateur du prix du risque de volatilité
      theta: variance moyenne de long terme
      sig_0: volatilité initiale
      sig_sig: volatilité de la volatilité
      rho: corrélation entre les movements browniens
      L: largeur de l'intervalle de troncature
      x: ln(S0/K), avec S0 la valeur initiale du sous-jacent
    """

    kappa_star=kappa+lambd
    theta_star=kappa*theta/(kappa+lambd)

    c_1=r*T+ (1-np.exp(-kappa_star*T))*(theta_star-sig_0**2)/(2*kappa_star) -(1/2)*theta_star*T
    # par souci de lisibilité on sépare en 5 le calcul de c_2.
    A=sig_sig*T*kappa_star*np.exp(-kappa_star*T)*(sig_0**2-theta_star)*(8*kappa_star*rho-4*sig_sig)
    B=kappa_star*rho*sig_sig*(1-np.exp(-kappa_star*T))*(16*theta_star-8*sig_0**2)
    C=2*theta_star*kappa_star*T*(-4*kappa_star*rho*sig_sig+sig_sig**2+4*kappa_star**2)
    D=sig_sig**2*((theta_star-2*sig_0**2)*np.exp(-2*kappa_star*T)+theta_star*(6*np.exp(-kappa_star*T)-7)+2*sig_0**2)
    E=8*kappa_star**2*(sig_0**2-theta_star)*(1-np.exp(-kappa_star*T))
    c_2=(1/(8*kappa_star**3))*(A+B+C+D+E)

    a=x+c_1-L*np.sqrt(np.abs(c_2))
    b=x+c_1+L*np.sqrt(np.abs(c_2))
    return a, b, c_1, c_2
