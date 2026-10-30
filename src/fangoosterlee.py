import numpy as np

from src.cumul import trunc_range
from src.carac import carac_heston
from src.U_k import chi, psi


def P(S0,K,T,r,kappa,lambdA,theta,sig_0,sig_sig,rho,N,L=12, call="False"):
  """
    renvoie le prix d'un call ou d'un put.

      S0: prix initial de l'actif,
      K: strike
      T: maturité,
      r: taux sans risque
      kappa: vitesse de retour à la moyenne
      lambdA: prix du risque de volatilité
      theta: variance moyenne de long terme
      sig_0:volatilité initiale
      sig_sig:volatilité de volatilité
      rho: corrélation des mouvements browniens
      N: nombre de coefficients cosinus gardés
      L: largeur de troncature
  """
  x=np.log(S0/K)
  k=np.arange(N)
  a,b,c_1,c_2=trunc_range(r,T,kappa,lambdA,theta,sig_0,sig_sig,rho,L,x)
  omega=k*np.pi/(b-a)
  kappa_star=kappa+lambdA
  theta_star=kappa*theta/kappa_star
  phi=carac_heston(omega,r,T,kappa_star,theta_star,sig_0,sig_sig,rho)
  Chi=chi(omega,a,a,0.0)
  Psi=psi(omega,a,a,0.0)
  U_k=2/(b-a)*(Psi-Chi)
  fact_e=np.exp(1j*omega*(x-a))
  termes=phi*U_k*fact_e
  poids=np.ones_like(omega)
  poids[0]=0.5
  somme=np.sum(poids*termes)
  prix_put=K*np.exp(-r*T)*np.real(somme)
  if call=="True":
    return prix_put+S0-K*np.exp(-r*T)
  else:
    return prix_put
