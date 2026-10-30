import numpy as np


def carac_heston(omega, r, T, ks, ts, sig_0, sig_sig, rho):
  """
  Renvoie les valeurs de la fonction caractèristique sur la grille de valeurs omega
  omega: grille de valeurs d'abscisse
  r: taux sans risque
  T: maturité
  ks: kappa_star
  ts: theta_star
  sig_0: volatilité initiale
  sig_sig: volatilité de volatilité
  rho: corrélation des mouvemtns browniens

  """
  h = ks- 1j * rho * sig_sig * omega
  D = np.sqrt(h**2 + (omega**2+ 1j * omega)* sig_sig**2)
  G = (h-D)/(h+D)
  Exp1 = 1j * omega * r * T + ((sig_0**2)/(sig_sig**2))*((1-np.exp(-D*T))/(1-G*np.exp(-D*T))) * (h-D)

  Exp2 = (ks*ts/sig_sig**2) * (T*(h-D) - 2*np.log((1-G*np.exp(-D*T))/(1-G)))

  return np.exp(Exp1 + Exp2)
