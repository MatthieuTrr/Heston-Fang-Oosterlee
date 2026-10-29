import numpy as np
from scipy.optimize import brentq

from src.blackscholes import black_scholes


def volatilite_implicite_put(prix, S0, K, r, T, vol_min=1e-6, vol_max=5.0):
    """
    Renvoie la volatilité implicite Black-Scholes d'un put européen.
    Lève une ValueError si le prix n'est pas inversible (hors des bornes
    d'absence d'arbitrage, ou pas de racine dans [vol_min, vol_max]).

      prix: prix du put
      S0: prix initial de l'actif
      K: strike
      r: taux sans risque
      T: maturité
    """
    borne_basse = max(K*np.exp(-r*T) - S0, 0.0)
    borne_haute = K*np.exp(-r*T)
    if not (borne_basse < prix < borne_haute):
        raise ValueError("Prix hors des bornes d'absence d'arbitrage : inversion impossible.")

    return brentq(lambda s: black_scholes(S0, K, r, T, s)[0] - prix, vol_min, vol_max)
