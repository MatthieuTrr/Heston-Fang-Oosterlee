import numpy as np


def chi(omega, a,p,q) :
  """
  Renvoie la valeur de chi_k(p,q) pour le calcul des V_k

    omega: grille de valeurs
    a: borne basse de l'intervalle de troncature
    p: 0 pour un call, a pour un put
    q: b pour un call, 0 pour un put
  """

  chi_q = np.exp(q)*(np.cos(omega*(q-a)) +omega * np.sin(omega*(q-a)))
  chi_p = np.exp(p)*(np.cos(omega*(p-a)) +omega * np.sin(omega*(p-a)))

  return (chi_q-chi_p)/(1+omega**2)


def psi(omega,a,p,q) :
  """
  Renvoie la valeur des psi_k(p,q) pour le calcul des V_k

    omega: grille de valeurs
    a: borne basse de l'intervalle de troncature
    p: 0 pour un call, a pour un put
    q: b pour un call, 0 pour un put
  """

  res = np.empty_like(omega)

  res[0] = q-p
  w = omega[1:]
  res[1:] = (np.sin(w*(q-a)) - np.sin(w*(p-a)))/w

  return res
