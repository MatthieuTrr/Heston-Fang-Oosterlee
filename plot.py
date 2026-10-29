from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from src.fangoosterlee import P
from src.blackscholes import black_scholes
from src.implied_vol import volatilite_implicite_put


def plot_convergence_bs(S0, K, T, r, kappa, lambdA, theta, sig_0, N):
    # Valeurs de sig_sig : de 1.0 (élevé, Feller violée) jusqu'à 0.001 (proche de 0)
    # On s'arrête à 1e-3 : en dessous, l'annulation numérique dans (h-D)/sig_sig**2 dégrade le résultat
    sig_sig_grid = np.linspace(1.0, 0.001, 100)

    # Prix Heston (put) pour chaque sig_sig, les autres paramètres restant ceux de la cellule des paramètres
    prix_heston = np.array([
        P(S0, K, T, r, kappa, lambdA, theta, sig_0, ss, 0, N)
        for ss in sig_sig_grid
    ])

    # Asymptote : prix Black-Scholes (put) de volatilité sig_0 (valable car sig_0**2 = theta et lambdA = 0)
    put_bs, _ = black_scholes(S0, K, r, T, sig_0)

    plt.figure(figsize=(8, 5))
    plt.plot(sig_sig_grid, np.abs(prix_heston-put_bs), label="Prix Heston (COS)")
    #plt.axhline(put_bs, color="red", linestyle="--", label=f"Asymptote : prix Black-Scholes = {put_bs:.4f}")
    plt.gca().invert_xaxis()   # on lit le graphique de sig_sig élevé vers 0
    plt.xlabel(r"$\sigma_\sigma$ (volatilité de la volatilité)")
    plt.ylabel("Prix du put")
    plt.title(r"Convergence du prix Heston vers Black-Scholes quand $\sigma_\sigma \to 0$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Écart avec B-S pour sig_sig = {sig_sig_grid[-1]:.3f} : {prix_heston[-1] - put_bs:.2e}")


def plot_smile_rho(S0, r, T, kappa, lambdA, theta, sig_0, sig_sig,
                   rhos=(-0.7, 0, 0.7), N=1024, L=12, n_strikes=31):
    """
    Smile de volatilité implicite (put) pour plusieurs valeurs de rho :
    influence de la corrélation sur la forme du smile.
    """
    strikes_rho = np.linspace(0.85*S0, 1.15*S0, n_strikes)

    plt.figure()

    for rho_test in rhos:
        vols = []
        echecs = 0

        for strike in strikes_rho:
            prix = P(S0, strike, T, r, kappa, lambdA, theta, sig_0, sig_sig, rho_test, N, L)

            try:
                vol = volatilite_implicite_put(prix, S0, strike, r, T)
            except ValueError:
                vol = np.nan
                echecs += 1

            vols.append(vol)

        plt.plot(
            strikes_rho, 100*np.array(vols),
            label=f"rho = {rho_test}"
        )

        print(f"rho={rho_test} | Inversions echouees : {echecs}")

    plt.xlabel("Strike K")
    plt.ylabel("Volatilite implicite (%)")
    plt.title("Influence de la correlation sur le smile")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_temps_calcul(S0, K, T, r, kappa, lambdA, theta, sig_0, sig_sig, rho):
    Ns = [32, 64, 128, 256, 512, 1024, 2048]
    temps_moyens = []

    for n in Ns:
        debut = perf_counter()
        for _ in range(100):
            P(S0, K, T, r, kappa, lambdA, theta, sig_0, sig_sig, rho, n)
        temps_ms = 1000 * (perf_counter() - debut) / 100
        temps_moyens.append(temps_ms)
        print(f"N={n:4d} | Temps moyen={temps_ms:.4f} ms")

    plt.figure()
    plt.plot(Ns, temps_moyens, "o-")
    plt.xlabel("N")
    plt.ylabel("Temps moyen par prix (ms)")
    plt.title("Temps de calcul de Heston-COS")
    plt.grid(alpha=0.3)
    plt.show()
