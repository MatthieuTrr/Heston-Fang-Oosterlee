import time

import numpy as np
import matplotlib.pyplot as plt

from src.fangoosterlee import P
from src.blackscholes import black_scholes
from src.implied_vol import volatilite_implicite_put


def plot_convergence_bs(S0, K, T, r, kappa, lambdA, theta, sig_0, N):
    
    sig_sig_grid = np.linspace(1.0, 0.001, 100)

    prix_heston = np.array([
        P(S0, K, T, r, kappa, lambdA, theta, sig_0, ss, 0, N)
        for ss in sig_sig_grid
    ])

    put_bs, _ = black_scholes(S0, K, r, T, sig_0)

    plt.figure(figsize=(8, 5))
    plt.plot(sig_sig_grid, np.abs(prix_heston-put_bs), label="Prix Heston (COS)")
    #plt.axhline(put_bs, color="red", linestyle="--", label=f"Asymptote : prix Black-Scholes = {put_bs:.4f}")
    plt.gca().invert_xaxis()   # on lit le graphique de sig_sig élevé vers 0
    plt.xlabel(r"$\sigma_\sigma$ (volatilité de la volatilité)")
    plt.ylabel("L'écart entre prix Black-Scholes et prix Heston")
    plt.title(r"Convergence du prix Heston vers Black-Scholes quand $\sigma_\sigma \to 0$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Écart avec B-S pour sig_sig = {sig_sig_grid[-1]:.3f} : {prix_heston[-1] - put_bs:.2e}")


def plot_smile_rho(S0, r, T, kappa, lambdA, theta, sig_0, sig_sig,
                   rhos=(-0.7, 0, 0.7), N=1024, L=12, n_strikes=31):
   
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


def plot_temps_calcul(S0, K, T, r, kappa, lambdA, theta,
                      sig_0, sig_sig, rho):
    Ns = [32, 64, 128, 256, 512, 1024, 2048]
    temps_moyens = []

    for n in Ns:
        debut = time.time()

        for i in range(100):
            P(S0, K, T, r, kappa, lambdA, theta,sig_0, sig_sig, rho, n)

        temps_ms = 1000 * (time.time() - debut) / 100
        temps_moyens.append(temps_ms)

        print(f"N={n:4d} | Temps moyen={temps_ms:.4f} ms")

    plt.figure()
    plt.plot(Ns, temps_moyens, "o-")
    plt.xlabel("N")
    plt.ylabel("Temps moyen par prix (ms)")
    plt.title("Temps de calcul de Heston-COS")
    plt.grid(alpha=0.3)
    plt.show()

def smile_symetrique():
    S0_smile = 100
    r_smile = 0.03
    T_smile = 1.0
    kappa_smile = 2.0
    lambdA_smile = 0.0
    theta_smile = 0.04       # Volatilité moyen terme de 20%
    sig_0_smile = 0.2        # Volatilité initiale de 20%
    
    
    rho_smile = 0.0          # Corrélation nulle -> Symétrie parfaite autour de S0
    sig_sig_smile = 0.5      # Vol of Vol plus forte -> Courbure / "U" bien marqué
    
    N_smile = 256
    L_smile = 12
    
    # Grille de Strikes K (de 70 à 130 autour du prix SPOT S0=100)
    K_grid = np.linspace(70, 130, 61)
    vols_implicites = []
    
    for K_val in K_grid:
        price_heston = P(S0_smile, K_val, T_smile, r_smile, kappa_smile, 
                         lambdA_smile, theta_smile, sig_0_smile, sig_sig_smile, 
                         rho_smile, N_smile, L_smile, call="False")
        
        vol_imp = implied_volatility(price_heston, S0_smile, K_val, r_smile, T_smile, option_type="put")
        vols_implicites.append(vol_imp * 100) # Conversion en %
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_grid, vols_implicites, 'b-', linewidth=2, label=rf"Heston ($\rho = {rho_smile}$, $\sigma_\sigma = {sig_sig_smile}$)")
    plt.axvline(S0_smile, color='gray', linestyle='--', alpha=0.7, label=f"Spot $S_0 = {S0_smile}$ (ATM)")
    
    plt.xlabel("Strike $K$")
    plt.ylabel("Volatilité Implicite (%)")
    plt.title(f"Smile de Volatilité Implicite Symétrique dans le modèle de Heston ($T = {T_smile}$ an)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

def plot_smile_volvol(S0, r, T, kappa, lambdA, theta, sig_0,
                     sig_sigs=(0.2, 0.4, 0.5751, 0.8),
                     N=1024, L=12):
    Ks = np.linspace(0.7*S0, 1.3*S0, 61)

    fig, ax = plt.subplots(figsize=(6, 4))

    for ss in sig_sigs:
        vols = []

        for strike in Ks:
            prix = P(
                S0, strike, T, r,
                kappa, lambdA, theta, sig_0, ss, 0,
                N, L
            )

            vol = volatilite_implicite_put(prix, S0, strike, r, T)
            vols.append(vol)

        ax.plot(Ks, 100*np.array(vols), label=f"σ_σ = {ss:g}")

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatilité implicite (%)")
    ax.set_title("Effet de la vol-of-vol sur le smile (ρ=0)")
    ax.grid(alpha=0.3)
    ax.legend()

def plot_convergence_N():
    #Étude de la convergence en fonction de N (à L fixe et grand)
    N_values = np.array([4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256])
    L_fixed = 16  

    errors_N = []
    for N_val in N_values:
        p_cos = P(S0, K, T, r, kappa, lambdA, theta, sig_0, sig_sig, rho, N=int(N_val), L=L_fixed)
        # Conversion explicite en float pour éviter les conflits avec les types mpf
        errors_N.append(float(abs(p_cos - prix_exact)))

    errors_N = np.array(errors_N)

    # Ajustement linéaire en échelle semi-log (log(erreur) = a * N + b)
    mask = errors_N > 1e-12
    coeffs = np.polyfit(N_values[mask], np.log(errors_N[mask]), 1)
    y_theo = np.exp(coeffs[1]) * np.exp(coeffs[0] * N_values)

    # Tracé
    plt.figure(figsize=(8, 5))
    plt.semilogy(N_values, errors_N, 'o-', color='b', label="Erreur absolue |P_COS - P_exact|")
    plt.semilogy(N_values, y_theo, '--', color='darkorange', label=f"Tendance théorique $\propto e^{{{coeffs[0]:.2f} N}}$")
    plt.xlabel("N (Nombre de termes de Fourier-Cosinus)")
    plt.ylabel("Erreur absolue (échelle log)")
    plt.title(f"Ordre de convergence de la méthode COS selon N (L = {L_fixed})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()

def plot_convergence_L():
    #Étude de la convergence en fonction de L (à N fixe et grand)
    L_values = np.linspace(2, 16, 15)  
    N_fixed = 512  

    errors_L = []
    for L_val in L_values:
        p_cos = P(S0, K, T, r, kappa, lambdA, theta, sig_0, sig_sig, rho, N=N_fixed, L=L_val)
        # Conversion explicite en float pour éviter les conflits avec les types mpf
        errors_L.append(float(abs(p_cos - prix_exact)))

    errors_L = np.array(errors_L)

    # Ajustement linéaire en échelle semi-log pour la tendance e^{-\alpha L}
    mask_L = errors_L > 1e-12
    coeffs_L = np.polyfit(L_values[mask_L], np.log(errors_L[mask_L]), 1)
    y_theo_L = np.exp(coeffs_L[1]) * np.exp(coeffs_L[0] * L_values)

    # Tracé
    plt.figure(figsize=(8, 5))
    plt.semilogy(L_values, errors_L, 's-', color='r', label="Erreur absolue |P_COS - P_exact|")
    plt.semilogy(L_values, y_theo_L, '--', color='darkblue', label=f"Tendance théorique $\propto e^{{{coeffs_L[0]:.2f} L}}$")
    plt.xlabel("L (Largeur de l'intervalle de troncature)")
    plt.ylabel("Erreur absolue (échelle log)")
    plt.title(f"Ordre de convergence de la méthode COS selon L (N = {N_fixed})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()
