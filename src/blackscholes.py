import numpy as np
from scipy.stats import norm


def black_scholes(S0, K, r, T, sigma):
    """
    Renvoie (prix_put, prix_call) sans dividendes.
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 et K doivent être positifs.")

    if T < 0 or sigma < 0:
        raise ValueError("T et sigma doivent être positifs ou nuls.")

    # À l'échéance
    if T == 0:
        return max(K - S0, 0.0), max(S0 - K, 0.0)

    # Cas sans volatilité
    if sigma == 0:
        strike_actualise = K * np.exp(-r*T)
        return (
            max(strike_actualise - S0, 0.0),
            max(S0 - strike_actualise, 0.0)
        )

    d1 = (
        np.log(S0 / K) + (r + 0.5*sigma**2)*T
    ) / (sigma*np.sqrt(T))

    d2 = d1 - sigma*np.sqrt(T)

    call_bs = (
        S0*norm.cdf(d1)
        - K*np.exp(-r*T)*norm.cdf(d2)
    )

    put_bs = (
        K*np.exp(-r*T)*norm.cdf(-d2)
        - S0*norm.cdf(-d1)
    )

    return put_bs, call_bs
