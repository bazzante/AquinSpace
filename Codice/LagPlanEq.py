import numpy as np

def lpe_derivatives_j2(t, state, mu=398600, J2 = 1.08263e-3, R_eq = 6378.137):
    """
    Calcola le derivate degli elementi orbitali (Equazioni Planetarie di Lagrange)
    considerando solo la perturbazione zonale J2.
    """
    # Scompattiamo lo stato attuale
    a, e, i, w, Omega, M = state

    # Calcolo del moto medio base (che aggiungeremo alla fine a dM_dt)
    # NOTA: Nel tuo file J2 di MATLAB mancava la parte "n", ma serve per propagare!
    n = np.sqrt(mu / a**3)

    # Traduzione esatta dal tuo output MATLAB
    t2 = np.cos(i)
    t3 = np.sin(i)
    t4 = R_eq**2
    t5 = e**2
    t7 = 1.0 / a**3
    t8 = 1.0 / a**5
    t6 = t3**2
    t9 = t5 - 1.0
    t10 = mu * t7
    t11 = t6 * (3.0 / 2.0)
    t12 = 1.0 / t9**2
    t13 = 1.0 / np.sqrt(t10)
    t14 = t11 - 1.0
    
    da_dt = 0.0
    de_dt = 0.0
    di_dt = 0.0
    
    dw_dt = (J2 * mu * t4 * t8 * t12 * t13 * t14 * (-3.0 / 2.0) + 
             J2 * mu * t2 * t3 * t4 * t8 * t12 * t13 * (1.0 / np.tan(i)) * (3.0 / 2.0))
             
    dOmega_dt = J2 * mu * t2 * t4 * t8 * t12 * t13 * (-3.0 / 2.0)
    
    # Derivata dell'anomalia media: aggiungiamo il moto medio 'n' alla perturbazione
    dM_dt = n + J2 * mu * t4 * t8 * 1.0 / (-t9)**(3.0 / 2.0) * t13 * t14 * (-3.0 / 2.0)

    # Restituiamo le derivate come una lista
    return [da_dt, de_dt, di_dt, dw_dt, dOmega_dt, dM_dt]