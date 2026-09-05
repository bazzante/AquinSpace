"""
orbital_lib.py
==============

Libreria di propagazione orbitale - versione 2.

Rispetto alla versione precedente, il motore delle perturbazioni non e' piu'
basato sulle Equazioni Planetarie di Lagrange (LPE) con sviluppo di
Kaula/Hansen (soluzione analitica del solo J2 per la Terra, sviluppo
simbolico enorme per la Luna): dopo il confronto numerico fatto in MATLAB
(vedi error_summary_table.csv - differenze dell'ordine di 1e-9..1e-6 sugli
elementi orbitali, sub-metrica sulla posizione) e' stato scelto come motore
definitivo il propagatore di Cowell (integrazione diretta di r, v) con
l'accelerazione perturbativa dovuta all'asfericita' del corpo centrale
calcolata con la formulazione di Pines (Pines, 1973), che a differenza
della formulazione classica in coordinate sferiche non ha singolarita'
matematiche ai poli ed e' quindi valida per qualunque orbita (equatoriale,
polare, ecc.) fino al grado/ordine scelto per lo sviluppo in armoniche
sferiche del potenziale.

Funzioni principali:
    car2kep, kep2car          conversioni elementi kepleriani <-> cartesiani
    kep2equinoctial            elementi equinoziali (solo per post-processing)
    osculating_orbit            baseline kepleriana (due corpi, non perturbata)
    load_gravity_coefficients_grace / _grail
                                 lettura e denormalizzazione dei coefficienti
                                 del campo gravitazionale (EGM2008 / GRGM900C)
    build_coeffs_matrix          tabella -> matrici Cmat, Smat indicizzabili
    accel_geopotential_pines     accelerazione perturbativa (Pines)
    eom_cowell_pines             equazioni del moto per l'integratore
    propagate_orbit_cowell_pines propagatore completo (Cowell + Pines)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

GRAVITATIONAL_CONSTANT = 6.67430e-20  # [km^3 kg^-1 s^-2]

# --- Corpi celesti: costanti di comodo per Terra e Luna --- #
EARTH_MU = 398600.4415          # [km^3/s^2]  (EGM2008)
EARTH_R_REF = 6378.1363         # [km]        (EGM2008)
EARTH_SIDEREAL_PERIOD = 86164.0905          # [s]
EARTH_ANGULAR_VELOCITY = 2 * np.pi / EARTH_SIDEREAL_PERIOD

MOON_MU = 4902.79996708864      # [km^3/s^2]  (GRGM900C)
MOON_R_REF = 1738.0             # [km]        (GRGM900C)
MOON_SIDEREAL_PERIOD = 27.321661 * 86400.0  # [s]
MOON_ANGULAR_VELOCITY = 2 * np.pi / MOON_SIDEREAL_PERIOD


# --- Classi --- #

class Body:
    """
    Classe per rappresentare un corpo celeste.

    Attributi:
        - nome: nome del corpo
        - position: posizione (vettore)
        - velocity: velocità (vettore)
        - mass: massa del corpo
        - radius: raggio del corpo
        - gravitational_parameter: parametro gravitazionale (G*M)
        - T_sidereal: periodo siderale del corpo
    """
    def __init__(self, nome, position, velocity, mass, radius, gravitational_parameter, T_sidereal):
        self.nome = nome
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.radius = radius
        self.gravitational_parameter = gravitational_parameter
        self.T_sidereal = T_sidereal
        self.angular_velocity = 2 * np.pi / T_sidereal


# --- Conversioni elementi orbitali <-> cartesiani (invariate) --- #

def car2kep(r, dotr, mu):
    """
    Calcolo degli elementi orbitali kepleriani da r e v.

    INPUT:
        r    : array-like [3], posizione
        dotr : array-like [3], velocità
        mu   : float, parametro gravitazionale

    OUTPUT:
        (a, ecc, I, w, Omega_hat, l, n, tperi)
    """
    r = np.array(r, dtype=float)
    dotr = np.array(dotr, dtype=float)

    h = np.cross(r, dotr)
    h_norm = np.linalg.norm(h)

    rho = np.linalg.norm(r)
    v2 = np.dot(dotr, dotr)

    en = 0.5 * v2 - mu / rho
    a = -mu / (2 * en)

    e_vec = np.cross(dotr, h) / mu - r / rho
    ecc = np.linalg.norm(e_vec)

    tol = 1e-12
    circular = ecc < tol

    if circular:
        e_vec = np.array([1.0, 0.0, 0.0])
        ecc = 0.0
    else:
        e_vec = e_vec / ecc

    if ecc > 1 + tol:
        raise ValueError('orbita iperbolica')

    if abs(ecc - 1) < tol:
        raise ValueError('orbita parabolica')

    cosI = h[2] / h_norm
    cosI = min(1.0, max(-1.0, cosI))
    I = np.arccos(cosI)

    if (I < tol) or (abs(I - np.pi) < tol):
        Omega_hat = 0.0
        lin_n = np.array([1.0, 0.0, 0.0])
    else:
        k_hat = np.array([0.0, 0.0, 1.0])
        lin_n_v = np.cross(k_hat, h)
        lin_n = lin_n_v / np.linalg.norm(lin_n_v)
        Omega_hat = np.arctan2(lin_n_v[1], lin_n_v[0])

    if circular:
        w = 0.0
    else:
        cosomega = np.dot(e_vec, lin_n)
        cosomega = min(1.0, max(-1.0, cosomega))
        w = np.arccos(cosomega)
        if e_vec[2] < 0:
            w = 2 * np.pi - w

    n = np.sqrt(mu / a**3)

    if circular:
        E_0 = 0.0
        l = 0.0
    else:
        cosE_0 = (1 - rho / a) / ecc
        cosE_0 = min(1.0, max(-1.0, cosE_0))
        sinE_0 = np.dot(r, dotr) / (ecc * a**2 * n)
        E_0 = np.arctan2(sinE_0, cosE_0)
        l = E_0 - ecc * sinE_0

    tperi = -l / n

    return a, ecc, I, w, Omega_hat, l, n, tperi


def kep2car(a, ecc, i, w, Omega, l, mu):
    """
    Da elementi kepleriani a vettori cartesiani (r, v).

    INPUT:
       a      semiasse maggiore [Km]
       ecc    eccentricità
       i      inclinazione [rad]
       w      argomento del pericentro [rad] (omega)
       Omega  longitudine del nodo ascendente [rad]
       l      anomalia media [rad]
       mu     parametro gravitazionale (G*M)

    OUTPUT:
       r_out, v_out  vettori posizione/velocità [3] (array NumPy)
    """
    tol = 1e-12
    max_iter = 50

    if ecc < 0.9:
        E = l
    else:
        E = np.pi

    for _ in range(max_iter):
        f_val = E - ecc * np.sin(E) - l
        f_der = 1 - ecc * np.cos(E)
        step = f_val / f_der
        E = E - step
        if abs(step) < tol:
            break

    x_orb = a * (np.cos(E) - ecc)
    y_orb = a * np.sqrt(1 - ecc**2) * np.sin(E)

    n = np.sqrt(mu / a**3)
    factor = (n * a) / (1 - ecc * np.cos(E))

    vx_orb = -factor * np.sin(E)
    vy_orb = factor * np.sqrt(1 - ecc**2) * np.cos(E)

    pos_orb = np.array([x_orb, y_orb, 0.0])
    vel_orb = np.array([vx_orb, vy_orb, 0.0])

    R_w = np.array([
        [np.cos(w), -np.sin(w), 0.0],
        [np.sin(w),  np.cos(w), 0.0],
        [0.0,        0.0,       1.0]
    ])
    R_i = np.array([
        [1.0, 0.0,        0.0],
        [0.0, np.cos(i), -np.sin(i)],
        [0.0, np.sin(i),  np.cos(i)]
    ])
    R_Om = np.array([
        [np.cos(Omega), -np.sin(Omega), 0.0],
        [np.sin(Omega),  np.cos(Omega), 0.0],
        [0.0,            0.0,           1.0]
    ])

    Rot_Matrix = R_Om @ R_i @ R_w

    r_out = Rot_Matrix @ pos_orb
    v_out = Rot_Matrix @ vel_orb

    return r_out, v_out


def kep2equinoctial(e, I, w, Omega, M):
    """
    Converte elementi orbitali classici (e, I, w, Omega, M) in elementi
    equinoziali (f, g, h, k, L) - SOLO per visualizzazione/post-processing,
    non cambia lo stato integrato dal propagatore.

    A differenza di (e, w, Omega), gli elementi (f, g, h, k) sono lisci e
    limitati anche per orbite quasi-circolari (e -> 0) e quasi-equatoriali
    (I -> 0): niente oscillazioni ampie/apparenti dovute alla singolarita'
    1/e delle equazioni di Lagrange riflessa nella definizione di omega.

    Definizioni (valide per orbite prograde, I < 180 deg):
        f = e*cos(w + Omega)
        g = e*sin(w + Omega)
        h = tan(I/2)*cos(Omega)
        k = tan(I/2)*sin(Omega)
        L = Omega + w + M          (longitudine media)

    NOTA su L: essendo una longitudine accumula nel tempo come M - se la si
    vuole rappresentare su piu' orbite senza il "dente di sega" del wrap,
    applicare np.unwrap(L) dopo la conversione. f, g, h, k non vanno mai
    srotolati: sono grandezze limitate per costruzione.

    INPUT: e, I, w, Omega, M in RADIANTI (scalari o array della stessa
           forma, es. le colonne di un DataFrame di elementi orbitali)
    OUTPUT: f, g, h, k, L nella stessa forma dell'input
    """
    e = np.asarray(e, dtype=float)
    I = np.asarray(I, dtype=float)
    w = np.asarray(w, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    M = np.asarray(M, dtype=float)

    f = e * np.cos(w + Omega)
    g = e * np.sin(w + Omega)
    h = np.tan(I / 2.0) * np.cos(Omega)
    k = np.tan(I / 2.0) * np.sin(Omega)
    L = Omega + w + M

    return f, g, h, k, L


def osculating_orbit(kepElements, mu):
    """
    Baseline kepleriana non perturbata (due corpi): propaga un periodo
    orbitale campionando 500 punti, senza alcun effetto di asfericita'.
    Utile come riferimento/visualizzazione rapida quando le perturbazioni
    sono disattivate.
    """
    a_0, ecc_0, i_0, w_0, Omega_0, l_0, n, _tperi = kepElements

    T_periodo = 2 * math.pi / n

    tempi = np.linspace(0, T_periodo, num=500)

    x_hist, y_hist, z_hist = [], [], []
    vx_hist, vy_hist, vz_hist = [], [], []
    orbEl_hist = []

    for t in tempi:
        l_t = l_0 + n * t
        r_t, v_t = kep2car(a_0, ecc_0, i_0, w_0, Omega_0, l_t, mu)

        x_hist.append(r_t[0]); y_hist.append(r_t[1]); z_hist.append(r_t[2])
        vx_hist.append(v_t[0]); vy_hist.append(v_t[1]); vz_hist.append(v_t[2])
        orbEl_hist.append(car2kep(r_t, v_t, mu))

    return x_hist, y_hist, z_hist, vx_hist, vy_hist, vz_hist, orbEl_hist


# --- Coefficienti del campo gravitazionale (EGM2008 / GRGM900C) --- #

def _denormalize_coefficients(l, m, C_norm, S_norm):
    """
    Denormalizzazione dei coefficienti del potenziale (Cap. 8, Vallado):

        factor = sqrt( (l-m)! * (2l+1) * k / (l+m)! ),  k = 1 se m=0, 2 se m>0

    Per gradi elevati (es. GRAIL, l fino a 900) i fattoriali diretti vanno
    in overflow molto prima (in MATLAB come in Python): qui il fattore si
    calcola tramite il logaritmo della funzione Gamma (math.lgamma), che
    e' numericamente stabile per qualunque l, m senza mai passare per un
    fattoriale esplicito.
    """
    l = np.asarray(l, dtype=float)
    m = np.asarray(m, dtype=float)
    C_norm = np.asarray(C_norm, dtype=float)
    S_norm = np.asarray(S_norm, dtype=float)

    factor = np.ones_like(C_norm)
    high_degree = l >= 2

    delta_k = np.where(m == 0, 1.0, 2.0)
    # log( (l-m)! ) - log( (l+m)! ) = lgamma(l-m+1) - lgamma(l+m+1)
    log_ratio = np.array([
        math.lgamma(li - mi + 1.0) - math.lgamma(li + mi + 1.0)
        for li, mi in zip(l, m)
    ])
    log_factor = 0.5 * (log_ratio + np.log((2.0 * l + 1.0) * delta_k))
    factor = np.where(high_degree, np.exp(log_factor), 1.0)

    return C_norm * factor, S_norm * factor


def load_gravity_coefficients_grace(filepath, degree):
    """
    Legge un file di coefficienti in formato .gfc (es. EGM2008 - GRACE) e
    restituisce un DataFrame con le colonne l, m, C, S (denormalizzati),
    troncato a l<=degree e m<=degree.

    Formato atteso (dopo l'intestazione, terminata dalla riga
    'end_of_head'): righe "gfc  l  m  C  S  sigma_C  sigma_S  ...".

    I coefficienti in questi file sono ordinati per grado l crescente: non
    appena si supera il grado richiesto la lettura si interrompe, invece di
    scorrere l'intero file (utile perche' EGM2008.gfc arriva a l=2190).
    """
    filepath = Path(filepath)
    ls, ms, Cs, Ss = [], [], [], []

    with open(filepath, 'r') as f:
        header_found = False
        for line in f:
            if not header_found:
                if 'end_of_head' in line:
                    header_found = True
                continue
            tokens = line.split()
            if len(tokens) < 5 or tokens[0] != 'gfc':
                continue
            l = int(float(tokens[1]))
            if l > degree:
                break  # file ordinato per l crescente: nulla di utile oltre questo punto
            m = int(float(tokens[2]))
            if m > degree:
                continue
            ls.append(l); ms.append(m)
            Cs.append(float(tokens[3])); Ss.append(float(tokens[4]))

    ls = np.array(ls); ms = np.array(ms)
    C_unnorm, S_unnorm = _denormalize_coefficients(ls, ms, Cs, Ss)

    return pd.DataFrame({'l': ls, 'm': ms, 'C': C_unnorm, 'S': S_unnorm})


def load_gravity_coefficients_grail(filepath, degree):
    """
    Legge un file di coefficienti in formato .tab (es. GRGM900C - GRAIL) e
    restituisce un DataFrame con le colonne l, m, C, S (denormalizzati),
    troncato a l<=degree e m<=degree.

    Formato atteso: prima riga = intestazione (raggio di riferimento, GM,
    grado/ordine massimo, ...), righe successive = "l, m, C, S, sigma_C,
    sigma_S" separate da virgola.

    I file GRAIL (es. GRGM900C) arrivano a grado 900 e possono superare i
    150-200 MB: i dati sono ordinati per l crescente, quindi il file viene
    letto a blocchi (chunk) e la lettura si interrompe non appena un intero
    blocco supera il grado richiesto, invece di caricare tutto in memoria.
    """
    filepath = Path(filepath)
    chunks = []
    for chunk in pd.read_csv(filepath, header=None, skiprows=1, usecols=[0, 1, 2, 3],
                              names=['l', 'm', 'C', 'S'], chunksize=200_000):
        chunks.append(chunk[(chunk['l'] <= degree) & (chunk['m'] <= degree)])
        if chunk['l'].min() > degree:
            break

    data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=['l', 'm', 'C', 'S'])
    data = data.sort_values(['l', 'm']).reset_index(drop=True)

    C_unnorm, S_unnorm = _denormalize_coefficients(
        data['l'].to_numpy(), data['m'].to_numpy(), data['C'].to_numpy(), data['S'].to_numpy()
    )

    return pd.DataFrame({'l': data['l'].to_numpy(), 'm': data['m'].to_numpy(),
                          'C': C_unnorm, 'S': S_unnorm})


def save_gravity_coefficients_csv(coeffs_table, filepath):
    """
    Salva una tabella di coefficienti (l, m, C, S) già denormalizzata in un
    CSV compatto - pensato per essere generato UNA VOLTA sola in locale con
    load_gravity_coefficients_grace/_grail troncando a un grado ragionevole
    (es. 50-60), cosi' il sito web non deve portarsi dietro i file originali
    EGM2008/GRAIL (rispettivamente qualche MB e centinaia di MB, il secondo
    oltre il limite di 100MB per file di GitHub): con l<=60 il CSV pesa
    poche decine di KB.
    """
    coeffs_table.to_csv(filepath, index=False)


def load_gravity_coefficients_csv(filepath, degree=None):
    """
    Legge una tabella di coefficienti già denormalizzata e (tipicamente) già
    troncata, salvata con save_gravity_coefficients_csv. Se degree è
    specificato, applica comunque un ulteriore filtro l<=degree, m<=degree
    (utile per usare un grado più basso di quello del file senza doverlo
    rigenerare).
    """
    data = pd.read_csv(filepath)
    if degree is not None:
        data = data[(data['l'] <= degree) & (data['m'] <= degree)].reset_index(drop=True)
    return data


def build_coeffs_matrix(coeffs_table, degree):
    """
    Converte coeffs_table (DataFrame con colonne l, m, C, S) in due matrici
    (degree+1)x(degree+1) indicizzabili in O(1) come Cmat[l, m], Smat[l, m].

    Va chiamata UNA SOLA VOLTA in fase di setup, fuori dall'integratore, per
    evitare ricerche ripetute dentro il ciclo caldo dell'ODE. Le righe l=0,1
    restano a zero per costruzione: il termine di punto materiale (-mu*r/r^3)
    NON e' incluso qui e va sommato a parte, come nella versione MATLAB.
    """
    Cmat = np.zeros((degree + 1, degree + 1))
    Smat = np.zeros((degree + 1, degree + 1))

    for l, m, C, S in zip(coeffs_table['l'], coeffs_table['m'],
                           coeffs_table['C'], coeffs_table['S']):
        l = int(l); m = int(m)
        if 2 <= l <= degree and m <= l:
            Cmat[l, m] = C
            Smat[l, m] = S

    return Cmat, Smat


# --- Accelerazione perturbativa: formulazione di Pines --- #

def accel_geopotential_pines(r_vec, theta_GMST, Cmat, Smat, degree, mu, R_ref):
    """
    Calcola l'accelerazione perturbativa dovuta all'asfericita' del corpo
    centrale usando la formulazione di Pines (Pines, 1973; "Uniform
    Representation of the Gravitational Potential and its Derivatives",
    AIAA Journal, Vol. 11, No. 11).

    A differenza della formulazione classica (che passa per
    latitudine/longitudine ed ha una singolarita' matematica ai poli),
    questa formulazione lavora SOLO con i coseni direttori S, T, U del
    vettore posizione in assi corpo-fissi (mai sin/cos di angoli, mai
    divisioni per cos(phi) o per (x^2+y^2)) - quindi e' priva di
    singolarita' in qualunque punto dell'orbita, poli inclusi.

    INPUT:
        r_vec        posizione inerziale [3] km
        theta_GMST   angolo di rotazione siderale corrente [rad]
        Cmat, Smat   matrici coefficienti da build_coeffs_matrix
        degree       grado massimo (= ordine massimo, modello quadrato)
        mu, R_ref    parametri del corpo centrale

    OUTPUT:
        a_pert       accelerazione perturbativa inerziale [3] km/s^2
    """
    r_vec = np.asarray(r_vec, dtype=float)

    # --- 1. Rotazione in assi corpo-fissi e coseni direttori ---
    ct, st = np.cos(theta_GMST), np.sin(theta_GMST)
    rnp = np.array([
        [ct,  st, 0.0],
        [-st, ct, 0.0],
        [0.0, 0.0, 1.0]
    ])  # rotazione inerziale -> corpo-fisso

    R_F = rnp @ r_vec
    RMAG = np.linalg.norm(R_F)

    S = R_F[0] / RMAG
    T = R_F[1] / RMAG
    U = R_F[2] / RMAG  # coseno direttore z (= sin(latitudine), mai usato come angolo)

    # --- 2. Funzioni di Legendre derivate (ANM), ricorsione non singolare ---
    ND = degree + 2
    ANM = np.zeros((ND + 1, ND + 1))
    ANM[0, 0] = 1.0

    for M in range(0, ND + 1):
        if M != 0:
            ANM[M, M] = (2 * M - 1) * ANM[M - 1, M - 1]
        if M != ND:
            ANM[M + 1, M] = (2 * M + 1) * U * ANM[M, M]
        if M < ND:
            for N in range(M + 2, ND + 1):
                ANM[N, M] = ((2 * N - 1) * U * ANM[N - 1, M] - (N + M - 1) * ANM[N - 2, M]) / (N - M)

    # --- 3. Potenze complesse (S+iT)^m via ricorsione (mai atan2/cos/sin) ---
    RM = np.zeros(degree + 2)
    IM = np.zeros(degree + 2)
    RM[0], IM[0] = 0.0, 0.0   # sentinella m = -1
    RM[1], IM[1] = 1.0, 0.0   # m = 0
    for M in range(1, degree + 1):
        idx = M + 1
        RM[idx] = S * RM[idx - 1] - T * IM[idx - 1]
        IM[idx] = S * IM[idx - 1] + T * RM[idx - 1]

    # --- 4. Somma principale (G1..G4) ---
    RHO = mu / (R_ref * RMAG)
    RHOP = R_ref / RMAG
    G1 = G2 = G3 = G4 = 0.0

    for N in range(0, degree + 1):
        G1T = G2T = G3T = G4T = 0.0

        for M in range(0, N + 1):
            idx = M + 1

            C_lm = Cmat[N, M]
            S_lm = Smat[N, M]

            DNM = C_lm * RM[idx] + S_lm * IM[idx]
            ENM = C_lm * RM[idx - 1] + S_lm * IM[idx - 1]
            FNM = S_lm * RM[idx - 1] - C_lm * IM[idx - 1]

            G1T += ANM[N, M] * M * ENM
            G2T += ANM[N, M] * M * FNM
            G3T += ANM[N, M + 1] * DNM
            G4T += ((N + M + 1) * ANM[N, M] + U * ANM[N, M + 1]) * DNM

        RHO = RHOP * RHO
        G1 += RHO * G1T
        G2 += RHO * G2T
        G3 += RHO * G3T
        G4 += RHO * G4T

    # --- 5. Accelerazione in assi corpo-fissi, poi rotazione a inerziale ---
    G_F = np.array([G1 - G4 * S, G2 - G4 * T, G3 - G4 * U])

    a_pert = rnp.T @ G_F
    return a_pert


# --- Equazioni del moto (Cowell) e propagatore completo --- #

def eom_cowell_pines(t, state, Cmat, Smat, degree, mu, R_ref, w_body):
    """
    Equazioni del moto per l'integrazione diretta (Cowell) di posizione e
    velocità, con l'accelerazione perturbativa calcolata via
    accel_geopotential_pines - priva di singolarita' ai poli (adatta anche
    per orbite polari, es. scenario LRO).

    state = [rx, ry, rz, vx, vy, vz]
    """
    r_vec = state[0:3]
    v_vec = state[3:6]

    r = np.linalg.norm(r_vec)
    theta_GMST = w_body * t

    a_twobody = -mu * r_vec / r**3
    a_pert = accel_geopotential_pines(r_vec, theta_GMST, Cmat, Smat, degree, mu, R_ref)

    a_tot = a_twobody + a_pert

    return np.concatenate([v_vec, a_tot])


def propagate_orbit_cowell_pines(kepElements, mu, R_ref, w_body, Cmat, Smat, degree,
                                  num_orbits=100, num_points=50000,
                                  rtol=1e-12, atol=1e-12, method='RK45'):
    """
    Propagatore orbitale completo: integra le equazioni di Cowell con
    l'accelerazione perturbativa di Pines e riconverte la traiettoria
    cartesiana in elementi orbitali kepleriani, in un formato analogo a
    quello prodotto dalla vecchia propagate_perturbed_orbit (cosi' il
    passaggio in interfaccia.py richiede solo di cambiare la funzione
    chiamata, non il modo in cui il risultato viene consumato).

    INPUT:
        kepElements : tupla (a, ecc, I, w, Omega, M, n, tperi) iniziale
                      (uscita di car2kep)
        mu, R_ref   : parametri del corpo centrale
        w_body      : velocita' angolare di rotazione del corpo centrale
                      [rad/s] (es. EARTH_ANGULAR_VELOCITY, MOON_ANGULAR_VELOCITY)
        Cmat, Smat  : matrici coefficienti da build_coeffs_matrix
        degree      : grado massimo del potenziale
        num_orbits  : numero di orbite kepleriane da propagare
        num_points  : numero di punti temporali campionati
        rtol, atol  : tolleranze dell'integratore
        method      : metodo di integrazione di solve_ivp

    OUTPUT:
        x_hist, y_hist, z_hist, vx_hist, vy_hist, vz_hist : liste di float,
            traiettoria cartesiana campionata
        t_sol       : array dei tempi campionati [s]
        elements_hist : array (6, num_points) con le righe
            [a, ecc, I, w, Omega, M] ricalcolate ad ogni istante
    """
    a_0 = kepElements[0]
    r0, v0 = kep2car(*kepElements[0:6], mu)

    T_periodo = 2 * math.pi / np.sqrt(mu / a_0**3)
    t_finale = T_periodo * num_orbits

    t_span = (0.0, t_finale)
    tempi_eval = np.linspace(0.0, t_finale, num_points)

    y0 = np.concatenate([r0, v0])

    sol = solve_ivp(
        fun=eom_cowell_pines,
        t_span=t_span,
        y0=y0,
        t_eval=tempi_eval,
        args=(Cmat, Smat, degree, mu, R_ref, w_body),
        method=method,
        rtol=rtol, atol=atol
    )

    if not sol.success:
        raise RuntimeError("Integrazione fallita: " + sol.message)

    x_hist, y_hist, z_hist = sol.y[0], sol.y[1], sol.y[2]
    vx_hist, vy_hist, vz_hist = sol.y[3], sol.y[4], sol.y[5]

    N = len(sol.t)
    elements_hist = np.zeros((6, N))
    for k in range(N):
        r_k = sol.y[0:3, k]
        v_k = sol.y[3:6, k]
        a_k, ecc_k, I_k, w_k, Omega_k, M_k, _n_k, _tperi_k = car2kep(r_k, v_k, mu)
        elements_hist[:, k] = [a_k, ecc_k, I_k, w_k, Omega_k, M_k]

    return (x_hist.tolist(), y_hist.tolist(), z_hist.tolist(),
            vx_hist.tolist(), vy_hist.tolist(), vz_hist.tolist(),
            sol.t, elements_hist)


def step_cowell_pines(state, t0, dt, Cmat, Smat, degree, mu, R_ref, w_body,
                       n_substeps=5, rtol=1e-10, atol=1e-10, method='RK45'):
    """
    Integra le equazioni di Cowell (con perturbazione di Pines) su un
    singolo intervallo [t0, t0+dt], campionando n_substeps punti intermedi.

    Pensata per la propagazione "ad oltranza" (un intervallo per ogni
    aggiornamento dell'interfaccia, finche' l'utente non preme STOP) invece
    del precalcolo in blocco di propagate_orbit_cowell_pines: dato che il
    propagatore di Cowell + Pines e' generico e non richiede di ripartire
    dagli elementi kepleriani, riprende semplicemente dall'ultimo stato
    integrato.

    INPUT:
        state       stato corrente [rx,ry,rz,vx,vy,vz]
        t0          tempo corrente [s]
        dt          durata dell'intervallo da integrare [s]
        n_substeps  numero di punti intermedi campionati nell'intervallo

    OUTPUT:
        new_state   stato al tempo t0+dt
        t_eval      array dei tempi campionati (t0 escluso, t0+dt incluso)
        r_hist      array (n_substeps, 3) delle posizioni campionate
        v_hist      array (n_substeps, 3) delle velocità campionate
    """
    state = np.asarray(state, dtype=float)
    t_eval = np.linspace(t0, t0 + dt, n_substeps + 1)[1:]

    sol = solve_ivp(
        fun=eom_cowell_pines,
        t_span=(t0, t0 + dt),
        y0=state,
        t_eval=t_eval,
        args=(Cmat, Smat, degree, mu, R_ref, w_body),
        method=method,
        rtol=rtol, atol=atol
    )

    if not sol.success:
        raise RuntimeError("Integrazione fallita: " + sol.message)

    new_state = sol.y[:, -1]
    r_hist = sol.y[0:3, :].T
    v_hist = sol.y[3:6, :].T

    return new_state, sol.t, r_hist, v_hist
