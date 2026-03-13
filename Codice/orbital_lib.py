import pandas as pd  # Importazione standard per pandas
import math
import numpy as np
import os
from scipy.integrate import solve_ivp

gravitatioal_coefficient = 6.67430e-20  # Costante gravitazionale (in unità SI) 

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
        self.nome = nome                                        # Nome del corpo
        self.position = position                                # Posizione (vettore 3D)
        self.velocity = velocity                                # Velocità (vettore 3D)
        self.mass = mass                                        # Massa del corpo
        self.radius = radius                                    # Raggio del corpo
        self.gravitational_parameter = gravitational_parameter  # Parametro gravitazionale  
        self.T_sidereal = T_sidereal                            # tempo in secondi per compiere un giro di 360 gradi
        self.angular_velocity = 2*np.pi/T_sidereal              # Velocità angolare (radianti al secondo) calcolata da T_sidereal

# --- Incolla qui le tue funzioni --- #

def car2kep(r, dotr, mu):
    """
    Calcolo degli elementi orbitali kepleriani da r e v.
    
    INPUT:
        r    : array-like [3], posizione
        dotr : array-like [3], velocità
        mu   : float, parametro gravitazionale
    
    OUTPUT:
        Restituisce una tupla con:
        (n, ecc, I, w, Omega_hat, l, a, tperi)
    """
    
    # Assicuriamoci che gli input siano numpy array float
    r = np.array(r, dtype=float)
    dotr = np.array(dotr, dtype=float)
    
    # Momento angolare
    h = np.cross(r, dotr)
    h_norm = np.linalg.norm(h)

    rho = np.linalg.norm(r)       # Lunghezza vettore r
    v2 = np.dot(dotr, dotr)       # Quadrato della velocità

    # Energia 
    en = 0.5 * v2 - mu / rho
    a = -mu / (2 * en)            # Semiasse maggiore

    # Vettore eccentricità
    e_vec = np.cross(dotr, h) / mu - r / rho
    ecc = np.linalg.norm(e_vec)
    
    # Controllo orbita circolare
    tol = 1e-12
    circular = ecc < tol

    if circular:
        # In Python usiamo array 1D standard [x, y, z]
        e_vec = np.array([1.0, 0.0, 0.0]) 
        ecc = 0.0
    else:
        e_vec = e_vec / ecc # normalizzo il vettore di Lenz
    
    # Controllo orbita chiusa
    if ecc > 1 + tol:
        raise ValueError('orbita iperbolica')

    if abs(ecc - 1) < tol:
        raise ValueError('orbita parabolica')

    # Inclinazione (Attenzione: indice 2 in Python corrisponde all'indice 3 di MATLAB)
    cosI = h[2] / h_norm
    cosI = min(1.0, max(-1.0, cosI))         # limito cos(i) "-1<cos(i)<1"
    I = np.arccos(cosI)

    # Linea dei nodi
    if (I < tol) or (abs(I - np.pi) < tol):  # controllo orbita equatoriale
        Omega_hat = 0.0                      # indefinito
        lin_n = np.array([1.0, 0.0, 0.0])    # sfera assialsimmetrica
    else:
        k_hat = np.array([0.0, 0.0, 1.0])    # versore asse z
        lin_n_v = np.cross(k_hat, h)
        lin_n = lin_n_v / np.linalg.norm(lin_n_v)
        # atan2 restituisce l'angolo in [-pi, pi]. Indici: y=1, x=0
        Omega_hat = np.arctan2(lin_n_v[1], lin_n_v[0]) 

    # Argomento del pericentro (omega -> w)
    if circular:
        w = 0.0
    else:
        # Calcolo l'angolo tramite arcocoseno
        cosomega = np.dot(e_vec, lin_n)
        cosomega = min(1.0, max(-1.0, cosomega)) # Protezione numerica
        w = np.arccos(cosomega)
        
        # Controllo del quadrante:
        # Indice 2 per la componente Z (in Matlab era 3)
        if e_vec[2] < 0:
            w = 2 * np.pi - w

    # Moto medio
    n = np.sqrt(mu / a**3)

    # Anomalia eccentrica iniziale
    if circular:
        E_0 = 0.0
        l = 0.0
    else:
        cosE_0 = (1 - rho / a) / ecc
        cosE_0 = min(1.0, max(-1.0, cosE_0))
        
        sinE_0 = np.dot(r, dotr) / (ecc * a**2 * n)
        E_0 = np.arctan2(sinE_0, cosE_0)
        l = E_0 - ecc * sinE_0

    # Tempo del pericentro
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
       r_out  vettore posizione [3x1] (array NumPy)
       v_out  vettore velocità [3x1] (array NumPy)
    """

    # 1. Risoluzione dell'Equazione di Keplero per trovare l'Anomalia Eccentrica (E)
    #    Metodo di Newton
    
    tol = 1e-12       # Tolleranza
    max_iter = 50     # Iterazioni massime
    
    # Controllo/inizializzazione anomalia eccentrica
    if ecc < 0.9:
        E = l
    else:
        E = np.pi
    
    # Ciclo di Newton
    for k in range(max_iter):
        f_val = E - ecc * np.sin(E) - l
        f_der = 1 - ecc * np.cos(E)
        
        step = f_val / f_der
        E = E - step
        
        if abs(step) < tol:
            break
            
    # 2. Coordinate nel piano orbitale 
    
    # Posizione
    x_orb = a * (np.cos(E) - ecc)
    y_orb = a * np.sqrt(1 - ecc**2) * np.sin(E)
    
    # Velocità
    # Calcoliamo prima il moto medio n
    n = np.sqrt(mu / a**3)
    # Evitiamo divisione per zero se l'orbita è parabolica (ecc ~ 1), ma qui assumiamo ellittica
    factor = (n * a) / (1 - ecc * np.cos(E))
    
    vx_orb = -factor * np.sin(E)
    vy_orb = factor * np.sqrt(1 - ecc**2) * np.cos(E)
    
    # Creiamo i vettori 3D nel piano orbitale
    pos_orb = np.array([x_orb, y_orb, 0.0])
    vel_orb = np.array([vx_orb, vy_orb, 0.0])
    
    # 3. Matrice di Rotazione (dal piano orbitale allo spazio inerziale)
    # Rotazione: R = R_z(-Omega) * R_x(i) * R_z(omega)
    # Nota: In Python usiamo np.array per le matrici
    
    # R_omega (rotazione su Z di w)
    R_w = np.array([
        [np.cos(w), -np.sin(w), 0.0],
        [np.sin(w),  np.cos(w), 0.0],
        [0.0,        0.0,       1.0]
    ])
    
    # R_i (rotazione su X di i)
    R_i = np.array([
        [1.0, 0.0,        0.0],
        [0.0, np.cos(i), -np.sin(i)],
        [0.0, np.sin(i),  np.cos(i)]
    ])
    
    # R_Omega (rotazione su Z di Omega)
    R_Om = np.array([
        [np.cos(Omega), -np.sin(Omega), 0.0],
        [np.sin(Omega),  np.cos(Omega), 0.0],
        [0.0,            0.0,           1.0]
    ])
    
    # Matrice di rotazione totale
    # In Python l'operatore '@' esegue la moltiplicazione matriciale riga-colonna
    Rot_Matrix = R_Om @ R_i @ R_w
    
    # 4. Applicazione rotazione
    # Moltiplichiamo la matrice 3x3 per il vettore 3x1
    r_out = Rot_Matrix @ pos_orb
    v_out = Rot_Matrix @ vel_orb
    
    return r_out, v_out
    

    """
    Calcola la funzione di eccentricità G_lpq(e)
    basato sull'equazione (9-31) del Vallado.
    
    IMPORTANTE:
    L'equazione (9-31) è definita per il caso specifico in cui q = 2p - l.
    E' dunque incompleta poiché nella tabella 9-1 del Vallado ci sono
    valori di G(e) anche per indici dove non vale q=2p-l.
    
    Input:
       l, p, q : indici interi 
       e       : eccentricità (float)
       
    Output:
       G_val   : valore della funzione
    """
    
    # 1. Determina p' (p_prime) come da testo sotto l'equazione
    if p <= l / 2:
        p_prime = p
    else:
        p_prime = l - p
    
    # Inizializza la somma
    # NOTA: In Python 'sum' è una parola riservata, usiamo 'sum_val'
    sum_val = 0.0
    
    # 2. Inizio sommatoria (su d)
    # Limite superiore: p' - 1
    max_d = p_prime - 1
    
    # range in Python esclude l'estremo superiore, quindi usiamo max_d + 1
    # Se max_d è -1 (cioè p_prime=0), il range(0, 0) è vuoto e il ciclo non parte (corretto)
    for d in range(max_d + 1):
        
        # Argomenti del primo coefficiente binomiale: (l-1) su (2d + l - 2p')
        n1 = l - 1
        k1 = 2 * d + l - 2 * p_prime
        
        # Argomenti del secondo coefficiente binomiale: (2d + l - 2p') su (d)
        n2 = k1 
        k2 = d
        
        # controllo dei binomiali (k deve essere tra 0 e n)
        check1 = (k1 >= 0) and (k1 <= n1)
        check2 = (k2 >= 0) and (k2 <= n2)
        
        if check1 and check2:
            bin1 = math.comb(n1, k1)
            bin2 = math.comb(n2, k2)
            
            # Termine di potenza: (e/2)^(2d + l - 2p')
            # Attenzione: ** per la potenza
            term_exp = (e / 2.0)**(2 * d + l - 2 * p_prime)
            
            # Aggiungi alla somma parziale
            sum_val += bin1 * bin2 * term_exp

    # 3. Termine iniziale: 1 / (1 - e^2)^(l - 0.5)
    #    Elevamento a potenza negativa
    term = (1 - e**2)**-(l - 0.5)
    
    # 4. Risultato finale
    G_val = term * sum_val
    
    return G_val

def osculating_orbit(kepElements, mu):
    # 1. Estraiamo i parametri dalla tupla restituita da car2kep
    # Ricorda: a, ecc, I, w, Omega_hat, l, n, tperi = orb_el
    a_0     = kepElements[0]
    ecc_0   = kepElements[1]
    i_0     = kepElements[2]
    w_0     = kepElements[3]
    Omega_0 = kepElements[4]
    l_0     = kepElements[5]
    n       = kepElements[6] # Moto medio (radianti / secondo)

    # 2. Definiamo il tempo di simulazione
    # Calcoliamo il periodo orbitale (T = 2 * pi / n)
    T_periodo = 2 * math.pi / n
    print(f"Periodo dell'orbita: {T_periodo:.2f} secondi (circa {T_periodo/3600:.2f} ore)")

    # Creiamo un array di tempi: da 0 fino a un periodo orbitale, diviso in 500 passi
    # (Usiamo np.linspace per generare facilmente i tempi)
    
    tempi = np.linspace(0, T_periodo, num=500) # 500 punti equispaziati da 0 a delta_t periodi

    # Creiamo liste vuote per salvare le posizioni X, Y, Z nel tempo
    x_hist = []
    y_hist = []
    z_hist = []

    # Creiamo liste vuote per salvare le posizioni X, Y, Z nel tempo
    vx_hist = []
    vy_hist = []
    vz_hist = []

    # Lista per salvare gli elementi orbitali in ogni istante (opzionale)   
    orbEl_hist = [] 

    # 3. Ciclo di propagazione
    for t in tempi:
        # Aggiorniamo solo l'anomalia media nel tempo
        l_t = l_0 + n * t
        
        # Ricalcoliamo posizione e velocità in questo istante t
        # Nota: passiamo l_t invece di l_0, gli altri parametri restano uguali
        r_t, v_t = kep2car(a_0, ecc_0, i_0, w_0, Omega_0, l_t, mu)
        
        # Salviamo le coordinate per il grafico
        x_hist.append(r_t[0])
        y_hist.append(r_t[1])
        z_hist.append(r_t[2])

        vx_hist.append(v_t[0])
        vy_hist.append(v_t[1])
        vz_hist.append(v_t[2])
        orbEl_hist.append(car2kep(r_t, v_t, mu)) # Tutti gli elementi orbitali

    return x_hist, y_hist, z_hist, vx_hist, vy_hist, vz_hist, orbEl_hist   

def propagate_perturbed_orbit(kepElements,mu, num_steps=50000, num_orbits=100):
    """
    Propaga un'orbita perturbata risolvendo numericamente le Equazioni di Lagrange.
    
    INPUT:
        kepElements : tupla (a, ecc, I, w, Omega, M, n, tperi) iniziale
        mu          : float, parametro gravitazionale
        J2, Req     : parametri per la perturbazione
        num_steps   : int, numero di punti da salvare per l'orbita
        num_orbits  : int, quante orbite propagare
    """
    # 1. Estrazione degli elementi iniziali (i primi 6)
    stato_iniziale = list(kepElements[0:6])
    
    # 2. Impostazione dei tempi
    a_0 = kepElements[0]
    T_periodo = 2 * math.pi / np.sqrt(mu / a_0**3)
    t_finale = T_periodo * num_orbits
    
    # Definizione dell'intervallo e dei punti in cui vogliamo i risultati
    t_span = (0, t_finale)
    tempi_eval = np.linspace(0, t_finale, num_steps)
    
    print(f"Propagazione perturbata per {num_orbits} orbita/e (Tempo tot: {t_finale:.2f} s)...")
    
    # 3. Integrazione Numerica con Runge-Kutta
    sol = solve_ivp(
        fun=lpe_derivatives_j2,     # La funzione con le derivate che abbiamo appena creato
        t_span=t_span,              # Da dove a dove integrare
        y0=stato_iniziale,          # Stato iniziale [a, e, i, w, Omega, M]
        t_eval=tempi_eval,          # In quali istanti voglio salvare i dati
        #args=(mu, J2, Req),          # Parametri extra da passare alla funzione
        method='RK45',              # Metodo di integrazione (Runge-Kutta 4(5))
        rtol=1e-9, atol=1e-9        # Tolleranze molto piccole per non perdere precisione!
    )
    
    # Controlliamo se l'integrazione è andata a buon fine
    if not sol.success:
        raise RuntimeError("Integrazione fallita: " + sol.message)
        
    # 4. Estraiamo la "storia" degli elementi orbitali dai risultati
    # sol.y è una matrice dove ogni riga è un elemento orbitale (6 righe in totale)
    # e ogni colonna è un istante di tempo.
    a_hist     = sol.y[0]
    e_hist     = sol.y[1]
    i_hist     = sol.y[2]
    w_hist     = sol.y[3]
    Omega_hist = sol.y[4]
    M_hist     = sol.y[5]
    
    # 5. Riconvertiamo gli elementi orbitali in posizioni X, Y, Z per ogni istante
    x_hist, y_hist, z_hist = [], [], []

    # Creiamo liste vuote per salvare le posizioni X, Y, Z nel tempo
    vx_hist = []
    vy_hist = []
    vz_hist = []
    
    for k in range(len(tempi_eval)):
        # kep2car(a, ecc, i, w, Omega, l, mu)
        r_t, v_t = kep2car(a_hist[k], e_hist[k], i_hist[k], w_hist[k], Omega_hist[k], M_hist[k], mu)
        x_hist.append(r_t[0])
        y_hist.append(r_t[1])
        z_hist.append(r_t[2])

        vx_hist.append(v_t[0])
        vy_hist.append(v_t[1])
        vz_hist.append(v_t[2])
        
    return x_hist, y_hist, z_hist, vx_hist, vy_hist, vz_hist, sol.t, sol.y

def precompute_perturbed_orbit(elements, mu, J2, Re, num_orbits=50):
    """
    Pre-calcola l'orbita perturbata ad altissima risoluzione (un punto ogni 60s)
    per garantire un disegno 3D perfettamente fluido e curvo.
    """
    from scipy.integrate import solve_ivp
    import numpy as np
    import math
    
    a_0, e_0, i_0, w_0, Omega_0, l_0, n_0, t_peri = elements
    y0 = [a_0, e_0, i_0, w_0, Omega_0, l_0]
    
    T_periodo = 2 * math.pi / n_0
    t_finale = T_periodo * num_orbits
    
    # FORZATURA: Generiamo sempre un punto ogni 60 secondi per avere la curva "liscia"
    tempi_eval = np.arange(0, t_finale, 60.0)
    
    sol = solve_ivp(
        fun=lpe_derivatives_j2,  
        t_span=(0, t_finale),
        y0=y0,
        t_eval=tempi_eval,
        args=(mu, J2, Re),
        method='LSODA',
        rtol=1e-6, atol=1e-6  # Tolleranza perfetta per questo step
    )
    
    if not sol.success:
        raise RuntimeError(f"Integrazione fallita: {sol.message}")
        
    return sol

# --- FILE potenziali perturbati --- #

# Heart
def lpe_derivatives_j2(t, state, mu=398600.4418, J2=1.08263e-3, R_eq=6371.0):
    """
    Calcola le derivate degli elementi orbitali (Equazioni Planetarie di Lagrange)
    considerando solo la perturbazione zonale J2.
    """
    a, e, i, w, Omega, M = state

    # --- PROTEZIONE SINGOLARITÀ MATEMATICHE ---
    # Le equazioni di Lagrange "esplodono" se l'orbita è perfettamente 
    # circolare (e=0) o perfettamente equatoriale (i=0).
    e = max(e, 1e-12)
    if abs(i) < 1e-12: i = 1e-12
    elif abs(i - np.pi) < 1e-12: i = np.pi - 1e-12

    n = np.sqrt(mu / a**3)
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
    
    dM_dt = n + J2 * mu * t4 * t8 * 1.0 / (-t9)**(3.0 / 2.0) * t13 * t14 * (-3.0 / 2.0)

    return [da_dt, de_dt, di_dt, dw_dt, dOmega_dt, dM_dt]

# Moon - L=10 - Q=2
def perturbed_moon(t, state, theta_GMST, mu=4902.8):
    """
    Derivate LPE per potenziale lunare generato con L_MAX=10, Q_MAX=2.
    """
    a, e, I_inc, w, Omega, M = state

    # Protezione singolarità matematiche
    e = max(e, 1e-12)
    if abs(I_inc) < 1e-12: I_inc = 1e-12
    elif abs(I_inc - np.pi) < 1e-12: I_inc = np.pi - 1e-12
    I = I_inc

    # --- Variabili temporanee ottimizzate ---
    x0 = np.sin(I)
    x1 = x0**8
    x2 = x0**10
    x3 = x0**2
    x4 = x0**6
    x5 = x0**4
    x6 = 3.3837890625*x3 + 82.4798583984375*x4 - 29.326171875*x5
    x7 = (765765/8192)*x1 - 4849845/131072*x2 - x6
    x8 = 2*w
    x9 = np.sin(x8)
    x10 = e**2
    x11 = e**4
    x12 = e**6
    x13 = e**8
    x14 = 9*x10 + (63/2)*x11 + (315/16)*x12 + (63/32)*x13
    x15 = a**(-11.0)
    x16 = 1 - x10
    x17 = x16**(-19/2)
    x18 = x15*x17
    x19 = x14*x18
    x20 = 1.2110073631743e+31*x19*x9
    x21 = 93.4771728515625*x1 - 37.0013809204102*x2 - x6
    x22 = x0**5
    x23 = x0**7
    x24 = x0**3
    x25 = 1.09375*x0
    x26 = -x25
    x27 = (15015/2048)*x23 + 7.3828125*x24 + x26
    x28 = -13.53515625*x22 + x27
    x29 = 5.07390783087972e+21*x28
    x30 = np.cos(w)
    x31 = e**3
    x32 = (15/2)*x31
    x33 = e**5
    x34 = 3*e + x32 + (15/8)*x33
    x35 = a**(-8.0)
    x36 = x16**(-13/2)
    x37 = x35*x36
    x38 = x34*x37
    x39 = x30*x38
    x40 = -3/4*x0
    x41 = (15/16)*x24
    x42 = -x40 - x41
    x43 = a**(-4.0)
    x44 = x16**(-5/2)
    x45 = x43*x44
    x46 = 219418370.14608*x45
    x47 = e*x30
    x48 = 3*Omega
    x49 = 3*theta_GMST
    x50 = -x48 + x49
    x51 = x50 + x8
    x52 = np.sin(x51)
    x53 = np.cos(x51)
    x54 = a**(-9.0)
    x55 = 4.08171193584022e+29*x54
    x56 = np.cos(I)
    x57 = x56**2
    x58 = 6*x57
    x59 = -x58
    x60 = x56**3
    x61 = 4*x60
    x62 = x61 + 2.0
    x63 = x59 + x62
    x64 = -x63
    x65 = (10395/128)*x0
    x66 = 15*x57
    x67 = -x66
    x68 = x67 + 3.0
    x69 = 3*x56
    x70 = 15*x60
    x71 = -x69 + x70
    x72 = x68 + x71
    x73 = (45045/256)*x24
    x74 = 42*x57
    x75 = -x74
    x76 = x75 + 6
    x77 = 12*x56
    x78 = -x77
    x79 = 56*x60
    x80 = x78 + x79
    x81 = -x76 - x80
    x82 = (135135/2048)*x22
    x83 = x64*x65 + x72*x73 + x81*x82
    x84 = x16**(-15/2)
    x85 = (21/4)*x10 + (35/4)*x11 + (105/64)*x12
    x86 = x84*x85
    x87 = x83*x86
    x88 = x55*x87
    x89 = x48 - x49
    x90 = x8 + x89
    x91 = np.sin(x90)
    x92 = np.cos(x90)
    x93 = 6.0*x57
    x94 = x93 - 2.0
    x95 = 4.0*x60 + x94
    x96 = -x95
    x97 = 3.0*x56
    x98 = -x97
    x99 = x70 + x98
    x100 = 15.0*x57
    x101 = x100 - 3.0
    x102 = x101 + x99
    x103 = x74 - 6.0
    x104 = 12.0*x56
    x105 = -x104
    x106 = x105 + x79
    x107 = -x103 - x106
    x108 = x102*x73 + x107*x82 + x65*x96
    x109 = x55*x86
    x110 = x108*x109
    x111 = 4*Omega
    x112 = 4*theta_GMST
    x113 = -x111 + x112
    x114 = x113 + x8
    x115 = np.sin(x114)
    x116 = np.cos(x114)
    x117 = 649.6875*x56
    x118 = x56**4
    x119 = -x79
    x120 = 24*x56
    x121 = x119 + x120
    x122 = 24*x57
    x123 = -x122
    x124 = 56*x118
    x125 = x123 + x124
    x126 = -x121 - x125
    x127 = (675675/2048)*x5
    x128 = 20*x60
    x129 = x105 + x128
    x130 = 1.0 - 15*x118
    x131 = x130 + x58
    x132 = -x129 - x131
    x133 = (135135/256)*x3
    x134 = -x117 - 10395/32*x118 + x126*x127 + x132*x133 + (10395/16)*x60 + 324.84375
    x135 = x109*x134
    x136 = Omega - theta_GMST
    x137 = x136 + x8
    x138 = np.sin(x137)
    x139 = np.cos(x137)
    x140 = 1.0*x56
    x141 = x140 + 1.0
    x142 = -x141
    x143 = (315/64)*x0
    x144 = 15.0*x56
    x145 = x144 + 5.0
    x146 = -x145
    x147 = (9009/1024)*x22
    x148 = 4.0*x56
    x149 = x148 + 2.0
    x150 = (3465/256)*x24
    x151 = 56*x56
    x152 = x151 + 14.0
    x153 = (6435/4096)*x23
    x154 = x142*x143 + x146*x147 + x149*x150 + x152*x153
    x155 = x109*x154
    x156 = 5*Omega
    x157 = 5*theta_GMST
    x158 = -x156 + x157
    x159 = x158 + x8
    x160 = np.sin(x159)
    x161 = np.cos(x159)
    x162 = x56**5
    x163 = 56*x162
    x164 = 40*x60
    x165 = 60*x57
    x166 = 70*x118
    x167 = x166 + 6.0
    x168 = -x165 + x167
    x169 = -x163 + x164 + x168
    x170 = -x169
    x171 = (675675/512)*x24
    x172 = 10*x60
    x173 = 30*x57
    x174 = -x173
    x175 = 15*x162
    x176 = -x175
    x177 = x172 + x174 + x176
    x178 = 5.0*x56
    x179 = 25*x118
    x180 = x179 + 5.0
    x181 = x178 + x180
    x182 = x177 + x181
    x183 = (135135/128)*x0
    x184 = x170*x171 + x182*x183
    x185 = x109*x184
    x186 = 6*Omega
    x187 = 6*theta_GMST
    x188 = -x186 + x187
    x189 = x188 + x8
    x190 = np.sin(x189)
    x191 = np.cos(x189)
    x192 = (2027025/32)*x60
    x193 = (2027025/64)*x162
    x194 = 31672.265625*x56
    x195 = 84*x162
    x196 = -x195
    x197 = 120*x60
    x198 = 36.0*x56
    x199 = x56**6
    x200 = 56*x199 + 4.0
    x201 = -60*x118 + x200
    x202 = x196 + x197 - x198 + x201
    x203 = (2027025/512)*x3
    x204 = -2027025/128*x199 + 15836.1328125*x57 - 15836.1328125
    x205 = (2027025/128)*x118 - x192 + x193 + x194 + x202*x203 + x204
    x206 = x109*x205
    x207 = x156 - x157
    x208 = x207 + x8
    x209 = np.sin(x208)
    x210 = np.cos(x208)
    x211 = -x164
    x212 = 60.0*x57
    x213 = x167 - x212
    x214 = x163 + x213
    x215 = x211 + x214
    x216 = 30.0*x57
    x217 = -x216
    x218 = 10.0*x60
    x219 = x175 - x178 + x217 - x218
    x220 = -x180 - x219
    x221 = x171*x215 + x183*x220
    x222 = x109*x221
    x223 = 8*Omega
    x224 = 8*theta_GMST
    x225 = x223 - x224
    x226 = x225 + x8
    x227 = np.sin(x226)
    x228 = np.cos(x226)
    x229 = (42567525/16)*x162
    x230 = (14189175/16)*x56
    x231 = x56**7
    x232 = (14189175/16)*x231
    x233 = (42567525/16)*x60
    x234 = x56**8
    x235 = -14189175/16*x199 + (14189175/32)*x234 + (14189175/16)*x57 - 14189175/32
    x236 = x229 + x230 - x232 - x233 - x235
    x237 = x109*x236
    x238 = -Omega + theta_GMST
    x239 = x238 + x8
    x240 = np.sin(x239)
    x241 = np.cos(x239)
    x242 = x140 - 1.0
    x243 = -x242
    x244 = 4*x56
    x245 = x244 - 2.0
    x246 = 15*x56
    x247 = x246 - 5
    x248 = -x247
    x249 = x151 - 14
    x250 = x143*x243 + x147*x248 + x150*x245 + x153*x249
    x251 = x109*x250
    x252 = x186 - x187
    x253 = x252 + x8
    x254 = np.sin(x253)
    x255 = np.cos(x253)
    x256 = 63344.53125*x60
    x257 = x195 + x198
    x258 = x257 - 120.0*x60
    x259 = x201 + x258
    x260 = 15836.1328125*x118 - x193 - x194 + x203*x259 + x204 + x256
    x261 = x109*x260
    x262 = -x223 + x224
    x263 = x262 + x8
    x264 = np.sin(x263)
    x265 = np.cos(x263)
    x266 = -x229 - x230 + x232 + x233 - x235
    x267 = x109*x266
    x268 = x111 - x112
    x269 = x268 + x8
    x270 = np.sin(x269)
    x271 = np.cos(x269)
    x272 = 24.0*x56
    x273 = -x272
    x274 = x273 + x79
    x275 = 24.0*x57
    x276 = -x275
    x277 = x124 + x276
    x278 = -x274 - x277
    x279 = -20.0*x60
    x280 = x130 + x93
    x281 = x104 + x280
    x282 = -x279 - x281
    x283 = x117 - 324.84375*x118 + x127*x278 + x133*x282 - 649.6875*x60 + 324.84375
    x284 = x109*x283
    x285 = 2*Omega
    x286 = 2*theta_GMST
    x287 = -x285 + x286
    x288 = x287 + x8
    x289 = np.sin(x288)
    x290 = np.cos(x288)
    x291 = 4.94610544005789e-7*x289 + 3.1750217626037e-7*x290
    x292 = 9.84375*x56
    x293 = -x148
    x294 = 4*x57
    x295 = x293 + x294
    x296 = (10395/256)*x3
    x297 = 10*x56
    x298 = x297 + x67
    x299 = x298 + 1.0
    x300 = (45045/1024)*x5
    x301 = 56*x57
    x302 = 28*x56
    x303 = -x301 + x302
    x304 = -x303 - 4
    x305 = (45045/4096)*x4
    x306 = -4.921875*x57 - 4.921875
    x307 = x292 + x295*x296 + x299*x300 + x304*x305 + x306
    x308 = x109*x307
    x309 = 7*Omega
    x310 = 7*theta_GMST
    x311 = -x309 + x310
    x312 = w + x311
    x313 = np.sin(x312)
    x314 = np.cos(x312)
    x315 = (14189175/128)*x57
    x316 = (4729725/128)*x199
    x317 = -14189175/128*x162 + (4729725/128)*x231 - 4729725/128*x56 + (14189175/128)*x60
    x318 = (14189175/128)*x118 - x315 - x316 + x317 + 4729725/128
    x319 = 2.34851089519e+26*x38
    x320 = x318*x319
    x321 = w + x287
    x322 = np.sin(x321)
    x323 = np.cos(x321)
    x324 = 2.0*x56
    x325 = 3*x57
    x326 = -x325
    x327 = x326 + 1.0
    x328 = -x324 - x327
    x329 = (945/64)*x0
    x330 = 10*x57
    x331 = -x330
    x332 = x244 + 2.0
    x333 = x331 + x332
    x334 = (3465/128)*x24
    x335 = 35*x57
    x336 = x297 - x335
    x337 = -x336 - 5
    x338 = x147*x337 + x328*x329 + x333*x334
    x339 = x319*x338
    x340 = x285 - x286
    x341 = w + x340
    x342 = np.sin(x341)
    x343 = np.cos(x341)
    x344 = 3.0*x57
    x345 = x324 - 1.0
    x346 = x344 + x345
    x347 = -x346
    x348 = x148 - 2.0
    x349 = x330 + x348
    x350 = x335 - 5.0
    x351 = -x297 - x350
    x352 = x147*x351 + x329*x347 + x334*x349
    x353 = x319*x352
    x354 = w + x158
    x355 = np.sin(x354)
    x356 = np.cos(x354)
    x357 = (51975/32)*x118
    x358 = 35*x162
    x359 = x358 - 50*x60
    x360 = x144 - x179 + x359 - 5.0
    x361 = -x173 - x360
    x362 = (51975/32)*x162 + 1624.21875*x56
    x363 = x133*x361 - x357 + x362 + 3248.4375*x57 - 51975/16*x60 - 1624.21875
    x364 = x319*x363
    x365 = w + x207
    x366 = np.sin(x365)
    x367 = np.cos(x365)
    x368 = x145 + x179 + x217
    x369 = x359 + x368
    x370 = (135135/256)*x3*x369 - x357 - x362 + 3248.4375*x57 + 3248.4375*x60 - 1624.21875
    x371 = x319*x370
    x372 = w + x113
    x373 = np.sin(x372)
    x374 = np.cos(x372)
    x375 = 12*x57
    x376 = -x375
    x377 = 8*x60
    x378 = -x377
    x379 = x376 + x378
    x380 = 8.0*x56
    x381 = 10*x118
    x382 = x381 + 2.0
    x383 = x380 + x382
    x384 = x379 + x383
    x385 = (10395/64)*x0
    x386 = -x128
    x387 = 35*x118 + 3.0
    x388 = x174 + x387
    x389 = x386 + x388
    x390 = -x389 - x77
    x391 = x384*x385 + x390*x73
    x392 = x319*x391
    x393 = w + x89
    x394 = np.sin(x393)
    x395 = np.cos(x393)
    x396 = 44.296875*x56
    x397 = 6.0*x56
    x398 = -x397
    x399 = x172 + x398
    x400 = x399 + x94
    x401 = (10395/128)*x3
    x402 = x66 - 3.0
    x403 = 35*x60
    x404 = -x144 + x403
    x405 = -x402 - x404
    x406 = 44.296875 - 44.296875*x57
    x407 = x300*x405 + x396 + x400*x401 + x406 - 44.296875*x60
    x408 = x319*x407
    x409 = x309 - x310
    x410 = w + x409
    x411 = np.sin(x410)
    x412 = np.cos(x410)
    x413 = (14189175/128)*x118 - x315 - x316 - x317 + 4729725/128
    x414 = x319*x413
    x415 = w + x238
    x416 = np.sin(x415)
    x417 = np.cos(x415)
    x418 = 5.47764409510595e-6*x416 - 8.83386032513283e-8*x417
    x419 = 1.09375*x56
    x420 = x97 - 1.0
    x421 = -x420
    x422 = (945/128)*x3
    x423 = x297 - 2.0
    x424 = (3465/512)*x5
    x425 = 35*x56
    x426 = 5 - x425
    x427 = (3003/2048)*x4
    x428 = x419 + x421*x422 + x423*x424 + x426*x427 - 1.09375
    x429 = x319*x428
    x430 = w + x268
    x431 = np.sin(x430)
    x432 = np.cos(x430)
    x433 = 12.0*x57
    x434 = -x433
    x435 = -x380 + x382 + x434
    x436 = -x377 - x435
    x437 = x129 + x388
    x438 = x385*x436 + x437*x73
    x439 = x319*x438
    x440 = w + x50
    x441 = np.sin(x440)
    x442 = np.cos(x440)
    x443 = x59 + 2.0
    x444 = -x399 - x443
    x445 = -x246 + x403 + x68
    x446 = x300*x445 - x396 + x401*x444 + x406 + (2835/64)*x60
    x447 = x319*x446
    x448 = 0.9375*x0
    x449 = (315/128)*x22 + x448
    x450 = (105/32)*x24 - x449
    x451 = 57326844710022.1*x450
    x452 = 2*e + (3/2)*x31
    x453 = a**(-6.0)
    x454 = x16**(-9/2)
    x455 = x453*x454
    x456 = x452*x455
    x457 = x30*x456
    x458 = a**(-7.0)
    x459 = 1.35127209159379e+23*x458
    x460 = 236.25*x56
    x461 = x128 + x78
    x462 = x131 + x461
    x463 = (945/8)*x118 - 118.125
    x464 = x401*x462 + x460 + x463 - 945/4*x60
    x465 = x16**(-11/2)
    x466 = (5/2)*x10 + (5/4)*x11
    x467 = x465*x466
    x468 = x464*x467
    x469 = x459*x468
    x470 = (155925/64)*x57
    x471 = (155925/64)*x118
    x472 = (155925/16)*x60
    x473 = (155925/32)*x56
    x474 = (155925/32)*x162 - x472 + x473
    x475 = (155925/64)*x199 - x470 - x471 - x474 + 155925/64
    x476 = x459*x467
    x477 = x475*x476
    x478 = (155925/64)*x199 - x470 - x471 + x474 + 155925/64
    x479 = x476*x478
    x480 = (105/32)*x0
    x481 = -x149
    x482 = (315/64)*x24
    x483 = x246 + 5.0
    x484 = (693/512)*x22
    x485 = x141*x480 + x481*x482 + x483*x484
    x486 = x476*x485
    x487 = 1.09232625193902e-6*x289 + 5.39857315699859e-7*x290
    x488 = 6.5625*x56
    x489 = -x244 + x294
    x490 = -x489
    x491 = (945/64)*x3
    x492 = -x298 - 1
    x493 = x424*x492 - x488 + x490*x491 + (105/32)*x57 + 3.28125
    x494 = x476*x493
    x495 = (945/32)*x0
    x496 = -x67 - x71 - 3
    x497 = x334*x496 + x495*x63
    x498 = x476*x497
    x499 = x281 + x386
    x500 = x401*x499 - x460 + x463 + 236.25*x60
    x501 = x476*x500
    x502 = x56 - 1.0
    x503 = x244 - 2
    x504 = -x503
    x505 = x247*x484 + x480*x502 + x482*x504
    x506 = x476*x505
    x507 = x61 + x94
    x508 = -x402 - x99
    x509 = x334*x508 + x495*x507
    x510 = x476*x509
    x511 = (45045/256)*x0
    x512 = -x101 - 15.0*x60 - x98
    x513 = (675675/1024)*x24
    x514 = x106 + 42.0*x57 - 6.0
    x515 = (2297295/4096)*x22
    x516 = 126*x57
    x517 = x516 - 14.0
    x518 = 210*x60
    x519 = -42.0*x56
    x520 = x518 + x519
    x521 = -x517 - x520
    x522 = (2078505/16384)*x23
    x523 = x511*x95 + x512*x513 + x514*x515 + x521*x522
    x524 = 1.23293986687242e+36*x19
    x525 = x523*x524
    x526 = (3465/512)*x0
    x527 = 56.0*x56
    x528 = -x527 - 14.0
    x529 = (109395/8192)*x23
    x530 = (15015/512)*x24
    x531 = (135135/4096)*x22
    x532 = 210*x56
    x533 = x532 + 42.0
    x534 = x0**9
    x535 = (230945/131072)*x534
    x536 = x141*x526 + x145*x531 + x481*x530 + x528*x529 + x533*x535
    x537 = x524*x536
    x538 = (241215975/32)*x231
    x539 = 22613997.65625*x60
    x540 = 7537999.21875*x56
    x541 = 784*x162
    x542 = 336*x231
    x543 = 560.0*x60
    x544 = 112.0*x56
    x545 = 56.0*x57
    x546 = 392*x199
    x547 = -x546
    x548 = 210*x234 + x545 + x547 - 14.0
    x549 = -140.0*x118 + x541 - x542 - x543 + x544 - x548
    x550 = (654729075/2048)*x3
    x551 = -241215975/32*x199 + (241215975/64)*x234 + 7537999.21875*x57 - 3768999.609375
    x552 = -22613997.65625*x162 + x538 + x539 - x540 + x549*x550 + x551
    x553 = x524*x552
    x554 = 4.25884464445994e-8*x289 - 3.18787653039088e-8*x290
    x555 = 13.53515625*x56
    x556 = 4.0*x57
    x557 = -x293 - x556
    x558 = (45045/512)*x3
    x559 = 10.0*x56
    x560 = -x559 - x67 - 1.0
    x561 = (675675/4096)*x5
    x562 = x303 + 4.0
    x563 = (765765/8192)*x4
    x564 = 84*x56
    x565 = -210*x57
    x566 = -x564 - x565 - 14
    x567 = (2078505/131072)*x1
    x568 = 6.767578125*x57 + 6.767578125
    x569 = -x555 + x557*x558 + x560*x561 + x562*x563 + x566*x567 + x568
    x570 = x524*x569
    x571 = -140*x118 - x541 + x542 - x544 - x548 + 560*x60
    x572 = (723647925/32)*x162 - x538 - x539 + x540 + x550*x571 + x551
    x573 = x524*x572
    x574 = 1407.65625*x56
    x575 = 1407.65625*x60
    x576 = x273 + x277 + 56.0*x60
    x577 = (11486475/4096)*x5
    x578 = x104 - 15.0*x118 + x279 + x93 + 1.0
    x579 = (2027025/1024)*x3
    x580 = 168*x60
    x581 = 210*x118
    x582 = x581 + 2.0
    x583 = x527 + 84.0*x57 - x580 - x582
    x584 = (14549535/16384)*x4
    x585 = 703.828125*x118 - x574 + x575 + x576*x577 + x578*x579 + x583*x584 - 703.828125
    x586 = x524*x585
    x587 = 10*Omega
    x588 = 10*theta_GMST
    x589 = -x587 + x588 + x8
    x590 = np.sin(x589)
    x591 = np.cos(x589)
    x592 = (206239658625/128)*x162
    x593 = x56**9
    x594 = (68746552875/256)*x593
    x595 = (68746552875/256)*x56
    x596 = (68746552875/64)*x60
    x597 = (68746552875/64)*x231
    x598 = x56**10
    x599 = (68746552875/256)*x118 + (68746552875/256)*x199 - 206239658625/512*x234 - 206239658625/512*x57 + (68746552875/512)*x598 + 68746552875/512
    x600 = -x592 - x594 - x595 + x596 + x597 + x599
    x601 = x524*x600
    x602 = -x93
    x603 = x602 + x62
    x604 = -x68 - x99
    x605 = x75 + x80 + 6.0
    x606 = -42*x56
    x607 = 14 - x516
    x608 = -x518 - x606 - x607
    x609 = x511*x603 + x513*x604 + x515*x605 + x522*x608
    x610 = x524*x609
    x611 = x119 + x125 + x272
    x612 = x129 + x280
    x613 = -84*x57
    x614 = -x151 + x580 - x582 - x613
    x615 = (45045/64)*x118 + x574 - x575 + x577*x611 + x579*x612 + x584*x614 - 703.828125
    x616 = x524*x615
    x617 = -40.0*x60
    x618 = -x214 - x617
    x619 = (11486475/1024)*x24
    x620 = 25.0*x118 + x219 + 5.0
    x621 = (2027025/512)*x0
    x622 = 140*x60
    x623 = -x622
    x624 = 210*x162
    x625 = x623 + x624
    x626 = x559 - 140.0*x57 + x581 + x625 + 10.0
    x627 = (43648605/8192)*x22
    x628 = x618*x619 + x620*x621 + x626*x627
    x629 = x524*x628
    x630 = 237541.9921875*x60
    x631 = 118770.99609375*x56
    x632 = -x202
    x633 = (34459425/1024)*x3
    x634 = 252*x162
    x635 = 280*x60
    x636 = 60.0*x56
    x637 = -x581
    x638 = 210*x199 + x637 + 2.0
    x639 = x173 - x634 + x635 - x636 + x638
    x640 = (218243025/8192)*x5
    x641 = (30405375/512)*x199 - 59385.498046875*x57 + 59385.498046875
    x642 = -30405375/512*x118 - 30405375/256*x162 + x630 - x631 + x632*x633 + x639*x640 + x641
    x643 = x524*x642
    x644 = 60.0*x118 - x200 - x258
    x645 = x634 + x636
    x646 = x216 - 280.0*x60 + x638 + x645
    x647 = -59385.498046875*x118 + 118770.99609375*x162 - x630 + x631 + x633*x644 + x640*x646 + x641
    x648 = x524*x647
    x649 = x311 + x8
    x650 = np.sin(x649)
    x651 = np.cos(x649)
    x652 = 28.0*x56
    x653 = x196 + 56*x231
    x654 = x652 + x653
    x655 = 126.0*x57
    x656 = 98*x199
    x657 = x581 - x656
    x658 = -x655 + x657 + 14.0
    x659 = x654 + x658
    x660 = (34459425/512)*x0
    x661 = 70*x60
    x662 = 294*x199
    x663 = 490*x118
    x664 = -294*x162
    x665 = 210*x231 + 14.0*x56 + x664
    x666 = -x565 - x661 + x662 - x663 - x665 - 14.0
    x667 = (218243025/2048)*x24
    x668 = x659*x660 + x666*x667
    x669 = x524*x668
    x670 = -x348
    x671 = x246 - 5.0
    x672 = -x249
    x673 = x532 - 42
    x674 = x242*x526 + x529*x672 + x530*x670 + x531*x671 + x535*x673
    x675 = x524*x674
    x676 = x409 + x8
    x677 = np.sin(x676)
    x678 = np.cos(x676)
    x679 = x655 + x656 - 14.0
    x680 = -210.0*x118 + x654 + x679
    x681 = 210.0*x57
    x682 = -70.0*x60 - x662 + x663 - x665 - x681 + 14.0
    x683 = x660*x680 + x667*x682
    x684 = x524*x683
    x685 = -x172 - x176 - x181 - x217
    x686 = x297 + 140*x57 + x625 + x637 - 10.0
    x687 = x169*x619 + x621*x685 + x627*x686
    x688 = x524*x687
    x689 = x587 - x588
    x690 = x689 + x8
    x691 = np.sin(x690)
    x692 = np.cos(x690)
    x693 = x592 + x594 + x595 - x596 - x597 + x599
    x694 = x524*x693
    x695 = x325 + x345
    x696 = (105/16)*x0
    x697 = -x245 - x330
    x698 = x482*x697 + x695*x696
    x699 = 7.77486819098842e+19*x456
    x700 = x698*x699
    x701 = 19.6875*x56
    x702 = (315/16)*x60
    x703 = x58 - 2.0
    x704 = -x399 - x703
    x705 = x491*x704 + 19.6875*x57 - x701 + x702 - 19.6875
    x706 = x699*x705
    x707 = 2*x56
    x708 = x327 + x707
    x709 = -x244 - x331 - 2
    x710 = x482*x709 + x696*x708
    x711 = x699*x710
    x712 = 8.69203554877188e-7*x416 + 3.5284374029458e-6*x417
    x713 = 0.9375*x56
    x714 = x69 - 1.0
    x715 = (105/32)*x3
    x716 = 2 - x297
    x717 = (315/256)*x5
    x718 = -x713 + x714*x715 + x716*x717 + 0.9375
    x719 = x699*x718
    x720 = (4725/8)*x57
    x721 = (4725/16)*x162 + (4725/16)*x56 - 4725/8*x60
    x722 = (4725/16)*x118 - x720 - x721 + 4725/16
    x723 = x699*x722
    x724 = (315/16)*x57
    x725 = 6*x56
    x726 = -x725
    x727 = x172 + x443 + x726
    x728 = x491*x727 + x701 - x702 + x724 - 19.6875
    x729 = x699*x728
    x730 = (4725/16)*x118 - x720 + x721 + 4725/16
    x731 = x699*x730
    x732 = 81.2109375*x56
    x733 = 81.2109375*x60
    x734 = x399 + x602 + 2.0
    x735 = (135135/512)*x3
    x736 = -x404 - x68
    x737 = 126*x60
    x738 = x606 + x737 + x76
    x739 = 81.2109375*x57 - 81.2109375
    x740 = x127*x736 + x563*x738 + x732 - x733 + x734*x735 + x739
    x741 = e**7
    x742 = 4*e + 21*x31 + (35/2)*x33 + (35/16)*x741
    x743 = a**(-10.0)
    x744 = x16**(-17/2)
    x745 = x743*x744
    x746 = x742*x745
    x747 = 7.09401534449031e+32*x746
    x748 = x740*x747
    x749 = -x218 - x398 - x94
    x750 = x101 + x404
    x751 = -x103 - x519 - x737
    x752 = x127*x750 + x563*x751 - x732 + x733 + x735*x749 + x739
    x753 = x747*x752
    x754 = (3465/128)*x0
    x755 = -x348 - 10.0*x57
    x756 = (45045/512)*x24
    x757 = x350 + x559
    x758 = -x302 - x517
    x759 = x346*x754 + x529*x758 + x755*x756 + x757*x82
    x760 = x747*x759
    x761 = (70945875/256)*x231
    x762 = (212837625/256)*x162
    x763 = 277132.32421875*x56
    x764 = 831396.97265625*x60
    x765 = 126*x231 + x664
    x766 = x520 + x658 + x765
    x767 = (70945875/256)*x199 + 831396.97265625*x57 - 277132.32421875
    x768 = -212837625/256*x118 + x633*x766 - x761 + x762 + x763 - x764 + x767
    x769 = x747*x768
    x770 = 210.0*x60
    x771 = -x606 - x637 - x679 - x765 - x770
    x772 = -831396.97265625*x118 + x633*x771 + x761 - x762 - x763 + x764 + x767
    x773 = x747*x772
    x774 = (675675/128)*x162
    x775 = 10557.421875*x60
    x776 = 5278.7109375*x56
    x777 = x216 + x360
    x778 = 30.0*x56
    x779 = 126*x162 + x623 + x778
    x780 = -x165 + x166 - x779 + 6.0
    x781 = 5278.7109375 - 10557.421875*x57
    x782 = (675675/128)*x118 + x203*x777 + x577*x780 - x774 + x775 - x776 + x781
    x783 = x747*x782
    x784 = -x358 - x368 + 50.0*x60
    x785 = x213 + x779
    x786 = 5278.7109375*x118 + x203*x784 + x577*x785 + x774 - x775 + x776 + x781
    x787 = x747*x786
    x788 = -x378 - x383 - x434
    x789 = (135135/256)*x0
    x790 = x104 + x389
    x791 = 126*x118 + x613 + 6.0
    x792 = -x121 - x791
    x793 = x171*x790 + x515*x792 + x788*x789
    x794 = x747*x793
    x795 = 9*Omega
    x796 = 9*theta_GMST
    x797 = -x795 + x796
    x798 = w + x797
    x799 = np.sin(x798)
    x800 = np.cos(x798)
    x801 = (2170943775/64)*x57
    x802 = (2170943775/64)*x199
    x803 = (6512831325/128)*x162 - 2170943775/64*x231 + (2170943775/256)*x56 + (2170943775/256)*x593 - 2170943775/64*x60
    x804 = (6512831325/128)*x118 + (2170943775/256)*x234 - x801 - x802 - x803 + 2170943775/256
    x805 = x747*x804
    x806 = x795 - x796
    x807 = w + x806
    x808 = np.sin(x807)
    x809 = np.cos(x807)
    x810 = (6512831325/128)*x118 + (2170943775/256)*x234 - x801 - x802 + x803 + 2170943775/256
    x811 = x747*x810
    x812 = w + x188
    x813 = np.sin(x812)
    x814 = np.cos(x812)
    x815 = 45.0*x57
    x816 = 30*x162
    x817 = x778 + x816
    x818 = 60*x60
    x819 = -x818
    x820 = 35*x199
    x821 = 75*x118
    x822 = x819 - x820 + x821
    x823 = -x815 + x817 + x822 + 5.0
    x824 = (2027025/256)*x0
    x825 = 126*x199
    x826 = -x197 + x257
    x827 = 90*x57 - x581 + x825 - x826 - 6.0
    x828 = x619*x827 + x823*x824
    x829 = x747*x828
    x830 = x324 - x344 + 1.0
    x831 = -x149 - x331
    x832 = x336 + 5.0
    x833 = x302 + x607
    x834 = -x833
    x835 = x529*x834 + x754*x830 + x756*x831 + x82*x832
    x836 = x747*x835
    x837 = w + x252
    x838 = np.sin(x837)
    x839 = np.cos(x837)
    x840 = 60.0*x60
    x841 = x820 - x821
    x842 = x815 + x817 - x840 + x841 - 5.0
    x843 = 90.0*x57
    x844 = -x637 - x825 - x826 - x843 + 6.0
    x845 = x619*x844 + x824*x842
    x846 = x747*x845
    x847 = 1.20499967550068e-6*x416 + 5.46825298616797e-8*x417
    x848 = 1.23046875*x56
    x849 = 2.0 - x559
    x850 = (45045/2048)*x5
    x851 = (3465/256)*x3
    x852 = x425 - 5.0
    x853 = 126*x56
    x854 = 14 - x853
    x855 = (109395/65536)*x1
    x856 = x305*x852 + x420*x851 - x848 + x849*x850 + x854*x855 + 1.23046875
    x857 = x747*x856
    x858 = x435 + 8.0*x60
    x859 = -x129 - x217 - x387
    x860 = x274 + x791
    x861 = x171*x859 + x515*x860 + x789*x858
    x862 = x747*x861
    x863 = 1.23046875*x0
    x864 = 43.9892578125*x22 - 13.53515625*x24 + (765765/32768)*x534 + x863
    x865 = (225225/4096)*x23 - x864
    x866 = 1.09330085366536e+28*x865
    x867 = x30*x746
    x868 = (45/8)*x57
    x869 = -45/8*x56 + (45/8)*x60
    x870 = -x868 + x869 + 45/8
    x871 = 25739107921980.9*x45
    x872 = e*x871
    x873 = x870*x872
    x874 = -x868 - x869 + 45/8
    x875 = x871*x874
    x876 = e*x875
    x877 = 2.84663988257272e-5*x416 + 5.89772619674322e-6*x417
    x878 = 1 - x69
    x879 = (15/16)*x3*x878 + (3/4)*x56 - 0.75
    x880 = x871*x879
    x881 = e*x880
    x882 = (105/2)*x60
    x883 = (105/2)*x56
    x884 = (105/4)*x118 - 105/4
    x885 = x882 - x883 - x884
    x886 = a**(-5.0)
    x887 = x16**(-7/2)
    x888 = x886*x887
    x889 = x10*x888
    x890 = 3.35509271763021e+16*x889
    x891 = x885*x890
    x892 = -x882 + x883 - x884
    x893 = x890*x892
    x894 = 3.18221440738855e-6*x289 + 3.02797674808195e-6*x290
    x895 = (15/4)*x56
    x896 = x489*x715 - 15/8*x57 + x895 - 1.875
    x897 = x890*x896
    x898 = (15/8)*x0
    x899 = (35/32)*x24
    x900 = x142*x898 + x332*x899
    x901 = x890*x900
    x902 = 1 - x56
    x903 = x503*x899 + x898*x902
    x904 = x890*x903
    x905 = 1.640625*x3
    x906 = -3465/1024*x4 + (315/64)*x5 - x905
    x907 = x467*x906
    x908 = 3.71811071523632e+18*x458*x9
    x909 = -3.3837890625*x4 + 4.921875*x5 - x905
    x910 = x467*x909
    x911 = (3465/256)*x22 - x27
    x912 = 5.07390783087972e+21*x911
    x913 = -2.4609375*x3 + 13.53515625*x5
    x914 = -45045/4096*x1 + (45045/2048)*x4 - x913
    x915 = x86*x914
    x916 = 7.89911228351117e+24*x54*x9
    x917 = -10.997314453125*x1 + 21.99462890625*x4 - x913
    x918 = x86*x917
    x919 = 0.75*x0
    x920 = x41 - x919
    x921 = x46*x920
    x922 = -3.28125*x24 + x449
    x923 = 57326844710022.1*x922
    x924 = -54.986572265625*x23 + x864
    x925 = 1.09330085366536e+28*x924
    x926 = (15/16)*x3 - 35/32*x5
    x927 = 652640480029.394*x926
    x928 = x889*x9
    x929 = 0.9375*x3 - 1.09375*x5
    x930 = 652640480029.394*x929
    x931 = 2.19476146751835e+25*x458
    x932 = 5*x56
    x933 = x179 + 5
    x934 = -x172 + x174 + x175 - x932 + x933
    x935 = x0*x467
    x936 = x934*x935
    x937 = x931*x936
    x938 = -x177 - x932 - x933
    x939 = x931*x938
    x940 = x935*x939
    x941 = -x325 - x707 + 1
    x942 = 48260827353714.3*x45
    x943 = x0*x942
    x944 = e*x943
    x945 = x941*x944
    x946 = -x326 - x707 - 1
    x947 = x943*x946
    x948 = e*x947
    x949 = w + x262
    x950 = np.sin(x949)
    x951 = np.cos(x949)
    x952 = 112*x56
    x953 = 336*x162
    x954 = 168*x57
    x955 = -x954
    x956 = 420*x118
    x957 = 126*x234 + x547 + x955 + x956 + 14
    x958 = 112*x231 + 336*x60 - x952 - x953 - x957
    x959 = 4.77452518969361e+37*x0
    x960 = x746*x959
    x961 = x958*x960
    x962 = w + x225
    x963 = np.sin(x962)
    x964 = np.cos(x962)
    x965 = 112*x231 + 336*x60 - x952 - x953 + x957
    x966 = x960*x965
    x967 = -x59 - x61 - 2.0
    x968 = x0*x967
    x969 = 2.20177959594483e+17*x889
    x970 = x968*x969
    x971 = -x58 - x61 + 2.0
    x972 = x0*x971
    x973 = x969*x972
    x974 = x797 + x8
    x975 = np.sin(x974)
    x976 = np.cos(x974)
    x977 = 1260*x118
    x978 = 378*x234
    x979 = 504*x57
    x980 = 1176*x199
    x981 = -504*x231 + x580 + 210*x593 + x634 - x853
    x982 = -x977 - x978 + x979 + x980 + x981 - 42
    x983 = 7.88321854070312e+41*x0
    x984 = x19*x983
    x985 = x982*x984
    x986 = x8 + x806
    x987 = np.sin(x986)
    x988 = np.cos(x986)
    x989 = x977 + x978 - x979 - x980 + x981 + 42.0
    x990 = x984*x989
    x991 = 3.23192661591661e+33*x54
    x992 = -x653 - x657 - x833
    x993 = x0*x86
    x994 = x992*x993
    x995 = x991*x994
    x996 = -x302 - x516 - x637 - x653 - x656 + 14
    x997 = x991*x996
    x998 = x993*x997
    x999 = 45*x57
    x1000 = 30*x56
    x1001 = x1000 + x816
    x1002 = -x1001 - x822 + x999 - 5
    x1003 = 2.47942202985548e+29*x0
    x1004 = x1003*x38
    x1005 = x1002*x1004
    x1006 = -x1001 - x819 - x841 - x999 + 5
    x1007 = x1004*x1006
    x1008 = 8*x56
    x1009 = x381 + 2
    x1010 = -x1008 - x1009 - x379
    x1011 = 2.29601576265127e+21*x0
    x1012 = x1011*x456
    x1013 = x1010*x1012
    x1014 = -x1008 + x1009 + x376 + x377
    x1015 = x1012*x1014
    x1016 = x340 + x8
    x1017 = np.sin(x1016)
    x1018 = np.cos(x1016)
    x1019 = x148 + x556
    x1020 = x559 - 1.0
    x1021 = x100 + x1020
    x1022 = -x1021
    x1023 = x652 - 4.0
    x1024 = x1023 + x301
    x1025 = x1019*x296 + x1022*x300 + x1024*x305 - x292 + x306
    x1026 = w + x136
    x1027 = np.sin(x1026)
    x1028 = np.cos(x1026)
    x1029 = -x419
    x1030 = x559 + 2.0
    x1031 = -x1030
    x1032 = x97 + 1.0
    x1033 = x425 + 5.0
    x1034 = x1029 + x1031*x424 + x1032*x422 + x1033*x427 - 1.09375
    x1035 = -x1019
    x1036 = x1020 + x66
    x1037 = x1035*x491 + x1036*x424 + x488 + 3.28125*x57 + 3.28125
    x1038 = -x1023 - x545
    x1039 = 84.0*x56
    x1040 = 210*x57
    x1041 = x1039 + x1040 - 14.0
    x1042 = x1021*x561 + x1035*x558 + x1038*x563 + x1041*x567 + x555 + x568
    x1043 = -x1032
    x1044 = x297 + 2.0
    x1045 = x1043*x715 + x1044*x717 + x713 + 0.9375
    x1046 = -35.0*x56 - 5.0
    x1047 = x853 + 14.0
    x1048 = x1030*x850 + x1043*x851 + x1046*x305 + x1047*x855 + x848 + 1.23046875
    x1049 = 0.75*x56
    x1050 = x69 + 1.0
    x1051 = -x1049 + (15/16)*x1050*x3 - 0.75
    x1052 = x1051*x871
    x1053 = 3.75*x56
    x1054 = x148 + x294
    x1055 = -x1053 + (105/32)*x1054*x3 - 1.875*x57 - 1.875
    x1056 = e*x1052*(-2.84663988257272e-5*x1027 + 5.89772619674322e-6*x1028) + x1025*x109*(-4.94610544005789e-7*x1017 + 3.1750217626037e-7*x1018) + x1034*x319*(-5.47764409510595e-6*x1027 - 8.83386032513283e-8*x1028) + x1037*x476*(1.09232625193902e-6*x1017 - 5.39857315699859e-7*x1018) + x1042*x524*(-4.25884464445994e-8*x1017 - 3.18787653039088e-8*x1018) + x1045*x699*(8.69203554877188e-7*x1027 - 3.5284374029458e-6*x1028) + x1048*x747*(-1.20499967550068e-6*x1027 + 5.46825298616797e-8*x1028) + x1055*x890*(3.18221440738855e-6*x1017 - 3.02797674808195e-6*x1018)
    x1057 = x1005*(7.63892521192145e-11*x813 + 7.40192922474978e-11*x814) + x1007*(7.63892521192145e-11*x838 - 7.40192922474978e-11*x839) + x1013*(6.22342154197505e-10*x373 - 2.14324371360738e-8*x374) + x1015*(6.22342154197505e-10*x431 + 2.14324371360738e-8*x432) + x1056 + x110*(1.93372595556997e-8*x91 - 3.82230038209537e-8*x92) + x135*(-8.89708640195944e-9*x115 + 1.37397940431602e-9*x116) + x155*(1.50482322083058e-6*x138 + 6.03239310464525e-9*x139) + x185*(1.05644790431213e-9*x160 + 4.51097880903126e-10*x161) + x20*x21 + x20*x7 + x206*(9.25105815436851e-11*x190 + 1.18291716961704e-10*x191) + x222*(1.05644790431213e-9*x209 - 4.51097880903126e-10*x210) + x237*(6.35432742037506e-12*x227 + 5.38549660383112e-12*x228) + x251*(1.50482322083058e-6*x240 - 6.03239310464525e-9*x241) + x261*(9.25105815436851e-11*x254 - 1.18291716961704e-10*x255) + x267*(6.35432742037506e-12*x264 - 5.38549660383112e-12*x265) + x284*(-8.89708640195944e-9*x270 - 1.37397940431602e-9*x271) + x29*x39 - x291*x308 + x320*(3.37128745252744e-11*x313 + 2.97455498890218e-11*x314) + x339*(2.39772469945409e-7*x322 + 6.39505856249416e-8*x323) + x353*(2.39772469945409e-7*x342 - 6.39505856249416e-8*x343) + x364*(7.37886275079827e-11*x355 - 3.7825910081444e-10*x356) + x371*(7.37886275079827e-11*x366 + 3.7825910081444e-10*x367) - x39*x912 + x392*(1.60785391674535e-9*x373 + 1.79159092745607e-9*x374) + x408*(-8.41362583908354e-9*x394 + 3.32458288051716e-8*x395) + x414*(3.37128745252744e-11*x411 - 2.97455498890218e-11*x412) - x418*x429 + x42*x46*x47 + x439*(1.60785391674535e-9*x431 - 1.79159092745607e-9*x432) + x447*(-8.41362583908354e-9*x441 - 3.32458288051716e-8*x442) + x451*x457 - x457*x923 + x469*(-2.58713916937584e-9*x115 + 3.06768664405555e-8*x116) - x47*x921 + x477*(2.18886231958567e-9*x190 - 3.36133901991209e-9*x191) + x479*(2.18886231958567e-9*x254 + 3.36133901991209e-9*x255) + x486*(-4.08580688583677e-6*x138 + 2.42500394191311e-6*x139) + x487*x494 + x498*(-1.42189834118728e-7*x52 + 1.37654352627358e-7*x53) + x501*(-2.58713916937584e-9*x270 - 3.06768664405555e-8*x271) + x506*(-4.08580688583677e-6*x240 - 2.42500394191311e-6*x241) + x510*(-1.42189834118728e-7*x91 - 1.37654352627358e-7*x92) + x525*(7.76427363229976e-9*x91 + 5.7252389295325e-9*x92) + x537*(-1.17806447595035e-6*x138 + 1.0358469204293e-6*x139) + x553*(8.17133897153459e-13*x227 + 6.50218686252389e-13*x228) - x554*x570 + x573*(8.17133897153459e-13*x264 - 6.50218686252389e-13*x265) + x586*(4.21152267970529e-9*x270 + 1.85644700154764e-9*x271) + x601*(-7.89021028371753e-15*x590 + 1.42621310964684e-14*x591) + x610*(7.76427363229976e-9*x52 - 5.7252389295325e-9*x53) + x616*(4.21152267970529e-9*x115 - 1.85644700154764e-9*x116) + x629*(-3.81172098514058e-11*x209 + 8.61636262652594e-11*x210) + x643*(1.60622945275753e-12*x190 + 2.90772375501745e-11*x191) + x648*(1.60622945275753e-12*x254 - 2.90772375501745e-11*x255) + x669*(-1.56085961235026e-12*x650 + 6.71981217755168e-12*x651) + x675*(-1.17806447595035e-6*x240 - 1.0358469204293e-6*x241) + x684*(-1.56085961235026e-12*x677 - 6.71981217755168e-12*x678) + x688*(-3.81172098514058e-11*x160 - 8.61636262652594e-11*x161) + x694*(-7.89021028371753e-15*x691 - 1.42621310964684e-14*x692) + x700*(1.69563677817661e-7*x342 + 7.11650409195949e-7*x343) + x706*(-1.5178139351361e-8*x394 + 2.87407372011456e-7*x395) + x711*(1.69563677817661e-7*x322 - 7.11650409195949e-7*x323) + x712*x719 + x723*(-7.67474944193523e-9*x355 + 6.76967206996227e-9*x356) + x729*(-1.5178139351361e-8*x441 - 2.87407372011456e-7*x442) + x731*(-7.67474944193523e-9*x366 - 6.76967206996227e-9*x367) + x748*(1.502521211696e-8*x441 - 1.66251508284257e-8*x442) + x753*(1.502521211696e-8*x394 + 1.66251508284257e-8*x395) + x760*(-9.62806699477347e-8*x342 + 1.3406950669359e-7*x343) + x769*(7.46324808137975e-12*x313 + 2.23641829609004e-13*x314) + x773*(7.46324808137975e-12*x411 - 2.23641829609004e-13*x412) + x783*(1.59210906090832e-10*x355 + 3.60544200058049e-10*x356) + x787*(1.59210906090832e-10*x366 - 3.60544200058049e-10*x367) + x794*(-1.21423645406312e-9*x373 + 1.61173491627614e-9*x374) + x805*(7.29444125482842e-14*x799 - 1.91340436625724e-13*x800) + x811*(7.29444125482842e-14*x808 + 1.91340436625724e-13*x809) + x829*(-3.97695045930581e-11*x813 + 2.79730980704444e-11*x814) + x836*(-9.62806699477347e-8*x322 - 1.3406950669359e-7*x323) + x846*(-3.97695045930581e-11*x838 - 2.79730980704444e-11*x839) - x847*x857 + x862*(-1.21423645406312e-9*x431 - 1.61173491627614e-9*x432) + x866*x867 - x867*x925 + x873*(-1.71281736008093e-6*x441 + 2.45416500876222e-7*x442) + x876*(-1.71281736008093e-6*x394 - 2.45416500876222e-7*x395) - x877*x881 + x88*(1.93372595556997e-8*x52 + 3.82230038209537e-8*x53) + x891*(2.54092521464849e-7*x115 - 1.65286524628759e-7*x116) + x893*(2.54092521464849e-7*x270 + 1.65286524628759e-7*x271) + x894*x897 + x901*(3.16447735929272e-6*x138 - 1.13925135459742e-5*x139) + x904*(3.16447735929272e-6*x240 + 1.13925135459742e-5*x241) - x907*x908 - x908*x910 - x915*x916 - x916*x918 - x927*x928 - x928*x930 + x937*(-1.67026313872155e-8*x209 + 2.38054298502945e-9*x210) + x940*(-1.67026313872155e-8*x160 - 2.38054298502945e-9*x161) + x945*(1.67071080669887e-6*x342 + 4.84286276536557e-6*x343) + x948*(1.67071080669887e-6*x322 - 4.84286276536557e-6*x323) + x961*(-7.22011525564534e-13*x950 + 4.28101471681974e-13*x951) + x966*(-7.22011525564534e-13*x963 - 4.28101471681974e-13*x964) + x970*(-1.60796708177047e-6*x52 + 1.59987737414091e-7*x53) + x973*(-1.60796708177047e-6*x91 - 1.59987737414091e-7*x92) + x985*(-2.26001853433503e-15*x975 + 1.76818510628331e-13*x976) + x990*(-2.26001853433503e-15*x987 - 1.76818510628331e-13*x988) + x995*(3.3262035518161e-11*x650 + 1.54706092524183e-11*x651) + x998*(3.3262035518161e-11*x677 - 1.54706092524183e-11*x678)
    x1058 = np.sqrt(x16)
    x1059 = a**(-3.0)
    x1060 = np.sqrt(x1059)
    x1061 = x1060**(-1.0)
    x1062 = 0.0142816344494344*x1061/a**2
    x1063 = x1062/(e + 1.0e-12)
    x1064 = x1058*x1063
    x1065 = x1062/(x1058*(x0 + 1.0e-12))
    x1066 = 3/2 - 3/2*x57
    x1067 = np.sin(x340)
    x1068 = np.cos(x340)
    x1069 = x16**(-3/2)
    x1070 = x1059*x1069
    x1071 = np.sin(x268)
    x1072 = np.cos(x268)
    x1073 = (315/8)*x118 - 315/4*x57 + 315/8
    x1074 = (3/2)*x10 + 1
    x1075 = x1074*x888
    x1076 = 4.47345695684029e+16*x1075
    x1077 = np.sin(x136)
    x1078 = np.cos(x136)
    x1079 = x24*x56
    x1080 = (105/16)*x1079
    x1081 = (15/4)*x0*x56 - x1080
    x1082 = (15/4)*x57
    x1083 = -x703
    x1084 = x1082 + x1083*x715 - 3.75
    x1085 = x23*x56
    x1086 = x0*x292
    x1087 = (225225/2048)*x1085 - x1086 + x24*x732
    x1088 = -x1087 + (45045/256)*x22*x56
    x1089 = (21/2)*x10 + (105/8)*x11 + (35/16)*x12 + 1
    x1090 = x1089*x84
    x1091 = x1090*x55
    x1092 = np.sin(x89)
    x1093 = np.cos(x89)
    x1094 = 6*x60
    x1095 = x1094 + x398
    x1096 = -x129
    x1097 = -x1000 + x661
    x1098 = x1095*x65 + x1096*x73 + x1097*x82
    x1099 = np.sin(x252)
    x1100 = np.cos(x252)
    x1101 = -150*x118 + 70*x199 + x843 - 10.0
    x1102 = -x1101
    x1103 = x1102*x203 - 2027025/32*x118 + (675675/32)*x199 + 63344.53125*x57 - 21114.84375
    x1104 = 9.84375*x57
    x1105 = -x94
    x1106 = 70*x57 - 10.0
    x1107 = -x1106
    x1108 = 20*x57 - 4.0
    x1109 = x1104 + x1105*x296 + x1107*x305 + x1108*x300 - 9.84375
    x1110 = np.sin(x207)
    x1111 = np.cos(x207)
    x1112 = 20*x162
    x1113 = 20.0*x56
    x1114 = x1112 + x1113
    x1115 = x1114 + x211
    x1116 = 70*x162 - 100*x60 + x778
    x1117 = -x1116
    x1118 = x1115*x183 + x1117*x171
    x1119 = np.sin(x225)
    x1120 = np.cos(x225)
    x1121 = (212837625/64)*x118 - 70945875/32*x199 + (70945875/128)*x234 - 70945875/32*x57 + 70945875/128
    x1122 = 20*x118 + 4.0
    x1123 = x1122 + x276
    x1124 = -x1123
    x1125 = x1124*x133 + (31185/64)*x118 + x127*x168 - 974.53125*x57 + 487.265625
    x1126 = x22*x56
    x1127 = x0*x488
    x1128 = (3465/128)*x1126 + x1127
    x1129 = -x1128 + (945/32)*x24*x56
    x1130 = 5*x10 + (15/8)*x11 + 1
    x1131 = x1130*x465
    x1132 = x1131*x459
    x1133 = (155925/16)*x57
    x1134 = -x1133 + (155925/16)*x118 - 51975/16*x199 + 51975/16
    x1135 = -6.5625*x57
    x1136 = -x1108
    x1137 = x1135 + x1136*x424 + x491*x703 + 6.5625
    x1138 = x1122 + x123
    x1139 = x1138*x401 - 2835/16*x118 + 354.375*x57 - 177.1875
    x1140 = -x1095
    x1141 = x1140*x495 + x334*x461
    x1142 = x534*x56
    x1143 = x0*x555
    x1144 = -175.95703125*x1079 + 659.8388671875*x1126 + (14549535/32768)*x1142 + x1143
    x1145 = -x1144 + (3828825/4096)*x23*x56
    x1146 = 1.23293986687242e+36*x1145
    x1147 = 18*x10 + (189/4)*x11 + (105/4)*x12 + (315/128)*x13 + 1
    x1148 = x1147*x17
    x1149 = x1148*x15
    x1150 = -x398 - 6.0*x60
    x1151 = -x661 + x778
    x1152 = -x564 + 252*x60
    x1153 = x1150*x511 + x1151*x515 + x1152*x522 + x129*x513
    x1154 = 1.23293986687242e+36*x1149
    x1155 = 784*x199
    x1156 = -x1155 + 840*x118 + 252*x234 - 336.0*x57 + 28.0
    x1157 = x1156*x550 - 28267497.0703125*x118 + (1206079875/64)*x199 - 1206079875/256*x234 + 18844998.046875*x57 - 4711249.51171875
    x1158 = -x1114 - x617
    x1159 = 280*x60 - x645
    x1160 = x1116*x619 + x1158*x621 + x1159*x627
    x1161 = -x213
    x1162 = 252*x118 + x955 + 12.0
    x1163 = x1123*x579 + x1161*x577 + x1162*x584 - 1055.7421875*x118 + 2111.484375*x57 - 1055.7421875
    x1164 = -252*x199 - 180.0*x57 + x956 + 12.0
    x1165 = x1101*x633 + x1164*x640 + 237541.9921875*x118 - 10135125/128*x199 - 237541.9921875*x57 + 79180.6640625
    x1166 = np.sin(x689)
    x1167 = np.cos(x689)
    x1168 = (206239658625/128)*x118 - 206239658625/128*x199 + (206239658625/256)*x234 - 206239658625/256*x57 - 41247931725/256*x598 + 41247931725/256
    x1169 = -13.53515625*x57
    x1170 = 4.0 - 20.0*x57
    x1171 = 28.0 - 252*x57
    x1172 = x1106*x563 + x1169 + x1170*x561 + x1171*x567 + x558*x94 + 13.53515625
    x1173 = np.sin(x409)
    x1174 = np.cos(x409)
    x1175 = 70*x231 - x624
    x1176 = -x1175 + 70.0*x56 - x770
    x1177 = 588*x162
    x1178 = -x1039 - x1177 + 252*x231 + 420*x60
    x1179 = x1176*x660 + x1178*x667
    x1180 = x0*x56
    x1181 = 22214419955.6797*x1070
    x1182 = 20*x56
    x1183 = -x1112 - x1182 - x211
    x1184 = x0*x931
    x1185 = x1183*x1184
    x1186 = np.sin(x806)
    x1187 = np.cos(x806)
    x1188 = 252*x56
    x1189 = 1512*x162
    x1190 = -x1188 - x1189 + 1008*x231 - 252*x593 + 1008*x60
    x1191 = x1190*x983
    x1192 = 70*x56
    x1193 = x1175 - x1192 + x518
    x1194 = x0*x991
    x1195 = x1193*x1194
    x1196 = x1094 + x726
    x1197 = x0*x1196
    x1198 = 2.93570612792644e+17*x1075
    x1199 = x1070*x1180
    x1200 = x3*x56
    x1201 = (45/16)*x1200
    x1202 = np.sin(w)
    x1203 = e*x1202
    x1204 = x1203*x46
    x1205 = -9.43103158893466e-8*x1077 - 1.54435046418598e-9*x1078
    x1206 = x1181*x1205
    x1207 = np.cos(x8)
    x1208 = 326320240014.697*x1207*x889
    x1209 = x0*x1053
    x1210 = 651.549312134827*x1067 + 994374.029246504*x1068
    x1211 = x5*x56
    x1212 = (1575/128)*x1211 + x713
    x1213 = 57326844710022.1*x1202*x456
    x1214 = 1.85905535761816e+18*x458
    x1215 = -4.84286276536557e-6*x322 - 1.67071080669887e-6*x323
    x1216 = 2*x0
    x1217 = x0*x725
    x1218 = -x1216 + x1217
    x1219 = 4.84286276536557e-6*x342 - 1.67071080669887e-6*x343
    x1220 = 3.28125*x1180
    x1221 = x1207*x1214
    x1222 = x1221*x467
    x1223 = x1215*x946
    x1224 = e*x56*x942
    x1225 = x1219*x941
    x1226 = -7.99938687070455e-8*x1092 + 8.03983540885234e-7*x1093
    x1227 = x1198*x1226
    x1228 = 5.89772619674322e-6*x1027 + 2.84663988257272e-5*x1028
    x1229 = (45/16)*x24
    x1230 = x56*x898
    x1231 = 8.26432623143797e-8*x1071 - 1.27046260732425e-7*x1072
    x1232 = x0*x60
    x1233 = 2.45416500876222e-7*x441 + 1.71281736008093e-6*x442
    x1234 = (45/8)*x0
    x1235 = (45/4)*x1180
    x1236 = x0*x57
    x1237 = (135/8)*x1236
    x1238 = -5.89772619674322e-6*x416 + 2.84663988257272e-5*x417
    x1239 = -2.45416500876222e-7*x394 + 1.71281736008093e-6*x395
    x1240 = 18*x1236
    x1241 = x4*x56
    x1242 = x1029 + 22.1484375*x1200 + (105105/2048)*x1241
    x1243 = 5.07390783087972e+21*x1202*x38
    x1244 = 7.99938687070455e-8*x52 + 8.03983540885234e-7*x53
    x1245 = x1244*x969
    x1246 = -7.99938687070455e-8*x91 + 8.03983540885234e-7*x92
    x1247 = x1246*x969
    x1248 = x0*x77
    x1249 = -x1248
    x1250 = x0*x375
    x1251 = x1249 + x1250
    x1252 = 3.94955614175558e+24*x54
    x1253 = -4.921875*x0*x56 + 54.140625*x1079
    x1254 = x1207*x1252
    x1255 = x1254*x86
    x1256 = -8.26432623143797e-8*x115 - 1.27046260732425e-7*x116
    x1257 = (105/2)*x0
    x1258 = (315/2)*x1236
    x1259 = 105*x1232
    x1260 = 8.26432623143797e-8*x270 - 1.27046260732425e-7*x271
    x1261 = 1.19027149251473e-9*x1110 + 8.35131569360775e-9*x1111
    x1262 = x1131*x1261
    x1263 = x1183*x1262
    x1264 = -5.69625677298712e-6*x1077 - 1.58223867964636e-6*x1078
    x1265 = 1.68066950995605e-9*x1099 - 1.09443115979283e-9*x1100
    x1266 = x0*x162
    x1267 = -120*x1236
    x1268 = x0*x118
    x1269 = 100*x1268
    x1270 = -1.51398837404098e-6*x1067 - 1.59110720369427e-6*x1068
    x1271 = x56*x696
    x1272 = (6891885/32768)*x1*x56 - 40.60546875*x1200 + 219.9462890625*x1211 + x848
    x1273 = 1.09330085366536e+28*x1202*x746
    x1274 = -2.14324371360738e-8*x373 - 6.22342154197505e-10*x374
    x1275 = x1010*x1274
    x1276 = 2.29601576265127e+21*x456*x56
    x1277 = 2.14324371360738e-8*x431 - 6.22342154197505e-10*x432
    x1278 = x1014*x1277
    x1279 = 6.0550368158715e+30*x15
    x1280 = -5.69625677298712e-6*x138 - 1.58223867964636e-6*x139
    x1281 = 1.875*x3
    x1282 = -35/8*x5
    x1283 = (15/8)*x56
    x1284 = x56*x715
    x1285 = x0*x122
    x1286 = -x1285
    x1287 = x0*x164
    x1288 = -x0*x120 + x1287
    x1289 = 494.879150390625*x1126 + 6.767578125*x1180 - 117.3046875*x24*x56
    x1290 = x1207*x1279*x17
    x1291 = x1290*x14
    x1292 = 5.69625677298712e-6*x240 - 1.58223867964636e-6*x241
    x1293 = x56*x991
    x1294 = -7.73530462620917e-12*x1173 - 1.66310177590805e-11*x1174
    x1295 = x1090*x1294
    x1296 = x1193*x1295
    x1297 = 6.76967206996227e-9*x355 + 7.67474944193523e-9*x356
    x1298 = (4725/4)*x1232
    x1299 = (4725/16)*x0 - 14175/8*x1236 + (23625/16)*x1268
    x1300 = -6.76967206996227e-9*x366 + 7.67474944193523e-9*x367
    x1301 = -3.5284374029458e-6*x1027 - 8.69203554877188e-7*x1028
    x1302 = (1575/128)*x22
    x1303 = x482*x56
    x1304 = 2.69274830191556e-12*x1119 - 3.17716371018753e-12*x1120
    x1305 = x0*x231
    x1306 = 630*x1236
    x1307 = 1050*x1268
    x1308 = x0*x199
    x1309 = -x1307 + 490*x1308
    x1310 = 3.5284374029458e-6*x416 - 8.69203554877188e-7*x417
    x1311 = -1.19027149251473e-9*x160 + 8.35131569360775e-9*x161
    x1312 = x1311*x939
    x1313 = x467*x56
    x1314 = 1.19027149251473e-9*x209 + 8.35131569360775e-9*x210
    x1315 = x1314*x934
    x1316 = 1.21250197095655e-6*x1077 + 2.04290344291838e-6*x1078
    x1317 = x3*x57
    x1318 = x5*x57
    x1319 = -1.51398837404098e-6*x1017 - 1.59110720369427e-6*x1018
    x1320 = 4.0*x0
    x1321 = x0*x1008
    x1322 = -8.84092553141657e-14*x1186 + 1.13000926716752e-15*x1187
    x1323 = x1149*x1322
    x1324 = 7.88321854070312e+41*x56
    x1325 = 1.51398837404098e-6*x289 - 1.59110720369427e-6*x290
    x1326 = 4*x0
    x1327 = -x1326
    x1328 = x1321 + x1327
    x1329 = 7.40192922474978e-11*x813 - 7.63892521192145e-11*x814
    x1330 = x1002*x1329
    x1331 = 2.47942202985548e+29*x38*x56
    x1332 = -7.40192922474978e-11*x838 - 7.63892521192145e-11*x839
    x1333 = x1006*x1332
    x1334 = 60*x1180
    x1335 = -x1334
    x1336 = 100*x0*x60
    x1337 = x0*x173
    x1338 = -x0*x821
    x1339 = 5*x0 + x1337 + x1338
    x1340 = x931*x935
    x1341 = -x1336
    x1342 = x1334 + x1341
    x1343 = -1.68066950995605e-9*x190 - 1.09443115979283e-9*x191
    x1344 = (467775/32)*x1266
    x1345 = (155925/32)*x0 - 467775/16*x1236 + (779625/32)*x1268
    x1346 = 1.68066950995605e-9*x254 - 1.09443115979283e-9*x255
    x1347 = 7.73530462620917e-12*x650 - 1.66310177590805e-11*x651
    x1348 = x1347*x992
    x1349 = -7.73530462620917e-12*x677 - 1.66310177590805e-11*x678
    x1350 = x1349*x997
    x1351 = -7.1310655482342e-15*x1166 + 3.94510514185877e-15*x1167
    x1352 = x0*x593
    x1353 = x0*x234
    x1354 = -30*x0
    x1355 = 150*x0*x118
    x1356 = 180*x1236
    x1357 = -x1355 + x1356
    x1358 = x0*x624 - 300*x1232
    x1359 = 90*x1180 + x1358
    x1360 = -2.6992865784993e-7*x1067 - 5.46163125969509e-7*x1068
    x1361 = x495*x56
    x1362 = x334*x56
    x1363 = 1.21250197095655e-6*x138 + 2.04290344291838e-6*x139
    x1364 = 3.28125*x3
    x1365 = -10395/512*x4
    x1366 = (105/32)*x56
    x1367 = x491*x56
    x1368 = x424*x56
    x1369 = 2.97455498890218e-11*x313 - 3.37128745252744e-11*x314
    x1370 = (4729725/128)*x0
    x1371 = (42567525/128)*x1236
    x1372 = (33108075/128)*x1308
    x1373 = (70945875/128)*x1268
    x1374 = (14189175/64)*x1180 - 14189175/32*x1232 + (14189175/64)*x1266
    x1375 = -2.97455498890218e-11*x411 - 3.37128745252744e-11*x412
    x1376 = -1.53384332202777e-8*x1071 + 1.29356958468792e-9*x1072
    x1377 = x385*x56
    x1378 = -48*x1180
    x1379 = 80*x1232
    x1380 = 28*x0
    x1381 = x0*x1188
    x1382 = x0*x956
    x1383 = -x1382
    x1384 = x0*x546 + x1383
    x1385 = 840*x1232
    x1386 = x0*x1177
    x1387 = x1385 - x1386
    x1388 = x991*x993
    x1389 = -x1385
    x1390 = x1380 + x1381
    x1391 = -1.21250197095655e-6*x240 + 2.04290344291838e-6*x241
    x1392 = -8.83386032513283e-8*x1027 + 5.47764409510595e-6*x1028
    x1393 = 22.1484375*x24
    x1394 = (105105/2048)*x23
    x1395 = x329*x56
    x1396 = x147*x56
    x1397 = 4.28101471681974e-13*x950 + 7.22011525564534e-13*x951
    x1398 = x1397*x958
    x1399 = 4.77452518969361e+37*x56*x746
    x1400 = -4.28101471681974e-13*x963 + 7.22011525564534e-13*x964
    x1401 = x1400*x965
    x1402 = 3.01619655232262e-9*x1077 - 7.52411610415292e-7*x1078
    x1403 = x4*x57
    x1404 = 8.83386032513283e-8*x416 + 5.47764409510595e-6*x417
    x1405 = -2.69274830191556e-12*x264 - 3.17716371018753e-12*x265
    x1406 = (14189175/16)*x0
    x1407 = (127702575/16)*x1236
    x1408 = (99324225/16)*x1308
    x1409 = (212837625/16)*x1268
    x1410 = (14189175/8)*x1180 - 42567525/8*x1266 + (14189175/4)*x1305
    x1411 = 2.69274830191556e-12*x227 - 3.17716371018753e-12*x228
    x1412 = 2.87407372011456e-7*x394 + 1.5178139351361e-8*x395
    x1413 = 19.6875*x0
    x1414 = (945/16)*x1236
    x1415 = 6.0*x0
    x1416 = -x1415
    x1417 = x1337 + x1416
    x1418 = -2.87407372011456e-7*x441 + 1.5178139351361e-8*x442
    x1419 = -x1337
    x1420 = x1248 + x1419
    x1421 = 7.11650409195949e-7*x342 - 1.69563677817661e-7*x343
    x1422 = 2.0*x0
    x1423 = (105/16)*x56
    x1424 = x0*x1182
    x1425 = -7.11650409195949e-7*x322 - 1.69563677817661e-7*x323
    x1426 = x1327 + x1424
    x1427 = -6.88271763136791e-8*x1092 + 7.1094917059364e-8*x1093
    x1428 = (945/32)*x56
    x1429 = x1240 + x1416
    x1430 = 12*x0
    x1431 = x0*x165
    x1432 = x401*x56
    x1433 = 1008*x1236
    x1434 = x0*x1155
    x1435 = -336*x1180
    x1436 = -2352*x1266
    x1437 = 1680*x1232
    x1438 = 1008*x0*x231 + x1435 + x1436 + x1437
    x1439 = 8.84092553141657e-14*x975 + 1.13000926716752e-15*x976
    x1440 = x1439*x982
    x1441 = x1324*x19
    x1442 = -8.84092553141657e-14*x987 + 1.13000926716752e-15*x988
    x1443 = x1442*x989
    x1444 = -1.91340436625724e-13*x799 - 7.29444125482842e-14*x800
    x1445 = (6512831325/32)*x1232
    x1446 = (2170943775/32)*x1305
    x1447 = (2170943775/256)*x0 - 6512831325/64*x1236 + (32564156625/128)*x1268 - 15196606425/64*x1308 + (19538493975/256)*x1353
    x1448 = 1.91340436625724e-13*x808 - 7.29444125482842e-14*x809
    x1449 = 3.01619655232262e-9*x138 - 7.52411610415292e-7*x139
    x1450 = (315/64)*x56
    x1451 = x296*x56
    x1452 = x300*x56
    x1453 = x305*x56
    x1454 = 4.921875*x3
    x1455 = -45045/512*x1 + x1454
    x1456 = 1.58751088130185e-7*x1067 + 2.47305272002894e-7*x1068
    x1457 = x56*x65
    x1458 = x56*x82
    x1459 = x56*x73
    x1460 = 5.17923460214652e-7*x1077 + 5.89032237975176e-7*x1078
    x1461 = -5.91458584808521e-11*x1099 - 4.62552907718425e-11*x1100
    x1462 = x56*x824
    x1463 = 180.0*x1180
    x1464 = -600*x1232 + 420*x1266 + x1463
    x1465 = -3.01619655232262e-9*x240 - 7.52411610415292e-7*x241
    x1466 = 5.46825298616797e-8*x1027 + 1.20499967550068e-6*x1028
    x1467 = 40.60546875*x24
    x1468 = 219.9462890625*x22
    x1469 = (6891885/32768)*x534
    x1470 = x56*x754
    x1471 = x56*x756
    x1472 = x529*x56
    x1473 = -2.6992865784993e-7*x1017 - 5.46163125969509e-7*x1018
    x1474 = 6.5625*x0
    x1475 = x0*x380
    x1476 = x1320 + x1475
    x1477 = 10.0*x0
    x1478 = x0*x1000
    x1479 = -5.46825298616797e-8*x416 + 1.20499967550068e-6*x417
    x1480 = 5040*x1232
    x1481 = 3024*x1305
    x1482 = 1008*x1180
    x1483 = 7056*x1266
    x1484 = x0*x977
    x1485 = x0*x979
    x1486 = -3528*x0*x199 - 126*x0 + 1890*x1353 + x1484 + x1485
    x1487 = -1.53384332202777e-8*x270 + 1.29356958468792e-9*x271
    x1488 = 236.25*x0
    x1489 = -945/2*x1232
    x1490 = 12.0*x0
    x1491 = -x1490
    x1492 = x1431 + x1491
    x1493 = x0*x104
    x1494 = -x1493
    x1495 = x0*x818
    x1496 = x1494 + x1495
    x1497 = 2.6992865784993e-7*x289 - 5.46163125969509e-7*x290
    x1498 = 10*x0
    x1499 = -x1498
    x1500 = x1478 + x1499
    x1501 = 1.53384332202777e-8*x115 + 1.29356958468792e-9*x116
    x1502 = x1249 + x1495
    x1503 = -x1431
    x1504 = x1430 + x1503
    x1505 = 7.1310655482342e-15*x590 + 3.94510514185877e-15*x591
    x1506 = (68746552875/256)*x0
    x1507 = (1031198293125/128)*x1268
    x1508 = (618718975875/256)*x1353
    x1509 = (206239658625/64)*x1236
    x1510 = (481225870125/64)*x1308
    x1511 = -206239658625/64*x0*x231 - 206239658625/256*x0*x56 + x0*x592 + x0*x596 + (343732764375/256)*x1352
    x1512 = -7.1310655482342e-15*x691 + 3.94510514185877e-15*x692
    x1513 = -6.88271763136791e-8*x91 + 7.1094917059364e-8*x92
    x1514 = 3.0*x0
    x1515 = -x1514
    x1516 = x0*x999
    x1517 = x1515 + x1516
    x1518 = 6.88271763136791e-8*x52 + 7.1094917059364e-8*x53
    x1519 = x1478 - x1516
    x1520 = 3*x0 + x1519
    x1521 = 5.17923460214652e-7*x138 + 5.89032237975176e-7*x139
    x1522 = (3465/512)*x56
    x1523 = x558*x56
    x1524 = x56*x561
    x1525 = x56*x563
    x1526 = x56*x567
    x1527 = 6.767578125*x3
    x1528 = -x1527 - 24249225/65536*x2 + 117.3046875*x5
    x1529 = -6.86989702158012e-10*x1071 + 4.44854320097972e-9*x1072
    x1530 = x183*x56
    x1531 = -48.0*x1180
    x1532 = x1379 + x1531
    x1533 = 120*x1180
    x1534 = x0*x635
    x1535 = x171*x56
    x1536 = -2.25548940451563e-10*x1110 - 5.28223952156065e-10*x1111
    x1537 = (135135/128)*x56
    x1538 = 20.0*x0 + x1269
    x1539 = x203*x56
    x1540 = 30.0*x0
    x1541 = -300*x1236 + 350*x1268 + x1540
    x1542 = -5.17923460214652e-7*x240 + 5.89032237975176e-7*x241
    x1543 = -6.39505856249416e-8*x342 - 2.39772469945409e-7*x343
    x1544 = x0*x397
    x1545 = x1422 + x1544
    x1546 = (945/64)*x56
    x1547 = x0*x1192
    x1548 = -1.59393826519544e-8*x1067 + 2.12942232222997e-8*x1068
    x1549 = x511*x56
    x1550 = x513*x56
    x1551 = x522*x56
    x1552 = x515*x56
    x1553 = 3.25109343126195e-13*x1119 - 4.0856694857673e-13*x1120
    x1554 = (654729075/1024)*x1180
    x1555 = 6.39505856249416e-8*x322 - 2.39772469945409e-7*x323
    x1556 = -x1422
    x1557 = x1499 + x1547
    x1558 = -1.91115019104768e-8*x1092 - 9.66862977784983e-9*x1093
    x1559 = (10395/128)*x56
    x1560 = x133*x56
    x1561 = x0*x1040
    x1562 = x127*x56
    x1563 = 3.7825910081444e-10*x366 - 7.37886275079827e-11*x367
    x1564 = 1624.21875*x0
    x1565 = 15.0*x0
    x1566 = 175*x1268
    x1567 = -150*x1236 + x1565 + x1566
    x1568 = x0*x636
    x1569 = -x1568
    x1570 = x1336 + x1569
    x1571 = (259875/32)*x1268
    x1572 = -6496.875*x1180 + (51975/8)*x1232
    x1573 = -3.7825910081444e-10*x355 - 7.37886275079827e-11*x356
    x1574 = 3.32458288051716e-8*x394 + 8.41362583908354e-9*x395
    x1575 = 44.296875*x0
    x1576 = 88.59375*x1180
    x1577 = 105*x1236
    x1578 = -x1565 + x1577
    x1579 = -3.32458288051716e-8*x441 + 8.41362583908354e-9*x442
    x1580 = x1478 - x1577
    x1581 = 1.58751088130185e-7*x1017 + 2.47305272002894e-7*x1018
    x1582 = 9.84375*x0
    x1583 = x0*x778
    x1584 = x1477 + x1583
    x1585 = 28.0*x0
    x1586 = x0*x952
    x1587 = -1.79159092745607e-9*x431 - 1.60785391674535e-9*x432
    x1588 = (10395/64)*x56
    x1589 = 8.0*x0
    x1590 = -x0*x272 + x1287
    x1591 = -x1589 + x1590
    x1592 = x0*x622
    x1593 = x1335 + x1592
    x1594 = -1.58751088130185e-7*x289 + 2.47305272002894e-7*x290
    x1595 = -x1320
    x1596 = -28*x0
    x1597 = x1586 + x1596
    x1598 = 1.79159092745607e-9*x373 - 1.60785391674535e-9*x374
    x1599 = x1286 + x1589
    x1600 = -5.91458584808521e-11*x254 - 4.62552907718425e-11*x255
    x1601 = 31672.265625*x0
    x1602 = 240*x1232
    x1603 = x0*x953
    x1604 = 36.0*x0
    x1605 = x1382 + x1604
    x1606 = -360.0*x1236 + x1603 + x1605
    x1607 = (10135125/64)*x1268
    x1608 = -x0*x194 + (6081075/64)*x1266
    x1609 = 5.91458584808521e-11*x190 - 4.62552907718425e-11*x191
    x1610 = 360*x1236
    x1611 = x1605 - x1610
    x1612 = x1602 - x1603 + x1611
    x1613 = -3.35990608877584e-12*x1173 + 7.80429806175129e-13*x1174
    x1614 = (34459425/512)*x56
    x1615 = x550*x56
    x1616 = 630.0*x1236
    x1617 = 84.0*x0
    x1618 = -1.45386187750873e-11*x1099 - 8.03114726378766e-13*x1100
    x1619 = x56*x660
    x1620 = x56*x667
    x1621 = -1.91115019104768e-8*x91 - 9.66862977784983e-9*x92
    x1622 = x0*x433 + x1493
    x1623 = x0*x564
    x1624 = x0*x954
    x1625 = x1491 + x1624
    x1626 = 1.3406950669359e-7*x342 + 9.62806699477347e-8*x343
    x1627 = (3465/128)*x56
    x1628 = x56*x735
    x1629 = 9.2822350077382e-10*x1071 - 2.10576133985265e-9*x1072
    x1630 = x56*x621
    x1631 = -120.0*x1180 + x1534
    x1632 = x56*x619
    x1633 = x56*x627
    x1634 = -1.3406950669359e-7*x322 + 9.62806699477347e-8*x323
    x1635 = 1.91115019104768e-8*x52 - 9.66862977784983e-9*x53
    x1636 = -x1624
    x1637 = x1430 + x1623 + x1636
    x1638 = -2.25548940451563e-10*x209 - 5.28223952156065e-10*x210
    x1639 = 280*x1268
    x1640 = x1267 + x1639
    x1641 = x1341 + x1568
    x1642 = x0*x216
    x1643 = 5.0*x0 + x1338
    x1644 = x1642 + x1643
    x1645 = -6.86989702158012e-10*x270 + 4.44854320097972e-9*x271
    x1646 = 649.6875*x0
    x1647 = x0*x212 + x1491
    x1648 = 24.0*x0
    x1649 = -x1648
    x1650 = x1624 + x1649
    x1651 = 224*x1232
    x1652 = x1531 + x1651
    x1653 = 2.86261946476625e-9*x1092 - 3.88213681614988e-9*x1093
    x1654 = (45045/256)*x56
    x1655 = -x1540
    x1656 = x56*x577
    x1657 = x56*x579
    x1658 = -84*x0
    x1659 = x56*x584
    x1660 = 2.25548940451563e-10*x160 - 5.28223952156065e-10*x161
    x1661 = x1533 - x1534
    x1662 = x1640 + x1661
    x1663 = x1336 + x1337 + x1643
    x1664 = 6.86989702158012e-10*x115 + 4.44854320097972e-9*x116
    x1665 = x1490 + x1503
    x1666 = 24*x0 + x1636
    x1667 = x1378 + x1651
    x1668 = 4.30818131326297e-11*x1110 + 1.90586049257029e-11*x1111
    x1669 = (2027025/512)*x56
    x1670 = -120.0*x1236
    x1671 = x56*x640
    x1672 = x56*x633
    x1673 = 840*x1236
    x1674 = 60.0*x0
    x1675 = x1484 + x1674
    x1676 = -2.23641829609004e-13*x411 - 7.46324808137975e-12*x412
    x1677 = 277132.32421875*x0
    x1678 = (496621125/256)*x1308
    x1679 = 1470*x1268
    x1680 = -x1679
    x1681 = 42*x0
    x1682 = 882*x1308
    x1683 = 252.0*x1180
    x1684 = x1386 + x1683
    x1685 = x1389 + x1684
    x1686 = (1064188125/256)*x1268
    x1687 = 2494190.91796875*x1236
    x1688 = -1662793.9453125*x1180 - 212837625/128*x1266
    x1689 = 2.23641829609004e-13*x313 - 7.46324808137975e-12*x314
    x1690 = -x1306
    x1691 = 42.0*x0
    x1692 = -1.59393826519544e-8*x1017 + 2.12942232222997e-8*x1018
    x1693 = 13.53515625*x0
    x1694 = -x1143
    x1695 = x0*x544
    x1696 = 420*x1180
    x1697 = 1.59393826519544e-8*x289 + 2.12942232222997e-8*x290
    x1698 = 1.66251508284257e-8*x394 - 1.502521211696e-8*x395
    x1699 = 81.2109375*x0
    x1700 = -162.421875*x1180
    x1701 = 243.6328125*x1236
    x1702 = x56*x789
    x1703 = -x1691
    x1704 = 378*x1236
    x1705 = -1.66251508284257e-8*x441 - 1.502521211696e-8*x442
    x1706 = 3.25109343126195e-13*x227 - 4.0856694857673e-13*x228
    x1707 = 7537999.21875*x0
    x1708 = (1688511825/32)*x1308
    x1709 = 112.0*x0
    x1710 = 3920*x1268
    x1711 = 2352*x1308
    x1712 = 1680*x1305 + x1436 + x1695
    x1713 = 67841992.96875*x1236
    x1714 = -15075998.4375*x1180 + (723647925/16)*x1266 - 241215975/8*x1305
    x1715 = -3.25109343126195e-13*x264 - 4.0856694857673e-13*x265
    x1716 = -2.79730980704444e-11*x838 + 3.97695045930581e-11*x839
    x1717 = (2027025/256)*x56
    x1718 = 90.0*x1180 + x1358
    x1719 = 756*x1266 + x1389
    x1720 = 2.79730980704444e-11*x813 + 3.97695045930581e-11*x814
    x1721 = -1.61173491627614e-9*x431 + 1.21423645406312e-9*x432
    x1722 = (135135/256)*x56
    x1723 = -168*x1180
    x1724 = 504*x1232 + x1723
    x1725 = -3.60544200058049e-10*x366 - 1.59210906090832e-10*x367
    x1726 = 5278.7109375*x0
    x1727 = 21114.84375*x1180
    x1728 = x1601*x57
    x1729 = (3378375/128)*x1268
    x1730 = -420*x1236
    x1731 = 630*x1268 + x1540 + x1730
    x1732 = 3.60544200058049e-10*x355 - 1.59210906090832e-10*x356
    x1733 = 1.61173491627614e-9*x373 + 1.21423645406312e-9*x374
    x1734 = 2.86261946476625e-9*x91 - 3.88213681614988e-9*x92
    x1735 = -2.86261946476625e-9*x52 - 3.88213681614988e-9*x53
    x1736 = -3.35990608877584e-12*x677 + 7.80429806175129e-13*x678
    x1737 = x1384 + x1585
    x1738 = 1960*x1232
    x1739 = 1764*x1266
    x1740 = 14.0*x0 + 1470*x1308 + x1680
    x1741 = 3.35990608877584e-12*x650 + 7.80429806175129e-13*x651
    x1742 = 9.2822350077382e-10*x270 - 2.10576133985265e-9*x271
    x1743 = 1407.65625*x0
    x1744 = 4222.96875*x1236
    x1745 = -1.45386187750873e-11*x254 - 8.03114726378766e-13*x255
    x1746 = 118770.99609375*x0
    x1747 = 1260*x1266 + x1389
    x1748 = 712625.9765625*x1236
    x1749 = x0*x631 - 91216125/256*x1266
    x1750 = -9.2822350077382e-10*x115 - 2.10576133985265e-9*x116
    x1751 = 1.45386187750873e-11*x190 - 8.03114726378766e-13*x191
    x1752 = 4.30818131326297e-11*x209 + 1.90586049257029e-11*x210
    x1753 = x1307 + x1730
    x1754 = -4.30818131326297e-11*x160 + 1.90586049257029e-11*x161
    x1755 = x1065*(x0*x1227*(6*x0 - x1240) + x0*x1245*x1251 + x0*x1247*(x1248 + x1250) + x1004*x1329*(-x1354 - x1357 - x1359) + x1004*x1332*(30*x0 + x1355 - x1356 + x1359) + x1012*x1274*(8*x0 + x1286 + x1288) + x1012*x1277*(8*x0 - x1285 - x1288) + 435093653352.929*x1075*(x1080 - x1209) + x1076*x1231*((315/2)*x0*x56 - 315/2*x1232) + x1076*x1264*(x1082 - x3*x724 - 15/4*x3 + (105/16)*x5) + x1076*x1270*((315/8)*x1079 + x1083*x1271 - 15/2*x1180) + x109*x1405*(x1406 - x1407 - x1408 + x1409 + x1410) + x109*x1411*(-x1406 + x1407 + x1408 - x1409 + x1410) + x109*x1449*(x142*x1450 + x1451*x149 + x1452*x146 + x1453*x152 + x1455 + 131.9677734375*x4 - 54.140625*x5) + x109*x1465*(x1450*x243 + x1451*x245 + x1452*x248 + x1453*x249 + x1455 + (135135/1024)*x4 - 3465/64*x5) + x109*x1581*(x1019*x1457 + x1022*x1459 + x1024*x1458 + x1086 - x1476*x296 + x1582 + x1584*x300 + x305*(-x1585 - x1586)) + x109*x1594*(x1086 + x1457*x295 + x1458*x304 + x1459*x299 + x1500*x300 - x1582 - x1597*x305 + x296*(-x1321 - x1595)) + x109*x1600*(-x0*x256 - 190033.59375*x1236 + x1462*x259 + x1601 + x1607 + x1608 + x203*(x1602 - x1606)) + x109*x1609*(-x0*x192 + (6081075/32)*x1236 + x1462*x202 - x1601 - x1607 + x1608 + x1612*x203) + x109*x1621*(x102*x1560 + x107*x1562 + x1559*x96 + x1622*x65 + x73*(-x1517 - x1583) + x82*(x1623 + x1625)) + x109*x1635*(x1251*x65 + x1520*x73 + x1559*x64 + x1560*x72 + x1562*x81 - x1637*x82) + x109*x1638*(x1537*x220 + x1539*x215 + x171*(-x1631 - x1640) + x183*(-x1641 - x1644)) + x109*x1645*(1299.375*x1232 + 1949.0625*x1236 + x127*(x1650 + x1652) + x133*(-x1496 - x1647) + x1530*x282 + x1535*x278 - x1646) + x109*x1660*(x1537*x182 + x1539*x170 - x1662*x171 + x183*(-x1335 - x1663)) + x109*x1664*((10395/8)*x1232 - 31185/16*x1236 + x126*x1535 + x127*(x1666 + x1667) + x132*x1530 + x133*(-x1502 - x1665) + x1646) + x1090*x1252*(x1087 - 175.95703125*x1126) + x1091*x1304*((212837625/16)*x0*x162 + (70945875/16)*x0*x56 - 212837625/16*x1232 - 70945875/16*x1305) + x1091*x1402*((225225/2048)*x1 + x1104 - 243.6328125*x1317 + (225225/256)*x1318 - 1576575/2048*x1403 - 9.84375*x3 - 45045/256*x4 + 81.2109375*x5) + x1091*x1456*(-x0*x701 + 487.265625*x1079 + (1576575/1024)*x1085 + x1105*x1457 + x1107*x1458 + x1108*x1459 - 225225/128*x1126) + x1091*x1461*(x1102*x1462 - 126689.0625*x1180 + (2027025/8)*x1232 - 2027025/16*x1266 + x1464*x203) + x1091*x1529*(x1124*x1530 + 1949.0625*x1180 - 31185/16*x1232 + x127*(x1533 - x1534) + x133*x1532 + x1535*x168) + x1091*x1536*(x1115*x1537 + x1117*x1539 + x1541*x171 + x183*(-x1267 - x1538)) + x1091*x1558*(x1095*x1559 + x1096*x1560 + x1097*x1562 - x1429*x65 + x1492*x73 + x82*(-x1354 - x1561)) + x1131*x1214*(-29.53125*x1079 + x1128) + x1132*x1265*((155925/8)*x1180 - 155925/4*x1232 + (155925/8)*x1266) + x1132*x1316*(x1135 + (2835/32)*x1317 - 17325/128*x1318 + 6.5625*x3 + (3465/128)*x4 - 945/32*x5) + x1132*x1360*(-2835/16*x1079 + (17325/64)*x1126 + x1136*x1362 + 13.125*x1180 + x1361*x703) + x1132*x1376*(x1138*x1377 - 708.75*x1180 + (2835/4)*x1232 + x401*(-x1378 - x1379)) + x1132*x1427*(x1140*x1428 + x1429*x495 + x1432*x461 + x334*(x1430 - x1431)) - x1148*x1279*(-934.771728515625*x1085 + x1144) + x1154*x1351*((206239658625/128)*x1180 - 206239658625/32*x1232 + (618718975875/64)*x1266 - 206239658625/32*x1305 + (206239658625/128)*x1352) + x1154*x1460*(-130945815/32768*x1*x57 - 3828825/4096*x1 + x1169 + 527.87109375*x1317 - 3299.1943359375*x1318 + (26801775/4096)*x1403 + (14549535/32768)*x2 + 13.53515625*x3 + 659.8388671875*x4 - 175.95703125*x5) + x1154*x1548*(-1055.7421875*x1079 - 26801775/2048*x1085 + x1106*x1552 + 6598.388671875*x1126 + (130945815/16384)*x1142 + x1170*x1550 + x1171*x1551 + 27.0703125*x1180 + x1549*x94) + x1154*x1553*(x1156*x1554 - 37689996.09375*x1180 + 113069988.28125*x1232 - 3618239625/32*x1266 + (1206079875/32)*x1305 + x550*(4704*x0*x162 + 672.0*x0*x56 - 3360*x1232 - 2016*x1305)) + x1154*x1613*(x1176*x1614 + x1178*x1615 + x660*(-70.0*x0 + x1309 + x1616) + x667*(2940*x0*x118 - 1260*x1236 - 1764*x1308 + x1617)) + x1154*x1618*(x1101*x1619 + x1164*x1620 + 475083.984375*x1180 - 950167.96875*x1232 + (30405375/64)*x1266 - x1464*x633 + x640*(x0*x1189 + 360.0*x1180 - x1437)) + x1154*x1629*(x1123*x1630 + x1161*x1632 + x1162*x1633 - 4222.96875*x1180 + 4222.96875*x1232 - x1532*x579 + x1631*x577 + x584*(-1008*x0*x60 - x1435)) + x1154*x1653*(x1150*x1654 + x1151*x1656 + x1152*x1659 + x129*x1657 - x1492*x513 + x511*(18.0*x1236 + x1416) + x515*(x1561 + x1655) + x522*(-756*x1236 - x1658)) + x1154*x1668*(x1116*x1672 + x1158*x1669 + x1159*x1671 - x1541*x619 + x621*(x1538 + x1670) + x627*(-x1673 + x1675)) + x1184*x1262*(20*x0 + x1267 + x1269) + x1190*x1323*x1324 + x1194*x1295*(70*x0 - x1306 - x1309) + x1196*x1227*x56 + x1199*x1210 - 4514284.91559795*x1199 - x1204*(-x1049 + x1201) + x1204*(-x1201 + (3/4)*x56) + x1206*x3 - x1206*x57 + x1208*((15/8)*x0*x56 - 35/8*x1079) + x1208*(1.875*x0*x56 - 4.375*x1079) + x1213*(-x1212 + (315/32)*x3*x56) - x1213*(x1212 - x292*x3) - x1215*x1218*x944 + x1219*x944*(x1216 + x1217) + x1222*(-10395/512*x1126 - x1220 + (315/16)*x24*x56) + x1222*(-20.302734375*x1126 - x1220 + 19.6875*x24*x56) + x1223*x1224 + x1224*x1225 + x1228*x872*(x1050*x1230 - x1229 + x919) + x1233*x872*(x1234 + x1235 - x1237) + x1238*x872*(x1229 + x1230*x878 + x40) + x1239*x872*(-x1234 + x1235 + x1237) + x1243*(-67.67578125*x1211 + x1242) - x1243*(-x1242 + (17325/256)*x5*x56) + x1245*x56*x967 + x1247*x56*x971 + x1255*(-87.978515625*x1085 - x1253 + 131.9677734375*x22*x56) + x1255*(-45045/512*x1085 - x1253 + (135135/1024)*x22*x56) + x1256*x890*(x1257 - x1258 + x1259) + x1260*x890*(-x1257 + x1258 + x1259) + x1263*x56*x931 - x1273*(-384.906005859375*x1241 + x1272) + x1273*(-x1272 + (1576575/4096)*x4*x56) + x1275*x1276 + x1276*x1278 + x1280*x890*(x1281 + x1282 + x1283*x142 + x1284*x332) - x1291*(-370.013809204102*x1142 - x1289 + 747.8173828125*x23*x56) - x1291*(-24249225/65536*x1142 - x1289 + (765765/1024)*x23*x56) + x1292*x890*(x1282 + x1283*x902 + x1284*x503 + (15/8)*x3) + x1293*x1296 + x1293*x1348*x86 + x1297*x699*((4725/4)*x1180 - x1298 + x1299) + x1300*x699*((4725/4)*x0*x56 - x1298 - x1299) + x1301*x699*(x1043*x1271 + x1044*x1303 - x1302 + 9.84375*x24 - x448) + x1310*x699*(x1271*x714 + x1302 + x1303*x716 - 315/32*x24 + x448) + x1311*x1340*(x1335 + x1336 + x1339) + x1312*x1313 + x1313*x1315*x931 + x1314*x1340*(x1339 + x1342) + x1319*x890*(3.75*x0 + x1054*x1271 + x1209 + x715*(-x1320 - x1321)) + x1323*x983*(252*x0 - 3024*x1236 + 7560*x1268 - 7056*x1308 + 2268*x1353) + x1325*x890*(x0*x895 - 15/4*x0 + x1271*x489 - x1328*x715) + x1330*x1331 + x1331*x1333 + x1343*x476*(x0*x472 + x0*x473 - x1344 + x1345) + x1346*x476*((155925/32)*x0*x56 + (155925/16)*x0*x60 - x1344 - x1345) + x1347*x1388*(x1380 - x1381 + x1384 + x1387) + x1349*x1388*(x1384 + x1386 + x1389 + x1390) + x1350*x56*x86 + x1363*x476*(-x1364 + x1365 + x1366*x141 + x1367*x481 + x1368*x483 + 19.6875*x5) + x1369*x319*(x1370 - x1371 - x1372 + x1373 + x1374) + x1375*x319*(-x1370 + x1371 + x1372 - x1373 + x1374) + x1391*x476*(x1365 + x1366*x502 + x1367*x504 + x1368*x247 + (315/16)*x5 - x715) + x1392*x319*(x1031*x1362 + x1032*x1395 + x1033*x1396 - x1393 - x1394 + 67.67578125*x22 + x25) + x1397*x960*(112*x0 + 1680*x1268 - x1433 - x1434 + x1438) + x1398*x1399 + x1399*x1401 + x1400*x960*(1680*x0*x118 + 112*x0 - x1433 - x1434 - x1438) + x1404*x319*(x1362*x423 + x1393 + x1394 + x1395*x421 + x1396*x426 - 17325/256*x22 + x26) + x1412*x699*(-39.375*x1180 + x1361*x704 + x1413 - x1414 + x491*(x1248 + x1417)) + x1418*x699*(-315/8*x1180 + x1361*x727 - x1413 + x1414 + x491*(6*x0 + x1420)) + x1421*x699*(x1367*x697 + x1423*x695 + x482*(x1326 + x1424) + x696*(-x1217 - x1422)) + x1425*x699*(x1218*x696 + x1367*x709 + x1423*x708 - x1426*x482) + x1439*x984*(x1480 + x1481 - x1482 - x1483 - x1486) + x1440*x1441 + x1441*x1443 + x1442*x984*(-x1480 - x1481 + x1482 + x1483 - x1486) + x1444*x747*((2170943775/32)*x1180 + (6512831325/32)*x1266 - x1445 - x1446 + x1447) + x1448*x747*((6512831325/32)*x0*x162 + (2170943775/32)*x0*x56 - x1445 - x1446 - x1447) + x1466*x747*(x1030*x1471 + x1043*x1470 + x1046*x1458 + x1047*x1472 + x1467 - x1468 - x1469 + 384.906005859375*x23 - x863) + x1473*x476*(x1035*x1361 + x1036*x1362 - x1127 - x1474 + x1476*x491 + x424*(-x1477 - x1478)) + x1479*x747*(x1458*x852 - x1467 + x1468 + x1469 + x1470*x420 + x1471*x849 + x1472*x854 - 1576575/4096*x23 + x863) + x1487*x476*(-708.75*x1236 + x1377*x499 + x1488 + x1489 + x401*(x1492 + x1496)) + x1497*x476*(-x1271 + x1328*x491 + x1361*x490 + x1362*x492 + x1474 - x1500*x424) + x1501*x476*((2835/4)*x1236 + x1377*x462 - x1488 + x1489 + x401*(x1502 + x1504)) + x1505*x524*(x1506 + x1507 + x1508 - x1509 - x1510 - x1511) + x1512*x524*(-x1506 - x1507 - x1508 + x1509 + x1510 - x1511) + x1513*x476*(x1428*x507 + x1432*x508 + x334*(x1478 + x1517) + x495*(-x1250 - x1493)) + x1518*x476*(-x1251*x495 + x1428*x63 + x1432*x496 - x1520*x334) + x1521*x524*(747.8173828125*x1 + x141*x1522 + x145*x1524 + x1523*x481 + x1525*x528 + x1526*x533 + x1528 - 494.879150390625*x4) + x1542*x524*((765765/1024)*x1 + x1522*x242 + x1523*x670 + x1524*x671 + x1525*x672 + x1526*x673 + x1528 - 2027025/4096*x4) + x1543*x319*(x1432*x349 + x1452*x351 + x147*(x1498 + x1547) + x1545*x329 + x1546*x347 + x334*(-x1320 - x1424)) + x1555*x319*(x1426*x334 + x1432*x333 + x1452*x337 - x147*x1557 + x1546*x328 + x329*(-x1217 - x1556)) + x1563*x319*(-9745.3125*x1236 + x133*(-x1567 - x1570) + x1530*x369 + x1564 + x1571 + x1572) + x1573*x319*(x0*x1133 + x133*(x1342 + x1567) + x1530*x361 - x1564 - x1571 + x1572) + x1574*x319*(132.890625*x1236 + x1377*x400 + x1459*x405 - x1575 + x1576 + x300*(x1478 + x1578) + x401*(-x1417 - x1493)) + x1579*x319*(-8505/64*x1236 + x1377*x444 + x1459*x445 + x1575 + x1576 + x300*(15*x0 + x1580) + x401*(-x1415 - x1420)) + x1587*x319*(x1560*x437 + x1588*x436 + x385*(x1285 + x1591) + x73*(-x1492 - x1593)) + x1598*x319*(x1560*x390 + x1588*x384 + x385*(-x1288 - x1599) + x73*(x1504 + x1593)) + x1626*x747*(x1390*x529 + x1525*x758 - x1545*x754 + x1562*x757 + x1627*x346 + x1628*x755 + x756*(x0*x1113 + x1320) + x82*(-x1477 - x1547)) + x1634*x747*(x1525*x834 + x1557*x82 + x1562*x832 + x1627*x830 + x1628*x831 + x529*(-x1381 - x1596) + x754*(x1544 + x1556) + x756*(-x1424 - x1595)) + x1676*x747*(3325587.890625*x1232 + x1619*x771 + x1677 - x1678 + x1686 - x1687 + x1688 + x633*(x1616 + x1680 - x1681 + x1682 + x1685)) + x1689*x747*((212837625/64)*x1232 + x1619*x766 - x1677 + x1678 - x1686 + x1687 + x1688 + x633*(x1679 - x1682 + x1685 + x1690 + x1691)) + x1692*x524*(x1021*x1550 + x1035*x1549 + x1038*x1552 + x1041*x1551 + x1476*x558 - x1584*x561 - x1693 + x1694 + x563*(x1585 + x1695) + x567*(-x1617 - x1696)) + x1697*x524*(x1549*x557 + x1550*x560 + x1551*x566 + x1552*x562 + x1597*x563 + x1693 + x1694 + x558*(x1475 + x1595) + x561*(x1477 - x1478) + x567*(-x1658 - x1696)) + x1698*x747*(x127*(-x1578 - x1583) + x1535*x750 + x1552*x751 + x1699 + x1700 - x1701 + x1702*x749 + x563*(x1623 + x1703 + x1704) + x735*(x1416 + x1493 + x1642)) + x1705*x747*(x127*(-x1565 - x1580) + x1535*x736 + x1552*x738 - x1699 + x1700 + x1701 + x1702*x734 + x563*(x1623 + x1681 - x1704) + x735*(x1415 + x1419 + x1493)) + x1706*x524*(113069988.28125*x1268 + x1554*x549 + x1707 - x1708 - x1713 + x1714 + x550*(x0*x543 + 1680.0*x1236 - x1709 - x1710 + x1711 + x1712)) + x1715*x524*(-3618239625/32*x1268 + x1554*x571 - x1707 + x1708 + x1713 + x1714 + x550*(560*x0*x60 - 1680*x1236 + x1709 + x1710 - x1711 + x1712)) + x1716*x747*(x1672*x844 + x1717*x842 + x619*(x1463 + x1611 + x1719) + x824*(180.0*x0*x57 - x1355 - x1540 - x1718)) + x1720*x747*(x1672*x827 + x1717*x823 + x619*(-180*x1180 - x1383 + x1604 - x1610 - x1719) + x824*(x1357 + x1655 + x1718)) + x1721*x747*(x1539*x859 + x1656*x860 + x171*(x1492 + x1569 + x1592) + x1722*x858 + x515*(-x1650 - x1724) + x789*(-x0*x275 - x1591)) + x1725*x747*(-21114.84375*x1232 + x1462*x784 + x1632*x785 - x1726 + x1727 + x1728 - x1729 + x203*(-150.0*x1236 + x1565 + x1566 + x1570) + x577*(-x1631 - x1731)) + x1732*x747*(-675675/32*x1232 + x1462*x777 + x1632*x780 + x1726 + x1727 - x1728 + x1729 + x203*(-x1567 - x1641) + x577*(x1661 + x1731)) + x1733*x747*(x1539*x790 + x1656*x792 + x171*(-x1593 - x1665) + x1722*x788 + x515*(x1666 + x1724) + x789*(x1590 + x1599)) + x1734*x524*(-x1622*x511 + x1654*x95 + x1656*x514 + x1657*x512 + x1659*x521 + x513*(x0*x815 + x1515 + x1583) + x515*(-x0*x1039 - x1625) + x522*(x1306 + x1381 + x1703)) + x1735*x524*(x1637*x515 + x1654*x603 + x1656*x605 + x1657*x604 + x1659*x608 + x511*(-x1250 - x1494) + x513*(-x1514 - x1519) + x522*(-x1381 - x1681 - x1690)) + x1736*x524*(x1614*x680 + x1615*x682 + x660*(840.0*x0*x60 - x1684 - x1737) + x667*(x0*x681 + 420.0*x1180 - x1738 + x1739 + x1740)) + x1741*x524*(x1614*x659 + x1615*x666 + x660*(-x1387 + x1683 - x1737) + x667*(x1561 - x1696 + x1738 - x1739 + x1740)) + x1742*x524*(-2815.3125*x1232 + x1630*x578 + x1632*x576 + x1633*x583 + x1743 - x1744 + x577*(-168.0*x1236 - x1649 - x1652) + x579*(x0*x840 + x1494 + x1647) + x584*(-56.0*x0 - 168.0*x1180 + x1385 + x1485)) + x1745*x524*(x0*x630 - 593854.98046875*x1268 + x1619*x644 + x1620*x646 - x1746 + x1748 + x1749 + x633*(-240.0*x1232 + x1606) + x640*(840.0*x0*x57 - x1568 - x1675 - x1747)) + x1750*x524*(-45045/16*x1232 + x1630*x612 + x1632*x611 + x1633*x614 - x1743 + x1744 + x577*(-x1636 - x1648 - x1667) + x579*(x1496 + x1665) + x584*(56*x0 + x1385 - x1485 + x1723)) + x1751*x524*((30405375/128)*x1232 + (152026875/256)*x1268 - x1612*x633 + x1619*x632 + x1620*x639 + x1746 - x1748 + x1749 + x640*(-x1334 + x1484 - x1673 + x1674 - x1747)) + x1752*x524*(x1669*x620 + x1671*x626 + x1672*x618 + x619*(x1631 + x1639 + x1670) + x621*(-100.0*x1232 + x1568 + x1644) + x627*(280.0*x0*x56 - x1385 - x1477 - x1753)) + x1754*x524*(x1662*x619 + x1669*x685 + x1671*x686 + x1672*x169 + x621*(x1569 + x1663) + x627*(-280*x1180 - x1389 - x1498 - x1753)))
    x1756 = (3/4)*x3 - 0.5
    x1757 = e*x44
    x1758 = x1059*x1757
    x1759 = x1202*x42
    x1760 = -x1281 + (105/64)*x5 + 0.375
    x1761 = e*x888
    x1762 = x10*x43*x887
    x1763 = 1097091850.7304*x1762
    x1764 = x1207*x1761
    x1765 = x454*x886
    x1766 = x1765*x31
    x1767 = 2284241680102.88*x1766
    x1768 = x1207*x929
    x1769 = x1207*x926
    x1770 = x1074*x1760
    x1771 = e*x1765
    x1772 = x455*((9/2)*x10 + 2)
    x1773 = x1202*x1772
    x1774 = 10*e + x32
    x1775 = x1364 + (1155/256)*x4 - 7.3828125*x5 - 0.3125
    x1776 = x1775*x465
    x1777 = x453*x465
    x1778 = 515941602390199.0*x1203*x1777
    x1779 = x450*x452
    x1780 = x1066*x1210
    x1781 = x465*(5*e + 5*x31)
    x1782 = x1221*x1781
    x1783 = e*x36*x458
    x1784 = 2.04496089337997e+19*x1783
    x1785 = 1.34203708705209e+17*x1761
    x1786 = x1081*x1264
    x1787 = x1207*x1784*x466
    x1788 = x1197*x1226
    x1789 = x1073*x1231
    x1790 = x37*((45/2)*x10 + (75/8)*x11 + 3)
    x1791 = x1202*x1790
    x1792 = 128695539609905.0*x1762
    x1793 = x1051*x1228
    x1794 = 21*e + (105/2)*x31 + (105/8)*x33
    x1795 = (225225/16384)*x1 - x1454 - 29.326171875*x4 + 20.302734375*x5 + 0.2734375
    x1796 = x1795*x84
    x1797 = x1238*x879
    x1798 = x28*x34
    x1799 = x35*x84
    x1800 = 6.59608018014363e+22*x1203*x1799
    x1801 = 3.1314198697882e+17*x1771
    x1802 = x1074*x1801
    x1803 = 241304136768571.0*x1762
    x1804 = x0*x1223
    x1805 = x0*x1225
    x1806 = x1233*x870
    x1807 = x84*((21/2)*e + 35*x31 + (315/32)*x33)
    x1808 = x1254*x1807
    x1809 = x1074*x1788
    x1810 = e*x54*x744
    x1811 = x1089*x1810
    x1812 = x1084*x1270
    x1813 = x1074*x1789
    x1814 = x1239*x874
    x1815 = x1244*x968
    x1816 = 4.40355919188966e+17*x1761
    x1817 = x1246*x972
    x1818 = x1810*x85
    x1819 = 5.92433421263338e+25*x1207*x1818
    x1820 = x1280*x900
    x1821 = 6.71018543526043e+16*x1761
    x1822 = x1256*x885
    x1823 = x1260*x892
    x1824 = 1.54124571716138e+18*x1766
    x1825 = x745*(63*x10 + (175/2)*x11 + (245/16)*x12 + 4)
    x1826 = x1202*x1825
    x1827 = x1292*x903
    x1828 = 2.34856490234115e+17*x1766
    x1829 = x1129*x1316
    x1830 = x1774*x465
    x1831 = x1830*x459
    x1832 = 36*e + 189*x31 + (315/2)*x33 + (315/16)*x741
    x1833 = -116.846466064453*x1 + x1527 + (2909907/65536)*x2 + 109.97314453125*x4 - 43.9892578125*x5 - 0.24609375
    x1834 = x1134*x1265
    x1835 = x17*x743
    x1836 = 1.85861145123112e+29*x1203*x1835
    x1837 = x742*x865
    x1838 = 18*e + 126*x31 + (945/8)*x33 + (63/4)*x741
    x1839 = x1290*x1838
    x1840 = 1.48639930075317e+24*x1783
    x1841 = x1130*x1840
    x1842 = 2.41423761427018e+26*x0*x1783
    x1843 = e*x15/x16**(21/2)
    x1844 = x1147*x1843
    x1845 = 7.77486819098842e+19*x1772
    x1846 = x1045*x1301
    x1847 = x1310*x718
    x1848 = x1055*x1319
    x1849 = 1.15045699501558e+32*x1207*x1843
    x1850 = x1011*x1772
    x1851 = x1325*x896
    x1852 = e*x1777
    x1853 = 6.99738137188958e+20*x1852
    x1854 = x1853*x452
    x1855 = x0*x452
    x1856 = x1275*x1855
    x1857 = 2.06641418638614e+22*x1852
    x1858 = x1278*x1855
    x1859 = x1297*x722
    x1860 = x1300*x730
    x1861 = x1794*x84
    x1862 = x1861*x55
    x1863 = x1088*x1402
    x1864 = x1121*x1304
    x1865 = x1141*x1427
    x1866 = x1139*x1376
    x1867 = x1421*x698
    x1868 = x1137*x1360
    x1869 = 4.84788992387492e+34*x0
    x1870 = x1781*x459
    x1871 = x1391*x505
    x1872 = x1363*x485
    x1873 = x1425*x710
    x1874 = 6.12256790376034e+30*x1811
    x1875 = x1840*x466
    x1876 = 2.34851089519e+26*x1790
    x1877 = x1034*x1392
    x1878 = x1412*x705
    x1879 = x1404*x428
    x1880 = x1418*x728
    x1881 = x18*x1832
    x1882 = x1311*x938
    x1883 = x1842*x466
    x1884 = 1.23293986687242e+36*x18
    x1885 = x1832*x1884
    x1886 = x1168*x1351
    x1887 = x1343*x475
    x1888 = x1346*x478
    x1889 = e*x1799
    x1890 = 3.053064163747e+27*x1889
    x1891 = x1890*x34
    x1892 = x1003*x1790
    x1893 = x1880*x452
    x1894 = x1190*x1322
    x1895 = 1.49781152273359e+43*x0
    x1896 = 2.34258574705759e+37*x1844
    x1897 = x0*x34
    x1898 = x1330*x1897
    x1899 = 3.22324863881212e+30*x1889
    x1900 = x1333*x1897
    x1901 = x1369*x318
    x1902 = x1375*x413
    x1903 = x1513*x509
    x1904 = x1145*x1460
    x1905 = x1518*x497
    x1906 = x1118*x1536
    x1907 = x1103*x1461
    x1908 = x1037*x1473
    x1909 = x1901*x34
    x1910 = x1807*x55
    x1911 = x1449*x154
    x1912 = x1487*x500
    x1913 = x1109*x1456
    x1914 = x1497*x493
    x1915 = x1818*x1869
    x1916 = x1349*x996
    x1917 = x1465*x250
    x1918 = x1501*x464
    x1919 = x1405*x266
    x1920 = x1411*x236
    x1921 = 6.12256790376034e+30*x1818
    x1922 = x1098*x1558
    x1923 = x1543*x352
    x1924 = 7.09401534449031e+32*x1825
    x1925 = x1048*x1466
    x1926 = x1479*x856
    x1927 = x1125*x1529
    x1928 = x1555*x338
    x1929 = e*x1835
    x1930 = 1.20598260856335e+34*x1929
    x1931 = x1930*x742
    x1932 = x1825*x959
    x1933 = x1587*x438
    x1934 = x1598*x391
    x1935 = x0*x742
    x1936 = x1398*x1935
    x1937 = 8.11669282247914e+38*x1929
    x1938 = x1401*x1935
    x1939 = x1444*x804
    x1940 = x1448*x810
    x1941 = x1179*x1613
    x1942 = x1157*x1553
    x1943 = x1838*x1884
    x1944 = x1521*x536
    x1945 = x1574*x407
    x1946 = x1563*x370
    x1947 = x1579*x446
    x1948 = x1542*x674
    x1949 = x1573*x363
    x1950 = 2.34258574705759e+37*x1843
    x1951 = x14*x1950
    x1952 = x1172*x1548
    x1953 = x1025*x1581
    x1954 = x18*x1838*x983
    x1955 = x1594*x307
    x1956 = x108*x1621
    x1957 = x1153*x1653
    x1958 = x1638*x221
    x1959 = x14*x1440
    x1960 = x1843*x1895
    x1961 = x14*x1443
    x1962 = x1160*x1668
    x1963 = x1505*x600
    x1964 = x1512*x693
    x1965 = x1626*x759
    x1966 = x1660*x184
    x1967 = x1635*x83
    x1968 = x1634*x835
    x1969 = x1165*x1618
    x1970 = x1600*x260
    x1971 = x1965*x742
    x1972 = x1163*x1629
    x1973 = x1609*x205
    x1974 = x1645*x283
    x1975 = x134*x1664
    x1976 = x1716*x845
    x1977 = x1720*x828
    x1978 = x1042*x1692
    x1979 = x1698*x752
    x1980 = x1721*x861
    x1981 = x1697*x569
    x1982 = x1705*x740
    x1983 = x1733*x793
    x1984 = x1676*x772
    x1985 = x1689*x768
    x1986 = x1736*x683
    x1987 = x1734*x523
    x1988 = x1741*x668
    x1989 = x14*x1987
    x1990 = x1735*x609
    x1991 = x1725*x786
    x1992 = x1706*x552
    x1993 = x1732*x782
    x1994 = x1715*x572
    x1995 = x1752*x628
    x1996 = x1742*x585
    x1997 = x1754*x687
    x1998 = x1750*x615
    x1999 = x1745*x647
    x2000 = x1751*x642
    x2001 = x0*x1312*x1781 + x0*x1350*x1807 + x1052*x1228 + x1130*x1183*x1261*x1842 + x1130*x1775*x1784 + x1146*x1460*x1881 - 66643259867.0392*x1180*x1205*x1758 + x1184*x1315*x1781 + x1185*x1261*x1830 + x1191*x1322*x1881 + x1193*x1294*x1811*x1869 + x1194*x1348*x1807 + x1195*x1294*x1861 - x1202*x1763*x920 - x1202*x921 + x1214*x1774*x1776 + x1215*x947 + x1225*x943 + x1238*x880 + x1239*x875 + x1252*x1794*x1796 + x1275*x1850 + x1278*x1850 - x1279*x17*x1832*x1833 + x1315*x1883 + x1330*x1892 + x1333*x1892 + x1348*x1915 + x1398*x1932 - x14*x1849*x21 - x14*x1849*x7 + x1401*x1932 + x1440*x1954 + x1443*x1954 - 9028569.83119589*x1756*x1758 + x1758*x1780 + x1759*x1763 + x1759*x46 + 1305280960058.79*x1760*x1761 + 8.80711838377931e+17*x1761*x1788 + x1764*x927 + x1764*x930 + x1767*x1768 + x1767*x1769 + 3045655573470.51*x1770*x1771 + 2.05499428954851e+18*x1771*x1809 + x1773*x451 - x1773*x923 + x1778*x1779 - x1778*x452*x922 + x1782*x906 + x1782*x909 + x1785*x1786 + x1785*x1789 + x1785*x1812 + x1786*x1802 + x1787*x906 + x1787*x909 + x1791*x29 - x1791*x912 + x1792*x1793 + x1792*x1797 + x1792*x1806 + x1792*x1814 + 5.92433421263338e+25*x1795*x1811 + x1798*x1800 - x1800*x34*x911 + x1801*x1813 + x1802*x1812 + x1803*x1804 + x1803*x1805 + x1806*x871 + x1808*x914 + x1808*x917 + x1815*x1816 + x1815*x1824 + x1816*x1817 + x1817*x1824 + x1819*x914 + x1819*x917 + x1820*x1821 + x1820*x1828 + x1821*x1822 + x1821*x1823 + x1821*x1827 + x1821*x1848 + x1821*x1851 + x1822*x1828 + x1823*x1828 + x1826*x866 - x1826*x925 + x1827*x1828 + x1828*x1848 + x1828*x1851 + x1829*x1831 + x1829*x1841 + x1831*x1834 + x1831*x1865 + x1831*x1866 + x1831*x1868 - 1.15045699501558e+32*x1833*x1844 + x1834*x1841 + x1836*x1837 - x1836*x742*x924 - x1839*x21 - x1839*x7 + x1841*x1865 + x1841*x1866 + x1841*x1868 + x1844*x1894*x1895 + x1845*x1846 + x1845*x1847 + x1845*x1859 + x1845*x1860 + x1845*x1867 + x1845*x1873 + x1845*x1878 + x1845*x1880 + x1846*x1854 + x1847*x1854 + x1853*x1893 + x1854*x1859 + x1854*x1860 + x1854*x1867 + x1854*x1873 + x1854*x1878 + x1856*x1857 + x1857*x1858 + x1862*x1863 + x1862*x1864 + x1862*x1906 + x1862*x1907 + x1862*x1913 + x1862*x1922 + x1862*x1927 + x1863*x1874 + x1864*x1874 + x1870*x1871 + x1870*x1872 + x1870*x1887 + x1870*x1888 + x1870*x1903 + x1870*x1905 + x1870*x1908 + x1870*x1912 + x1870*x1914 + x1870*x1918 + x1871*x1875 + x1872*x1875 + x1874*x1906 + x1874*x1907 + x1874*x1913 + x1874*x1922 + x1874*x1927 + x1875*x1887 + x1875*x1888 + x1875*x1903 + x1875*x1905 + x1875*x1908 + x1875*x1912 + x1875*x1914 + x1875*x1918 + x1876*x1877 + x1876*x1879 + x1876*x1901 + x1876*x1902 + x1876*x1923 + x1876*x1928 + x1876*x1933 + x1876*x1934 + x1876*x1945 + x1876*x1946 + x1876*x1947 + x1876*x1949 + x1877*x1891 + x1879*x1891 + x1882*x1883 + x1885*x1886 + x1885*x1941 + x1885*x1942 + x1885*x1952 + x1885*x1957 + x1885*x1962 + x1885*x1969 + x1885*x1972 + x1886*x1896 + x1890*x1909 + x1891*x1902 + x1891*x1923 + x1891*x1928 + x1891*x1933 + x1891*x1934 + x1891*x1945 + x1891*x1946 + x1891*x1947 + x1891*x1949 + x1896*x1904 + x1896*x1941 + x1896*x1942 + x1896*x1952 + x1896*x1957 + x1896*x1962 + x1896*x1969 + x1896*x1972 + x1898*x1899 + x1899*x1900 + x1910*x1911 + x1910*x1917 + x1910*x1919 + x1910*x1920 + x1910*x1953 + x1910*x1955 + x1910*x1956 + x1910*x1958 + x1910*x1966 + x1910*x1967 + x1910*x1970 + x1910*x1973 + x1910*x1974 + x1910*x1975 + x1911*x1921 + x1915*x1916 + x1917*x1921 + x1919*x1921 + x1920*x1921 + x1921*x1953 + x1921*x1955 + x1921*x1956 + x1921*x1958 + x1921*x1966 + x1921*x1967 + x1921*x1970 + x1921*x1973 + x1921*x1974 + x1921*x1975 + x1924*x1925 + x1924*x1926 + x1924*x1939 + x1924*x1940 + x1924*x1965 + x1924*x1968 + x1924*x1976 + x1924*x1977 + x1924*x1979 + x1924*x1980 + x1924*x1982 + x1924*x1983 + x1924*x1984 + x1924*x1985 + x1924*x1991 + x1924*x1993 + x1925*x1931 + x1926*x1931 + x1930*x1971 + x1931*x1939 + x1931*x1940 + x1931*x1968 + x1931*x1976 + x1931*x1977 + x1931*x1979 + x1931*x1980 + x1931*x1982 + x1931*x1983 + x1931*x1984 + x1931*x1985 + x1931*x1991 + x1931*x1993 + x1936*x1937 + x1937*x1938 + x1943*x1944 + x1943*x1948 + x1943*x1963 + x1943*x1964 + x1943*x1978 + x1943*x1981 + x1943*x1986 + x1943*x1987 + x1943*x1988 + x1943*x1990 + x1943*x1992 + x1943*x1994 + x1943*x1995 + x1943*x1996 + x1943*x1997 + x1943*x1998 + x1943*x1999 + x1943*x2000 + x1944*x1951 + x1948*x1951 + x1950*x1989 + x1951*x1963 + x1951*x1964 + x1951*x1978 + x1951*x1981 + x1951*x1986 + x1951*x1988 + x1951*x1990 + x1951*x1992 + x1951*x1994 + x1951*x1995 + x1951*x1996 + x1951*x1997 + x1951*x1998 + x1951*x1999 + x1951*x2000 + x1959*x1960 + x1960*x1961
    x2002 = x1757*x886
    x2003 = x453*x887
    x2004 = x10*x2003
    x2005 = 1631601200073.48*x2004
    x2006 = x454*x458
    x2007 = 1.30133875033271e+19*x35
    x2008 = x1207*x2007
    x2009 = 102956431687924.0*x2002
    x2010 = 193043309414857.0*x2002
    x2011 = x36*x54
    x2012 = 2.23672847842014e+17*x2003
    x2013 = x1074*x2012
    x2014 = 3.55460052758003e+25*x743
    x2015 = x1207*x2014
    x2016 = 1.10088979797241e+18*x2004
    x2017 = 1.67754635881511e+17*x2004
    x2018 = x15*x744
    x2019 = 9.45890464115651e+23*x35
    x2020 = x1131*x2019
    x2021 = 1.53633302726284e+26*x35
    x2022 = a**(-12.0)
    x2023 = 4.66492091459305e+20*x2006
    x2024 = x2023*x452
    x2025 = 1.37760945759076e+22*x2006
    x2026 = 2.90873395432495e+34*x743
    x2027 = 3.6735407422562e+30*x743
    x2028 = x1090*x2027
    x2029 = x2019*x467
    x2030 = 1.878808716152e+27*x2011
    x2031 = x2030*x34
    x2032 = 8.67154039477343e+42*x0*x2022
    x2033 = 1.35623385355966e+37*x2022
    x2034 = x1148*x2033
    x2035 = 1.98353762388438e+30*x2011
    x2036 = x2027*x86
    x2037 = 7.09401534449031e+33*x2018
    x2038 = x2037*x742
    x2039 = 4.77452518969361e+38*x2018
    x2040 = x17*x2033
    x2041 = x14*x2040
    x2042 = x17*x2032

    # --- Equazioni finali ---
    da_dt = 0
    de_dt = -x1057*x1064
    di_dt = x1057*x1065*x56 - x1065*(x1005*(-4.58335512715287e-10*x813 - 4.44115753484987e-10*x814) + x1007*(4.58335512715287e-10*x838 - 4.44115753484987e-10*x839) + x1013*(-2.48936861679002e-9*x373 + 8.57297485442953e-8*x374) + x1015*(2.48936861679002e-9*x431 + 8.57297485442953e-8*x432) + x1056 + 14809613303.7865*x1066*x1070*(-4.47625475358074e-5*x1067 + 2.93300168734876e-8*x1068) + x1073*x1076*(5.08185042929699e-7*x1071 + 3.30573049257519e-7*x1072) + x1076*x1081*(1.58223867964636e-6*x1077 - 5.69625677298712e-6*x1078) + x1076*x1084*(3.18221440738855e-6*x1067 - 3.02797674808195e-6*x1068) + x1088*x1091*(7.52411610415292e-7*x1077 + 3.01619655232262e-9*x1078) + x1090*x1195*(1.16417124313564e-10*x1173 - 5.41471323834642e-11*x1174) + x1091*x1098*(2.90058893335495e-8*x1092 - 5.73345057314305e-8*x1093) + x1091*x1103*(2.77531744631055e-10*x1099 - 3.54875150885113e-10*x1100) + x1091*x1109*(-4.94610544005789e-7*x1067 + 3.1750217626037e-7*x1068) + x1091*x1118*(2.64111976078032e-9*x1110 - 1.12774470225781e-9*x1111) + x1091*x1121*(2.54173096815002e-11*x1119 + 2.15419864153245e-11*x1120) + x1091*x1125*(-1.77941728039189e-8*x1071 - 2.74795880863205e-9*x1072) + x110*(2.90058893335495e-8*x91 - 5.73345057314305e-8*x92) + x1129*x1132*(-2.04290344291838e-6*x1077 + 1.21250197095655e-6*x1078) + x1131*x1185*(-4.17565784680387e-8*x1110 + 5.95135746257363e-9*x1111) + x1132*x1134*(6.56658695875701e-9*x1099 + 1.00840170597363e-8*x1100) + x1132*x1137*(1.09232625193902e-6*x1067 - 5.39857315699859e-7*x1068) + x1132*x1139*(-5.17427833875168e-9*x1071 - 6.13537328811109e-8*x1072) + x1132*x1141*(-2.13284751178092e-7*x1092 - 2.06481528941037e-7*x1093) + x1146*x1149*(-5.89032237975176e-7*x1077 + 5.17923460214652e-7*x1078) + x1149*x1191*(-1.01700834045076e-14*x1186 - 7.95683297827491e-13*x1187) + x1153*x1154*(1.16464104484496e-8*x1092 + 8.58785839429875e-9*x1093) + x1154*x1157*(3.26853558861384e-12*x1119 + 2.60087474500956e-12*x1120) + x1154*x1160*(-9.52930246285144e-11*x1110 + 2.15409065663148e-10*x1111) + x1154*x1163*(8.42304535941058e-9*x1071 + 3.71289400309528e-9*x1072) + x1154*x1165*(4.8186883582726e-12*x1099 - 8.72317126505236e-11*x1100) + x1154*x1168*(-3.94510514185877e-14*x1166 - 7.1310655482342e-14*x1167) + x1154*x1172*(-4.25884464445994e-8*x1067 - 3.18787653039088e-8*x1068) + x1154*x1179*(-5.4630086432259e-12*x1173 - 2.35193426214309e-11*x1174) - x1180*x1181*(1.54435046418598e-9*x1077 - 9.43103158893466e-8*x1078) + x1197*x1198*(-2.4119506226557e-6*x1092 - 2.39981606121136e-7*x1093) + x135*(1.77941728039189e-8*x115 - 2.74795880863205e-9*x116) + x155*(7.52411610415292e-7*x138 + 3.01619655232262e-9*x139) + x185*(-2.64111976078032e-9*x160 - 1.12774470225781e-9*x161) + x206*(-2.77531744631055e-10*x190 - 3.54875150885113e-10*x191) + x222*(2.64111976078032e-9*x209 - 1.12774470225781e-9*x210) + x237*(2.54173096815002e-11*x227 + 2.15419864153245e-11*x228) + x251*(-7.52411610415292e-7*x240 + 3.01619655232262e-9*x241) + x261*(2.77531744631055e-10*x254 - 3.54875150885113e-10*x255) + x267*(-2.54173096815002e-11*x264 + 2.15419864153245e-11*x265) + x284*(-1.77941728039189e-8*x270 - 2.74795880863205e-9*x271) + x291*x308 + x320*(-2.35990121676921e-10*x313 - 2.08218849223153e-10*x314) + x339*(-4.79544939890818e-7*x322 - 1.27901171249883e-7*x323) + x353*(4.79544939890818e-7*x342 - 1.27901171249883e-7*x343) + x364*(-3.68943137539914e-10*x355 + 1.8912955040722e-9*x356) + x371*(3.68943137539914e-10*x366 + 1.8912955040722e-9*x367) + x392*(-6.43141566698142e-9*x373 - 7.16636370982428e-9*x374) + x408*(-2.52408775172506e-8*x394 + 9.97374864155149e-8*x395) + x414*(2.35990121676921e-10*x411 - 2.08218849223153e-10*x412) + x418*x429 + x439*(6.43141566698142e-9*x431 - 7.16636370982428e-9*x432) + x447*(2.52408775172506e-8*x441 + 9.97374864155149e-8*x442) + x469*(5.17427833875168e-9*x115 - 6.13537328811109e-8*x116) + x477*(-6.56658695875701e-9*x190 + 1.00840170597363e-8*x191) + x479*(6.56658695875701e-9*x254 + 1.00840170597363e-8*x255) + x486*(-2.04290344291838e-6*x138 + 1.21250197095655e-6*x139) - x487*x494 + x498*(2.13284751178092e-7*x52 - 2.06481528941037e-7*x53) + x501*(-5.17427833875168e-9*x270 - 6.13537328811109e-8*x271) + x506*(2.04290344291838e-6*x240 + 1.21250197095655e-6*x241) + x510*(-2.13284751178092e-7*x91 - 2.06481528941037e-7*x92) + x525*(1.16464104484496e-8*x91 + 8.58785839429875e-9*x92) + x537*(-5.89032237975176e-7*x138 + 5.17923460214652e-7*x139) + x553*(3.26853558861384e-12*x227 + 2.60087474500956e-12*x228) + x554*x570 + x573*(-3.26853558861384e-12*x264 + 2.60087474500956e-12*x265) + x586*(8.42304535941058e-9*x270 + 3.71289400309528e-9*x271) + x601*(3.94510514185877e-14*x590 - 7.1310655482342e-14*x591) + x610*(-1.16464104484496e-8*x52 + 8.58785839429875e-9*x53) + x616*(-8.42304535941058e-9*x115 + 3.71289400309528e-9*x116) + x629*(-9.52930246285144e-11*x209 + 2.15409065663148e-10*x210) + x643*(-4.8186883582726e-12*x190 - 8.72317126505236e-11*x191) + x648*(4.8186883582726e-12*x254 - 8.72317126505236e-11*x255) + x669*(5.4630086432259e-12*x650 - 2.35193426214309e-11*x651) + x675*(5.89032237975176e-7*x240 + 5.17923460214652e-7*x241) + x684*(-5.4630086432259e-12*x677 - 2.35193426214309e-11*x678) + x688*(9.52930246285144e-11*x160 + 2.15409065663148e-10*x161) + x694*(-3.94510514185877e-14*x691 - 7.1310655482342e-14*x692) + x700*(3.39127355635322e-7*x342 + 1.4233008183919e-6*x343) + x706*(-4.55344180540829e-8*x394 + 8.62222116034369e-7*x395) + x711*(-3.39127355635322e-7*x322 + 1.4233008183919e-6*x323) - x712*x719 + x723*(3.83737472096761e-8*x355 - 3.38483603498113e-8*x356) + x729*(4.55344180540829e-8*x441 + 8.62222116034369e-7*x442) + x731*(-3.83737472096761e-8*x366 - 3.38483603498113e-8*x367) + x748*(-4.50756363508801e-8*x441 + 4.98754524852772e-8*x442) + x753*(4.50756363508801e-8*x394 + 4.98754524852772e-8*x395) + x760*(-1.92561339895469e-7*x342 + 2.6813901338718e-7*x343) + x769*(-5.22427365696583e-11*x313 - 1.56549280726303e-12*x314) + x773*(5.22427365696583e-11*x411 - 1.56549280726303e-12*x412) + x783*(-7.9605453045416e-10*x355 - 1.80272100029025e-9*x356) + x787*(7.9605453045416e-10*x366 - 1.80272100029025e-9*x367) + x794*(4.85694581625249e-9*x373 - 6.44693966510455e-9*x374) + x805*(-6.56499712934558e-13*x799 + 1.72206392963152e-12*x800) + x811*(6.56499712934558e-13*x808 + 1.72206392963152e-12*x809) + x829*(2.38617027558348e-10*x813 - 1.67838588422666e-10*x814) + x836*(1.92561339895469e-7*x322 + 2.6813901338718e-7*x323) + x846*(-2.38617027558348e-10*x838 - 1.67838588422666e-10*x839) + x847*x857 + x862*(-4.85694581625249e-9*x431 - 6.44693966510455e-9*x432) + x873*(5.13845208024278e-6*x441 - 7.36249502628667e-7*x442) + x876*(-5.13845208024278e-6*x394 - 7.36249502628667e-7*x395) + x877*x881 + x88*(-2.90058893335495e-8*x52 - 5.73345057314305e-8*x53) + x891*(-5.08185042929699e-7*x115 + 3.30573049257519e-7*x116) + x893*(5.08185042929699e-7*x270 + 3.30573049257519e-7*x271) - x894*x897 + x901*(1.58223867964636e-6*x138 - 5.69625677298712e-6*x139) + x904*(-1.58223867964636e-6*x240 - 5.69625677298712e-6*x241) + x937*(-4.17565784680387e-8*x209 + 5.95135746257363e-9*x210) + x940*(4.17565784680387e-8*x160 + 5.95135746257363e-9*x161) + x945*(3.34142161339775e-6*x342 + 9.68572553073115e-6*x343) + x948*(-3.34142161339775e-6*x322 + 9.68572553073115e-6*x323) + x961*(5.77609220451627e-12*x950 - 3.42481177345579e-12*x951) + x966*(-5.77609220451627e-12*x963 - 3.42481177345579e-12*x964) + x970*(2.4119506226557e-6*x52 - 2.39981606121136e-7*x53) + x973*(-2.4119506226557e-6*x91 - 2.39981606121136e-7*x92) + x985*(1.01700834045076e-14*x975 - 7.95683297827491e-13*x976) + x990*(-1.01700834045076e-14*x987 - 7.95683297827491e-13*x988) + x995*(-1.16417124313564e-10*x650 - 5.41471323834642e-11*x651) + x998*(1.16417124313564e-10*x677 - 5.41471323834642e-11*x678))
    dw_dt = x1064*x2001 - x1755*x56
    dOmega_dt = x1755
    dM_dt = 70.0199969086592*x1060 - x1063*x16*x2001 - 0.0285632688988689*x1061*(877673480.584319*e*x1202*x44*x886*x920 + 66643259867.0392*x0*x1069*x1205*x43*x56 - x0*x1263*x2021 - x0*x1296*x2026 + 9028569.83119589*x1069*x1756*x43 - x1069*x1780*x43 - x1089*x1796*x2014 - x1130*x1776*x2007 + 6.66054049745865e+31*x1147*x17*x1833*x2022 - x1148*x1894*x2032 + 1.09330085366536e+29*x1202*x15*x742*x744*x924 - 343961068260133.0*x1202*x1779*x2006 - 4.05912626470377e+22*x1202*x1798*x2011 - 1.09330085366536e+29*x1202*x1837*x2018 + 4.05912626470377e+22*x1202*x34*x36*x54*x911 + 343961068260133.0*x1202*x452*x454*x458*x922 + 6.66054049745865e+31*x1207*x14*x17*x2022*x21 + 6.66054049745865e+31*x1207*x14*x17*x2022*x7 - x1314*x2021*x936 - x1347*x2026*x994 - x1501*x2019*x468 - x1635*x2027*x87 - 877673480.584319*x1759*x2002 - x1768*x2005 - x1769*x2005 - 2175468266764.65*x1770*x2003 - x1786*x2013 - x1793*x2009 - x1797*x2009 - x1804*x2010 - x1805*x2010 - x1806*x2009 - 1.46785306396322e+18*x1809*x2003 - x1812*x2013 - x1813*x2012 - x1814*x2009 - x1815*x2016 - x1817*x2016 - x1820*x2017 - x1822*x2017 - x1823*x2017 - x1827*x2017 - x1829*x2020 - x1834*x2020 - x1846*x2024 - x1847*x2024 - x1848*x2017 - x1851*x2017 - x1856*x2025 - x1858*x2025 - x1859*x2024 - x1860*x2024 - x1863*x2028 - x1864*x2028 - x1865*x2020 - x1866*x2020 - x1867*x2024 - x1868*x2020 - x1871*x2029 - x1872*x2029 - x1873*x2024 - x1877*x2031 - x1878*x2024 - x1879*x2031 - x1882*x2021*x935 - x1886*x2034 - x1887*x2029 - x1888*x2029 - x1893*x2023 - x1898*x2035 - x1900*x2035 - x1902*x2031 - x1903*x2029 - x1904*x2034 - x1905*x2029 - x1906*x2028 - x1907*x2028 - x1908*x2029 - x1909*x2030 - x1911*x2036 - x1912*x2029 - x1913*x2028 - x1914*x2029 - x1916*x2026*x993 - x1917*x2036 - x1919*x2036 - x1920*x2036 - x1922*x2028 - x1923*x2031 - x1925*x2038 - x1926*x2038 - x1927*x2028 - x1928*x2031 - x1933*x2031 - x1934*x2031 - x1936*x2039 - x1938*x2039 - x1939*x2038 - x1940*x2038 - x1941*x2034 - x1942*x2034 - x1944*x2041 - x1945*x2031 - x1946*x2031 - x1947*x2031 - x1948*x2041 - x1949*x2031 - x1952*x2034 - x1953*x2036 - x1955*x2036 - x1956*x2036 - x1957*x2034 - x1958*x2036 - x1959*x2042 - x1961*x2042 - x1962*x2034 - x1963*x2041 - x1964*x2041 - x1966*x2036 - x1968*x2038 - x1969*x2034 - x1970*x2036 - x1971*x2037 - x1972*x2034 - x1973*x2036 - x1974*x2036 - x1975*x2036 - x1976*x2038 - x1977*x2038 - x1978*x2041 - x1979*x2038 - x1980*x2038 - x1981*x2041 - x1982*x2038 - x1983*x2038 - x1984*x2038 - x1985*x2038 - x1986*x2041 - x1988*x2041 - x1989*x2040 - x1990*x2041 - x1991*x2038 - x1992*x2041 - x1993*x2038 - x1994*x2041 - x1995*x2041 - x1996*x2041 - x1997*x2041 - x1998*x2041 - x1999*x2041 - x2000*x2041 - x2008*x907 - x2008*x910 - x2015*x915 - x2015*x918)/a

    return [da_dt, de_dt, di_dt, dw_dt, dOmega_dt, dM_dt]

# --- Corpi celesti --- #

Terra = Body('Terra', np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 5.972e24, 6371.0, 398600.4418, 86164.1) 
Moon = Body('Luna', np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 7.342e22, 1738.0, 4902.79996708864, 27.321661 * 86400)
