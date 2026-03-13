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
    
def F_lmp(l, m, p, I):

    """
    Calcola la funzione di inclinazione di Kaula F_lmp(I).
    Basato sull'equazione (9-30) del Vallado.
    
    Input:
       l, m, p : indici interi 
       I       : Inclinazione (rad)
       
    Output:
       F_val   : valore della funzione (float)
    """
    
    # 1. Calcolo di k (parte intera di (l-m)/2)
    # Usa la divisione intera // per ottenere un indice intero
    k = (l - m) // 2
    
    # Inizializzo valore totale a 0.0
    F_val = 0.0
    
    # 2. Prima sommatoria (su t): da 0 a min(p, k)
    t_max = min(p, k)
    
    # Ricorda: range(a, b) in Python va da a fino a b-1. Quindi usiamo t_max + 1
    for t in range(t_max + 1):
        
        # Termine da moltiplicare al seno
        # math.factorial richiede interi positivi. 
        # Matematicamente, con i limiti imposti su t, gli argomenti sono sempre >= 0.
        num = math.factorial(2*l - 2*t)
        den = (math.factorial(t) * math.factorial(l - t) * math.factorial(l - m - 2*t) * (2**(2*l - 2*t)))
        
        term = num / den
        
        # Termine seno: sin(I)^(l-m-2t)
        term_sin = np.sin(I)**(l - m - 2*t)
        
        # Seconda Sommatoria (su s): da 0 a m
        sum_s = 0.0
        for s in range(m + 1):
            # termine binomiale (m su s) -> math.comb(n, k)
            term_bin = math.comb(m, s)
            
            # Termine Coseno: cos(I)^s
            term_cos = np.cos(I)**s
            
            # Terza Sommatoria (su c)
            sum_c = 0.0
            
            # Limiti della sommatoria in c
            c_min = max(0, p - t - m + s)
            c_max = min(l - m - 2*t + s, p - t)
            
            # range va fino a c_max + 1 per includere c_max
            for c in range(c_min, c_max + 1):
                
                # Argomenti del primo binomiale: (l-m-2t+s) su (c)
                n1 = l - m - 2*t + s
                k1 = c
                
                # Argomenti del secondo binomiale: (m-s) su (p-t-c)
                n2 = m - s
                k2 = p - t - c
                
                # Controllo dei binomiali
                check1 = (k1 >= 0) and (k1 <= n1)
                check2 = (k2 >= 0) and (k2 <= n2)
                
                if check1 and check2:
                    bin1 = math.comb(n1, k1)
                    bin2 = math.comb(n2, k2)
                    
                    # Termine (-1)^(c-k)
                    # Nota: qui 'k' è quello calcolato all'inizio della funzione
                    term_segno = (-1)**(c - k)
                    
                    sum_c += bin1 * bin2 * term_segno
            
            # Aggiungi il risultato della somma su c alla somma su s
            sum_s += term_bin * term_cos * sum_c
            
        # Aggiungi tutto alla somma totale su t
        F_val += term * term_sin * sum_s
        
    return F_val

def G_lpq(l, p, q, e):
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

def estrapolazione_file_grail(degree, filename='gggrx_0900c_sha.tab.txt'):
    """
    Legge il file dei coefficienti GRAIL e restituisce quelli denormalizzati
    con l<=degree e m<=degree.
    
    Input:
        degree   : grado massimo (int)
        filename : percorso del file .txt/.tab (str, opzionale)
        
    Output:
        df_out   : pandas DataFrame con colonne ['l', 'm', 'C', 'S']
    """
    
    # Verifica esistenza file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Il file '{filename}' non è stato trovato nella cartella.")

    # 1. Legge la tabella
    #    Modificato per leggere file separati da virgola (CSV) e saltare la prima riga di metadati
    try:
        data = pd.read_csv(filename, sep=',', header=None, comment='%', skiprows=1,
                           usecols=[0, 1, 2, 3], names=['l', 'm', 'C', 'S'])
    except Exception as e:
        print("Errore nella lettura del file. Verifica il formato.")
        raise e

    # 2. Maschera per l e m fino al valore degree scelto
    #    In pandas si filtra così: df[condizione]
    mask = (data['l'] <= degree) & (data['m'] <= degree)
    df_filtered = data[mask].copy()

    # 3. Denormalizzazione (Conversione per Kaula)
    #    Creiamo due liste vuote per salvare i risultati
    C_unnorm = []
    S_unnorm = []
    
    #    Iteriamo sulle righe del DataFrame filtrato
    for index, row in df_filtered.iterrows():
        curr_l = int(row['l'])
        curr_m = int(row['m'])
        val_C = row['C']
        val_S = row['S']
        
        # Calcolo del fattore di denormalizzazione
        # Formula: sqrt( (l-m)! * (2l+1) * k / (l+m)! )
        # dove k = 1 se m=0, k = 2 se m > 0
        
        if curr_m == 0:
            delta_factor = 1
        else:
            delta_factor = 2
        
        # Calcolo fattore
        # math.factorial gestisce interi molto grandi automaticamente in Python
        numeratore = math.factorial(curr_l - curr_m) * (2 * curr_l + 1) * delta_factor
        denominatore = math.factorial(curr_l + curr_m)
        
        factor = math.sqrt(numeratore / denominatore)
        
        # Applicazione conversione e salvataggio nella lista
        C_unnorm.append(val_C * factor)
        S_unnorm.append(val_S * factor)

    # 4. Aggiorniamo il DataFrame con i valori denormalizzati
    df_filtered['C'] = C_unnorm
    df_filtered['S'] = S_unnorm
    
    # (Opzionale) ordina per l poi m
    df_filtered = df_filtered.sort_values(by=['l', 'm'])
    
    # Resetta l'indice (estetico, fa partire le righe da 0, 1, 2...)
    df_filtered = df_filtered.reset_index(drop=True)

    # Riepilogo a schermo
    print(f"Trovati {len(df_filtered)} coefficienti con l<={degree} e m<={degree}")
    # print(df_filtered) # Decommenta se vuoi stampare tutta la tabella
    
    return df_filtered

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
    # (Usiamo numpy.linspace per generare facilmente i tempi)
    
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

# --- Corpi celesti --- #

Terra = Body('Terra', np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 5.972e24, 6371.0, 398600.4418, 86164.1) 
Moon = Body('Luna', np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 7.342e22, 1738.0, 4902.79996708864, 27.321661 * 86400)
