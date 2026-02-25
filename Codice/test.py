import numpy as np
from scipy.integrate import solve_ivp
import orbital_lib as ol  # Assumiamo che le tue funzioni siano qui

# ---------------------------------------------------------
# 1. Dati Luna
# ---------------------------------------------------------
mu = 4902.79996708864             # parametro gravitazionale lunare [km^3/s^2]
R_eq = 1738.0                     # Raggio di riferimento lunare [Km]

# Calcolo velocità angolare luna
T_moon_sidereal = 27.321661 * 86400  # Tempo in secondi per compiere un giro
w_moon = 2 * np.pi / T_moon_sidereal # Velocità angolare luna
theta_0 = 0.0

# ---------------------------------------------------------
# 2. Vettori Iniziali 
# ---------------------------------------------------------

# --- SCENARIO 1: inclinazione 45 ---
r_in = np.array([1798.0, 0.0, 0.0])        # km 
v_in = np.array([0.0, 1.1682, 1.1682])     # km/s 

# --- SCENARIO 2: LRO (Polare Bassa - 50 km) ---
# r_in = np.array([1788.0, 0.0, 0.0])
# v_in = np.array([0.0, 0.0, 1.6556])
# print('=== SCENARIO 2: LRO POLARE (h=50km) ===')

# --- SCENARIO 3: Apollo Parking Orbit ---
# r_in = np.array([1848.0, 0.0, 0.0])
# v_in = np.array([0.0, 1.6285, 0.0])
# print('=== SCENARIO 3: APOLLO 11 (h=110km) ===')

print(f"r: {r_in}")
print(f"v: {v_in}\n")

# ---------------------------------------------------------
# 3. Conversione in Elementi Orbitali (t=0)
# ---------------------------------------------------------
# Nota: La nostra funzione car2kep in python restituisce:
# n, ecc, I, w, Omega_hat, l, a, tperi
n_mean, ecc, I, w, Omega, M, a, tperi = ol.car2kep(r_in, v_in, mu)

# Vettore di stato iniziale per l'integratore: [a, e, I, w, Omega, M]
y0 = np.array([a, ecc, I, w, Omega, M])

print('=== 2. PARAMETRI ORBITALI ===')
print(f"a: {a:.2f}, e: {ecc:.4f}, I: {np.degrees(I):.2f} deg")
print(f"w: {np.degrees(w):.2f} deg, Om: {np.degrees(Omega):.2f} deg, M: {np.degrees(M):.2f} deg")

# Periodo orbitale
T = 2 * np.pi / n_mean

# ---------------------------------------------------------
# 4. Preparazione Integrazione (ode45 -> solve_ivp)
# ---------------------------------------------------------

# Tempo di simulazione
num_points = 50000
t_start = 0
t_end = 10 * T
t_eval = np.linspace(t_start, t_end, num_points) # Punti specifici dove vogliamo l'output

# Opzioni integratore
# In Python rtol e atol si passano direttamente a solve_ivp
rel_tol = 1e-14
abs_tol = 1e-16

# --- Definizione Wrapper ---
# In Python definiamo la funzione wrapper prima di chiamare il solver.
# solve_ivp richiede una funzione f(t, y)
def eom_wrapper(t, y):
    # Estrai le variabili di stato
    a_in, e_in, I_in, w_in, Om_in, M_in = y
    
    # 1. Calcola l'angolo di rotazione della Luna al tempo t attuale
    current_theta_GMST = w_moon * t
    
    # 2. Chiama le equazioni di Lagrange (che devono essere in orbital_lib)
    # ATTENZIONE: Questa funzione deve essere definita in orbital_lib!
    # L'ordine degli argomenti deve corrispondere a quello della tua funzione MATLAB
    
    dydt = ol.diff_LagEquations_Kaula_15(a_in, e_in, I_in, w_in, Om_in, M_in, current_theta_GMST)
    
    # Assicurati che dydt sia una lista o un array numpy piatto
    return dydt

# --- Chiamata al Solver ---
print("Inizio propagazione...")

# solve_ivp(funzione, intervallo_tempo, condizioni_iniziali, metodo, t_eval, argomenti_extra...)
solution = solve_ivp(eom_wrapper, 
                     (t_start, t_end), 
                     y0, 
                     method='RK45', 
                     t_eval=t_eval, 
                     rtol=rel_tol, 
                     atol=abs_tol)

# ---------------------------------------------------------
# 5. Risultati
# ---------------------------------------------------------
# solution.t contiene i tempi
# solution.y contiene gli stati (dimensione 6 x N)
# Per averli come in MATLAB (N x 6), facciamo la trasposta .T

t_out = solution.t
y_out = solution.y.T

print(f"Propagazione completata. Punti calcolati: {len(t_out)}")

# Esempio di accesso ai dati finali
a_final = y_out[-1, 0]
print(f"Semiasse maggiore finale: {a_final:.4f} km")                                            