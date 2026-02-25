import sympy as sp
import numpy as np
import orbital_lib as ol  # Per leggere la tabella GRAIL
import sys

# Impostazioni
L_MAX = 5   # IMPORTANTE: Inizia con 5 per testare. 15 in simbolico richiede MOLTA RAM e tempo.
Q_MAX = 2   # Come nel tuo script
FILENAME_OUT = "kaula_force_model.py"

print(f"--- Inizio Generazione Simbolica (L_max={L_MAX}) ---")

# 1. Dichiarazione variabili simboliche
# real=True aiuta sympy a semplificare
a, e, I, w, Omega, M, theta_GMST = sp.symbols('a e I w Omega M theta_GMST', real=True)
mu = 4902.79996708864
R_ref = 1738.0

# Variabili ausiliarie simboliche
n = sp.sqrt(mu / a**3)
eta = sp.sqrt(1 - e**2)

# 2. Estrapolazione coefficienti Grail (usiamo la tua funzione Python)
print("Lettura coefficienti GRAIL...")
try:
    grail_table = ol.estrapolazione_file_grail(L_MAX)
except Exception as err:
    print(f"Errore: {err}")
    sys.exit()

# 3. Costruzione Potenziale (Cicli)
print("Costruzione del potenziale U_pert (questo potrebbe richiedere tempo)...")
U_pert = 0

# Definiamo versioni locali di F e G compatibili con Sympy
# (Le tue versioni in orbital_lib usano math.factorial che non accetta simboli, 
#  quindi le riscriviamo qui usando sp.factorial e sp.sin/cos)

def F_lmp_sym(l, m, p, I_sym):
    k = (l - m) // 2
    val = 0
    t_max = min(p, k)
    
    for t in range(t_max + 1):
        num = sp.factorial(2*l - 2*t)
        den = (sp.factorial(t) * sp.factorial(l - t) * sp.factorial(l - m - 2*t) * (2**(2*l - 2*t)))
        term = num / den
        term_sin = sp.sin(I_sym)**(l - m - 2*t)
        
        sum_s = 0
        for s in range(m + 1):
            term_bin = sp.binomial(m, s)
            term_cos = sp.cos(I_sym)**s
            
            sum_c = 0
            c_min = max(0, p - t - m + s)
            c_max = min(l - m - 2*t + s, p - t)
            
            for c in range(c_min, c_max + 1):
                n1 = l - m - 2*t + s
                k1 = c
                n2 = m - s
                k2 = p - t - c
                
                if (k1 <= n1) and (k2 <= n2):
                    bin1 = sp.binomial(n1, k1)
                    bin2 = sp.binomial(n2, k2)
                    term_segno = (-1)**(c - k)
                    sum_c += bin1 * bin2 * term_segno
            
            sum_s += term_bin * term_cos * sum_c
        val += term * term_sin * sum_s
    return val

def G_lpq_sym(l, p, q, e_sym):
    if p <= l / 2:
        p_prime = p
    else:
        p_prime = l - p
        
    sum_val = 0
    max_d = p_prime - 1
    
    for d in range(max_d + 1):
        n1 = l - 1
        k1 = 2 * d + l - 2 * p_prime
        n2 = k1 
        k2 = d
        
        if (k1 >= 0) and (k1 <= n1) and (k2 >= 0) and (k2 <= n2):
            bin1 = sp.binomial(n1, k1)
            bin2 = sp.binomial(n2, k2)
            term_exp = (e_sym / 2)**(2 * d + l - 2 * p_prime)
            sum_val += bin1 * bin2 * term_exp

    term = (1 - e_sym**2)**-(l - sp.Rational(1, 2)) # Rational evita float approx
    return term * sum_val

# Ciclo principale
count = 0
total_cycles = len(grail_table) * (L_MAX+1) * (2*Q_MAX + 1) # stima grezza

for idx, row in grail_table.iterrows():
    l = int(row['l'])
    m = int(row['m'])
    C_lm = float(row['C'])
    S_lm = float(row['S'])
    
    if l < 2: continue # Kaula parte da l=2

    for p in range(0, l + 1):
        for q in range(-Q_MAX, Q_MAX + 1):
            
            # A. Funzione F
            F_val = F_lmp_sym(l, m, p, I) # Passiamo I simbolo
            
            # B. Funzione G (solo se q = 2p - l)
            if q == (2*p - l):
                G_val = G_lpq_sym(l, p, q, e) # Passiamo e simbolo
            else:
                G_val = 0 # O 1, come nel tuo script originale avevi 0
                
            if G_val == 0 or F_val == 0:
                continue

            # D. Funzione S
            theta_lmpq = (l - 2*p)*w + (l - 2*p + q)*M + m*(Omega - theta_GMST)
            
            if (l - m) % 2 == 0:
                S_val = C_lm * sp.cos(theta_lmpq) + S_lm * sp.sin(theta_lmpq)
            else:
                S_val = -S_lm * sp.cos(theta_lmpq) + C_lm * sp.sin(theta_lmpq)
            
            # E. Termine completo
            term = (mu/a) * (R_ref/a)**l * F_val * G_val * S_val
            U_pert += term

print("Calcolo derivate parziali...")
dR_da = sp.diff(U_pert, a)
dR_de = sp.diff(U_pert, e)
dR_di = sp.diff(U_pert, I)
dR_dw = sp.diff(U_pert, w)
dR_dOmega = sp.diff(U_pert, Omega)
dR_dM = sp.diff(U_pert, M)

print("Assemblaggio Equazioni di Lagrange...")
eps_val = 1e-12
denom_e = e + eps_val
denom_sinI = sp.sin(I) + eps_val

da_dt = (2 / (n * a)) * dR_dM
de_dt = eta**2 / (n * a**2 * denom_e) * dR_dM - eta / (n * a**2 * denom_e) * dR_dw
di_dt = sp.cos(I) / (n * a**2 * eta * denom_sinI) * dR_dw - 1 / (n * a**2 * eta * denom_sinI) * dR_dOmega
dw_dt = eta / (n * a**2 * denom_e) * dR_de - (sp.cos(I)/denom_sinI) / (n * a**2 * eta ) * dR_di
dOmega_dt = 1 / (n * a**2 * eta * denom_sinI) * dR_di
dl_dt = n - eta**2 / (n * a**2 * denom_e) * dR_de - 2 / (n * a) * dR_da

# Creazione lista delle equazioni
dLagEq = [da_dt, de_dt, di_dt, dw_dt, dOmega_dt, dl_dt]

# 4. Generazione File Python
print(f"Scrittura file {FILENAME_OUT}...")

# Usiamo il printer di numpy per convertire sin(x) -> numpy.sin(x) etc
from sympy.printing.numpy import NumPyPrinter

with open(FILENAME_OUT, 'w') as f:
    f.write("import numpy as np\n\n")
    f.write("def diff_LagEquations_Kaula_15(a, e, I, w, Omega, M, theta_GMST):\n")
    f.write("    # Funzione generata automaticamente da script simbolico\n")
    f.write("    # Equazioni di Lagrange per potenziale lunare\n\n")
    
    # Nomi variabili output
    varnames = ['da_dt', 'de_dt', 'di_dt', 'dw_dt', 'dOmega_dt', 'dl_dt']
    
    for i, eq in enumerate(dLagEq):
        # Convertiamo l'equazione simbolica in stringa codice Python/Numpy
        code_str = NumPyPrinter().doprint(eq)
        f.write(f"    {varnames[i]} = {code_str}\n")
    
    f.write("\n    return np.array([da_dt, de_dt, di_dt, dw_dt, dOmega_dt, dl_dt])\n")

print("Finito! Ora puoi importare la funzione nel tuo main.")