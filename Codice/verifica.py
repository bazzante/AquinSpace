#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modulo dinamica orbitale.
Richiede Python >= 3.6 (per uso di f-string). Se eseguito con versione più vecchia interrompe.
"""
import sys
if sys.version_info < (3, 6):
    sys.stderr.write("Questo script richiede Python >= 3.6. Versione rilevata: {}\n".format(sys.version))
    sys.exit(1)

import numpy as np
import math
from scipy.optimize import root
import orbital_lib as ol  
import matplotlib.pyplot as plt 

G = 6.67430e-20  # km^3/kg/s^2, costante di gravitazione universale

class Body:
    """
    Classe per rappresentare un corpo celeste.
    
    Attributi:
      - nome: nome del corpo
      - position: posizione (vettore)
      - velocity: velocità (vettore)
      - mass: massa del corpo
      - gravitazionale_parameter: parametro gravitazionale standard (mu = G * mass)
      - R_eq: raggio equatoriale del corpo (utile per calcolare J2
    """
    def __init__(self, nome, position, velocity, mass, gravitazionale_parameter,R_eq):
        self.nome = nome
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.gravitazionale_parameter = gravitazionale_parameter
        self.R_eq = R_eq

Terra = Body("Terra", [0, 0, 0], [0, 0, 0],5.976*10**24, 398600, 6378.137,6378.137)
Moon = Body("Luna", [384400, 0, 0], [0, 1.022, 0],7.348*10**22, 4902.8, 1737.4)  





# 1. Vettori Iniziali (Il caso LEO che ti ho dato prima)
# --- Caso di Test B: Orbita Molniya (Alta eccentricità) ---
r_in = [ -1529.9, -2672.7, 6155.1 ] # km
v_in = [  8.72, -2.69, 4.21 ]   # km/s

orb_el= ol.car2kep(r_in, v_in, mu)
print("Elementi orbitali iniziali (a, e, i, omega, Omega, M):")
print(orb_el)

r_return, v_return = ol.kep2car(orb_el[0], orb_el[1], orb_el[2], orb_el[3], orb_el[4], orb_el[5], mu)
print("Vettori ricostruiti da kep2car:")
print("r:", r_return)
print("v:", v_return)

print("\n--- Inizio Propagazione Orbitale ---")

x_hist, y_hist, z_hist, vx_hist, vy_hist, vz_hist, t_hist, orbEl_hist = ol.propagate_perturbed_orbit(orb_el, mu)


# 4. Visualizzazione Grafica 3D (Opzionale ma molto utile!)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Disegniamo la Terra al centro come un punto blu
ax.scatter(0, 0, 0, color='blue', s=100, label='Centro Terra')

# Disegniamo l'orbita propagata
ax.plot(x_hist, y_hist, z_hist, color='red', label='Orbita Propagata')

# Disegniamo il punto di partenza
ax.scatter(r_in[0], r_in[1], r_in[2], color='green', s=50, label='Posizione Iniziale')

ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title('Propagazione dell\'Orbita Molniya')
ax.legend()

# Per avere gli assi con la stessa scala
max_range = np.array([max(x_hist)-min(x_hist), max(y_hist)-min(y_hist), max(z_hist)-min(z_hist)]).max() / 2.0
mid_x = (max(x_hist)+min(x_hist)) * 0.5
mid_y = (max(y_hist)+min(y_hist)) * 0.5
mid_z = (max(z_hist)+min(z_hist)) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
