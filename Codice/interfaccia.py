import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import math

import orbital_lib as ol
import importlib
importlib.reload(ol) # <-- FORZA L'AGGIORNAMENTO DELLA LIBRERIA!

# ---------------------- Config Costanti ----------------------
EARTH_RADIUS = 6371.0
MU_EARTH = 398600.4418
ATMOSPHERE_SCALE = 1.02
MAX_TRAJ_POINTS = 6000
TRIM_TRAJ_TO = 4000

PRESETS = {
    'LEO 400km equatoriale': {'alt': 400.0, 'incl': 0.0},
    'LEO 800km polare':      {'alt': 800.0, 'incl': 90.0},
    'GTO':                   {'perigee_alt': 250.0, 'apogee_alt': 35786.0, 'incl': 0.0},
    'GEO':                   {'alt': 35786.0, 'incl': 0.0}
}

def ensure_state():
    defaults = {
        'running': False, 'position': [7000.0, 0.0, 0.0], 'velocity': [0.0, 7.5, 0.0],
        'trajectory': [], 'mean_anomaly': 0.0, 'sim_time': 0.0, 'mode': 'Anomalia',
        'elements': None, 'precomputed_sol': None, 'precomputed_idx': 0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def compute_orbit_step(delta_M=None, dt=None):
    if st.session_state.elements is None: return
    a, e, inc, w, Omega, l0, n, t_peri = st.session_state.elements
    if st.session_state.mode == 'Anomalia':
        if delta_M is None: return
        M_new = (st.session_state.mean_anomaly + delta_M) % (2 * np.pi)
        st.session_state.sim_time = (M_new - l0) / n
        st.session_state.mean_anomaly = M_new
    else:
        if dt is None: return
        st.session_state.sim_time += dt
        M_new = (l0 + n * st.session_state.sim_time) % (2 * np.pi)
        st.session_state.mean_anomaly = M_new

    new_pos, new_vel = ol.kep2car(a, e, inc, w, Omega, M_new, MU_EARTH)
    st.session_state.position = new_pos.tolist()
    st.session_state.velocity = new_vel.tolist()
    st.session_state.trajectory.append(st.session_state.position.copy())
    if len(st.session_state.trajectory) > MAX_TRAJ_POINTS:
        st.session_state.trajectory = st.session_state.trajectory[-TRIM_TRAJ_TO:]
    return dict(e=e, a=a, i=inc, M=M_new, n=n, omega=w, Omega=Omega, T=2*np.pi/n)

def compute_step_from_precomputed(dt_slider):
    """Avanza sfogliando i dati pre-calcolati ad alta densità per una curva perfetta."""
    sol = st.session_state.precomputed_sol
    idx = st.session_state.precomputed_idx
    
    if sol is None or idx >= len(sol.t) - 1:
        st.session_state.running = False
        return None
        
    current_time = sol.t[idx]
    target_time = current_time + dt_slider # Calcoliamo a che tempo vogliamo arrivare
    
    new_idx = idx
    # Aggiungiamo ALLA TRAIETTORIA tutti i punti intermedi in un colpo solo
    while new_idx < len(sol.t) and sol.t[new_idx] <= target_time:
        a_k, e_k, inc_k, w_k, Omega_k, M_k = sol.y[:, new_idx]
        pos, vel = ol.kep2car(a_k, e_k, inc_k, w_k, Omega_k, M_k, MU_EARTH)
        st.session_state.trajectory.append(pos.tolist())
        new_idx += 1
        
    if len(st.session_state.trajectory) > MAX_TRAJ_POINTS:
        st.session_state.trajectory = st.session_state.trajectory[-TRIM_TRAJ_TO:]
        
    if new_idx == idx: new_idx += 1
    if new_idx >= len(sol.t):
        new_idx = len(sol.t) - 1
        st.session_state.running = False
        
    st.session_state.precomputed_idx = new_idx
    
    # Aggiorniamo la posizione visiva allo stato finale di questo scatto
    a_new, e_new, inc_new, w_new, Omega_new, M_new = sol.y[:, new_idx]
    n_new = np.sqrt(MU_EARTH / a_new**3)
    pos, vel = ol.kep2car(a_new, e_new, inc_new, w_new, Omega_new, M_new, MU_EARTH)
    
    st.session_state.position = pos.tolist()
    st.session_state.velocity = vel.tolist()
    st.session_state.sim_time = sol.t[new_idx]
    st.session_state.elements = (a_new, e_new, inc_new, w_new, Omega_new, M_new, n_new, 0.0)
    
    return dict(e=e_new, a=a_new, i=inc_new, M=M_new, n=n_new, omega=w_new, Omega=Omega_new, T=2*math.pi/n_new)

def generate_earth_mesh(res_lat=90, res_lon=180, radius=EARTH_RADIUS):
    phi = np.linspace(0, np.pi, res_lat)
    theta = np.linspace(0, 2*np.pi, res_lon)
    th_grid, ph_grid = np.meshgrid(theta, phi)
    x = radius * np.sin(ph_grid) * np.cos(th_grid)
    y = radius * np.sin(ph_grid) * np.sin(th_grid)
    z = radius * np.cos(ph_grid)
    shade = np.clip(np.sin(ph_grid) * np.cos(th_grid)*0.8 + np.cos(ph_grid)*0.4, 0, 1)
    return x, y, z, shade

def build_figure(show_axes, show_traj, limit):
    if 'earth_mesh' not in st.session_state: st.session_state.earth_mesh = generate_earth_mesh()
    x, y, z, shade = st.session_state.earth_mesh
    fig = go.Figure()
    
    if show_axes:
        fig.add_scatter3d(x=[-limit, limit], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=4))
        fig.add_scatter3d(x=[0,0], y=[-limit, limit], z=[0,0], mode='lines', line=dict(color='green', width=4))
        fig.add_scatter3d(x=[0,0], y=[0,0], z=[-limit, limit], mode='lines', line=dict(color='blue', width=4))
        
    fig.add_surface(x=x, y=y, z=z, surfacecolor=shade, colorscale='Blues', showscale=False, opacity=0.95)
    
    px, py, pz = st.session_state.position
    fig.add_scatter3d(x=[px], y=[py], z=[pz], mode='markers', marker=dict(size=5, color='orange'))
    
    if show_traj and len(st.session_state.trajectory) > 1:
        traj = np.array(st.session_state.trajectory)
        fig.add_scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode='lines', line=dict(color='yellow', width=2))
        
    # --- LA MODIFICA È QUI SOTTO ---
    fig.update_layout(
        uirevision='locked',  # <-- QUESTA È LA MAGIA CHE BLOCCA LA TELECAMERA!
        scene=dict(
            bgcolor='black', 
            xaxis=dict(range=[-limit, limit]), 
            yaxis=dict(range=[-limit, limit]), 
            zaxis=dict(range=[-limit, limit]), 
            aspectmode='cube'
        ), 
        margin=dict(l=0, r=0, b=0, t=0), 
        showlegend=False
    )
    return fig

# ---------------------- Main App ----------------------
def main():
    st.set_page_config(page_title='Orbita 3D', layout='wide')
    # NUOVO TITOLO PER CAPIRE SE IL FILE SI È AGGIORNATO
    st.title('AquinSpace - Simulatore di Orbite 3D ') 
    ensure_state()
    
    with st.sidebar:
        st.header('Configurazione iniziale')
        sx = st.number_input('Posizione X', value=float(st.session_state.position[0]))
        sy = st.number_input('Posizione Y', value=float(st.session_state.position[1]))
        sz = st.number_input('Posizione Z', value=float(st.session_state.position[2]))
        vx = st.number_input('Velocità VX', value=float(st.session_state.velocity[0]))
        vy = st.number_input('Velocità VY', value=float(st.session_state.velocity[1]))
        vz = st.number_input('Velocità VZ', value=float(st.session_state.velocity[2]))
        
        mode = st.radio('Modalità avanzamento', ['Anomalia','Tempo'], index=0 if st.session_state.mode=='Anomalia' else 1)
        st.session_state.mode = mode
        delta_M = st.slider('ΔM per step (rad)', 0.001, 0.05, 0.01, step=0.001, disabled=mode!='Anomalia')
        dt = st.slider('Δt per step (s)', 10.0, 5000.0, 100.0, step=10.0, disabled=mode!='Tempo')
        
        use_perturbation = st.checkbox('Attiva Perturbazione J2', value=False)
        if use_perturbation and mode == 'Anomalia':
            st.warning("⚠️ Passa alla modalità 'Tempo' per usare le perturbazioni.")
            
        max_range = st.number_input('Scala vista (km)', value=7000, min_value=7000, step=500)

    colA, colB, colC = st.columns(3)
    if colA.button('▶️ Start'):
        st.session_state.position = [sx, sy, sz]
        st.session_state.velocity = [vx, vy, vz]
        try:
            st.session_state.elements = ol.car2kep(st.session_state.position, st.session_state.velocity, MU_EARTH)
        except Exception as ex:
            st.error(f"Errore: {ex}")
            st.session_state.running = False
        else:
            st.session_state.sim_time = 0.0
            st.session_state.trajectory = [list(st.session_state.position)]
            
            if use_perturbation and mode == 'Tempo':
                with st.spinner("🚀 Pre-calcolo dell'orbita perturbata in corso..."):
                    try:
                        sol = ol.precompute_perturbed_orbit(
                            elements=st.session_state.elements,
                            mu=MU_EARTH, J2=1.08262668e-3, Re=EARTH_RADIUS, num_orbits=50
                        )
                        st.session_state.precomputed_sol = sol
                        st.session_state.precomputed_idx = 0
                        st.session_state.running = True
                    except Exception as e:
                        st.error(f"❌ Errore di pre-calcolo: {e}")
                        st.session_state.running = False
            else:
                st.session_state.precomputed_sol = None
                st.session_state.running = True

    if colB.button('⏸️ Stop'): st.session_state.running = False
    if colC.button('🔄 Reset Traccia'): st.session_state.trajectory = [list(st.session_state.position)]

    # --- Creazione di un "Contenitore Fisso" per bloccare lo sfarfallio ---
    plot_placeholder = st.empty()
    text_placeholder = st.empty()

    orbital_info = None
    if st.session_state.running:
        if use_perturbation and st.session_state.mode == 'Tempo' and st.session_state.get('precomputed_sol') is not None:
            # Passiamo il dt dallo slider alla nuova funzione!
            orbital_info = compute_step_from_precomputed(dt_slider=dt) 
        else:
            if st.session_state.mode == 'Anomalia': orbital_info = compute_orbit_step(delta_M=delta_M)
            else: orbital_info = compute_orbit_step(dt=dt)
            
    fig = build_figure(True, True, max_range)
    
    # Inseriamo il grafico e il testo nel contenitore fisso
    with plot_placeholder.container():
        st.plotly_chart(fig, use_container_width=True)
        
    with text_placeholder.container():
        if orbital_info:
            st.markdown(f"**Tempo simulato:** {st.session_state.sim_time:.1f} s")
        
    if st.session_state.running:
        time.sleep(0.15) # Pausa allungata per far funzionare bene il bottone Stop!
        st.rerun()

if __name__ == '__main__':
    main()