import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import math
import orbital_lib as ol
import importlib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


# Forza la ricarica della libreria
importlib.reload(ol) 

# ---------------------- Configurazione Pagina ----------------------
st.set_page_config(
    page_title='AquinSpace',
    layout='wide', 
    initial_sidebar_state="collapsed" 
)

# ---------------------- Costanti & Preset ----------------------
EARTH_RADIUS = 6371.0
MU_EARTH = 398600.4418
MAX_TRAJ_POINTS = 6000
TRIM_TRAJ_TO = 4000

PRESETS = {
    'LEO 400km equatoriale': {'alt': 400.0, 'incl': 0.0},
    'LEO 800km polare':      {'alt': 800.0, 'incl': 90.0},
    'GTO':                   {'perigee_alt': 250.0, 'apogee_alt': 35786.0, 'incl': 0.0},
    'GEO':                   {'alt': 35786.0, 'incl': 0.0}
}

# ---------------------- Funzioni di Stato e Reset ----------------------
def ensure_state():
    """Inizializza le variabili di stato se non esistono."""
    defaults = {
        'running': False, 'position': [7000.0, 0.0, 0.0], 'velocity': [0.0, 7.5, 0.0],
        'trajectory': [], 'mean_anomaly': 0.0, 'sim_time': 0.0, 'mode': 'Anomalia',
        'elements': None, 'precomputed_sol': None, 'precomputed_idx': 0,
        'earth_mesh': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def apply_preset(name):
    """Applica i valori del preset allo stato."""
    if name not in PRESETS: return
    data = PRESETS[name]
    if name == 'GEO':
        r = EARTH_RADIUS + data['alt']
        v = np.sqrt(MU_EARTH / r)
        st.session_state.position = [r, 0.0, 0.0]
        st.session_state.velocity = [0.0, v, 0.0]
    elif 'perigee_alt' in data: # GTO
        rp = EARTH_RADIUS + data['perigee_alt']
        ra = EARTH_RADIUS + data['apogee_alt']
        a = (rp + ra) / 2
        vp = np.sqrt(MU_EARTH * (2/rp - 1/a))
        st.session_state.position = [rp, 0.0, 0.0]
        st.session_state.velocity = [0.0, vp, 0.0]
    else: # LEO
        r = EARTH_RADIUS + data['alt']
        v_mag = np.sqrt(MU_EARTH / r)
        incl_rad = np.radians(data['incl'])
        st.session_state.position = [r, 0.0, 0.0]
        st.session_state.velocity = [0.0, v_mag * np.cos(incl_rad), v_mag * np.sin(incl_rad)]
    
    st.session_state.trajectory = [list(st.session_state.position)]

# ---------------------- Funzioni di Calcolo e Grafica ----------------------
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
    return dict(T=2*np.pi/n)

def compute_step_from_precomputed(dt_slider):
    sol = st.session_state.precomputed_sol
    idx = st.session_state.precomputed_idx
    
    if sol is None or idx >= len(sol.t) - 1:
        st.session_state.running = False
        return None
        
    current_time = sol.t[idx]
    target_time = current_time + dt_slider 
    
    new_idx = idx
    while new_idx < len(sol.t) and sol.t[new_idx] <= target_time:
        a_k, e_k, inc_k, w_k, Omega_k, M_k = sol.y[:, new_idx]
        M_k = M_k % (2 * np.pi) 
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
    
    a_new, e_new, inc_new, w_new, Omega_new, M_new = sol.y[:, new_idx]
    n_new = np.sqrt(MU_EARTH / a_new**3)
    M_new = M_new % (2 * np.pi) 
    
    pos, vel = ol.kep2car(a_new, e_new, inc_new, w_new, Omega_new, M_new, MU_EARTH)
    
    st.session_state.position = pos.tolist()
    st.session_state.velocity = vel.tolist()
    st.session_state.sim_time = sol.t[new_idx]
    st.session_state.elements = (a_new, e_new, inc_new, w_new, Omega_new, M_new, n_new, 0.0)
    
    return dict(T=2*math.pi/n_new)

def generate_earth_mesh():
    if st.session_state.earth_mesh is not None: return st.session_state.earth_mesh
    
    # MODIFICA QUI: Abbassiamo la risoluzione da 90x180 a 45x90!
    phi = np.linspace(0, np.pi, 45)
    theta = np.linspace(0, 2*np.pi, 90)
    
    th_grid, ph_grid = np.meshgrid(theta, phi)
    x = EARTH_RADIUS * np.sin(ph_grid) * np.cos(th_grid)
    y = EARTH_RADIUS * np.sin(ph_grid) * np.sin(th_grid)
    z = EARTH_RADIUS * np.cos(ph_grid)
    shade = np.clip(np.sin(ph_grid) * np.cos(th_grid)*0.8 + np.cos(ph_grid)*0.4, 0, 1)
    st.session_state.earth_mesh = (x, y, z, shade)
    return x, y, z, shade

def build_figure(max_range):
    x, y, z, shade = generate_earth_mesh()
    fig = go.Figure()
    
    # Assi
    fig.add_scatter3d(x=[-max_range, max_range], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=4), hoverinfo='none')
    fig.add_scatter3d(x=[0,0], y=[-max_range, max_range], z=[0,0], mode='lines', line=dict(color='green', width=4), hoverinfo='none')
    fig.add_scatter3d(x=[0,0], y=[0,0], z=[-max_range, max_range], mode='lines', line=dict(color='blue', width=4), hoverinfo='none')
        
    # Terra
    fig.add_surface(x=x, y=y, z=z, surfacecolor=shade, colorscale='Blues', showscale=False, opacity=0.95, hoverinfo='none')
    
    # Satellite
    px, py, pz = st.session_state.position
    fig.add_scatter3d(x=[px], y=[py], z=[pz], mode='markers', marker=dict(size=6, color='orange', symbol='diamond'), name='Satellite')
    
    # Traiettoria
    if len(st.session_state.trajectory) > 1:
        traj = np.array(st.session_state.trajectory)
        fig.add_scatter3d(x=traj[:,0], y=traj[:,1], z=traj[:,2], mode='lines', line=dict(color='yellow', width=3), name='Traiettoria')
        
    fig.update_layout(
        uirevision='locked', 
        scene=dict(
            bgcolor='black',
            xaxis=dict(range=[-max_range, max_range], visible=False),
            yaxis=dict(range=[-max_range, max_range], visible=False),
            zaxis=dict(range=[-max_range, max_range], visible=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        height=700 
    )
    return fig

def build_animated_figure(x_hist, y_hist, z_hist, max_range):
    """
    Crea un grafico Plotly 3D con la traiettoria completa e un'animazione 
    fluida del satellite che la percorre.
    """
    x_earth, y_earth, z_earth, shade = generate_earth_mesh()
    fig = go.Figure()
    
    # 1. Traccia 0, 1, 2: Assi
    fig.add_scatter3d(x=[-max_range, max_range], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=4), hoverinfo='none', name='Asse X')
    fig.add_scatter3d(x=[0,0], y=[-max_range, max_range], z=[0,0], mode='lines', line=dict(color='green', width=4), hoverinfo='none', name='Asse Y')
    fig.add_scatter3d(x=[0,0], y=[0,0], z=[-max_range, max_range], mode='lines', line=dict(color='blue', width=4), hoverinfo='none', name='Asse Z')
        
    # 2. Traccia 3: Terra
    fig.add_surface(x=x_earth, y=y_earth, z=z_earth, surfacecolor=shade, colorscale='Blues', showscale=False, opacity=0.95, hoverinfo='none', name='Terra')
    
    # 3. Traccia 4: Traiettoria Statica Completa (la linea gialla)
    fig.add_scatter3d(x=x_hist, y=y_hist, z=z_hist, mode='lines', line=dict(color='yellow', width=2), name='Traiettoria', hoverinfo='none')
    
    # 4. Traccia 5: Il Satellite (Posizione Iniziale)
    fig.add_scatter3d(x=[x_hist[0]], y=[y_hist[0]], z=[z_hist[0]], mode='markers', marker=dict(size=6, color='orange', symbol='diamond'), name='Satellite')
    
    # --- Creazione dei Frame per l'animazione ---
    # Per evitare di bloccare il browser con 50.000 frame (nel caso di perturbazioni),
    # eseguiamo un "sottocampionamento" (decimation). Creiamo un frame ogni N punti.
    num_frames = min(len(x_hist), 500) # Massimo 500 frame per la fluidità
    step = max(1, len(x_hist) // num_frames)
    
    frames = []
    for i in range(0, len(x_hist), step):
        # Aggiorniamo SOLO la traccia del satellite (che è la traccia numero 5)
        frame = go.Frame(
            data=[go.Scatter3d(x=[x_hist[i]], y=[y_hist[i]], z=[z_hist[i]])],
            traces=[5] # Indice della traccia del satellite inserita sopra
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # --- Configurazione dei pulsanti Play/Pause ---
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=1.0, # Posizione dei pulsanti nel grafico
            buttons=[
                dict(label="▶ Play Animazione",
                     method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pausa",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )],
        uirevision='locked', 
        scene=dict(
            bgcolor='black',
            xaxis=dict(range=[-max_range, max_range], visible=False),
            yaxis=dict(range=[-max_range, max_range], visible=False),
            zaxis=dict(range=[-max_range, max_range], visible=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        height=700 
    )
    return fig
# ---------------------- MAIN APPLICATION ----------------------
def main():
    ensure_state()

    # --- INIEZIONE CSS PER SFONDO NERO E BOTTONI ---
    st.markdown("""
        <style>
        .stApp { background-color: #000000; }
        .stApp, h1, h2, h3, h4, h5, h6, p, label, span { color: #FFFFFF !important; }
        [data-testid="stExpander"] { background-color: #111111; border-color: #333333; }
        
        /* Stile personalizzato per i bottoni "primary" */
        button[kind="primary"] {
            background-color: #ff4b4b !important; /* Rosso */
            color: white !important;
            border: none;
            transition: 0.2s;
        }
        button[kind="primary"]:hover {
            background-color: #ff3333 !important; /* Rosso più scuro al passaggio del mouse */
        }
        button[kind="primary"]:active {
            background-color: #ffffff !important; /* Bianco durante il click */
            color: #ff4b4b !important; /* Testo rosso durante il click */
        }
        </style>
        """, unsafe_allow_html=True)

    # --- HEADER: Logo e Titolo AquinSpace ---
    # Aumentata la proporzione della prima colonna (da 1 a 1.5) per ingrandire il logo!
    col_logo, col_header_title = st.columns([1.5, 2.5], gap="large")
    
    with col_logo:
        try:
            # L'immagine ora si espanderà per riempire questa colonna molto più larga
            st.image(str(BASE_DIR / "logo_scritta.jpg"), use_container_width=True)
        except:
            st.warning("Logo non trovato")
            
    with col_header_title:
        # Ho aggiunto un po' di margine superiore (margin-top: 40px) per centrare 
        # la scritta rispetto al logo diventato più alto
        st.markdown("<h1 style='font-size: 5rem; font-weight: 800; margin-top: 40px;'>AquinSpace</h1>", unsafe_allow_html=True)

    st.divider()

    # --- LAYOUT PRINCIPALE A DUE COLONNE ---
    left_panel, right_panel = st.columns([1, 2], gap="medium")

    # ================== PANNELLO SINISTRO (Controlli) ==================
    with left_panel:
        st.subheader("Pannello di Controllo")
        
        with st.expander("1. Stato Iniziale e Preset", expanded=True):
            preset_choice = st.selectbox('Scegli un Preset', ['(Personalizzato)'] + list(PRESETS.keys()))
            if preset_choice != '(Personalizzato)':
                # ---> ECCO LA MODIFICA: Aggiunto type="primary" <---
                if st.button(f'Applica {preset_choice}', type="primary", use_container_width=True):
                    apply_preset(preset_choice)
                    st.rerun()

            col_pos, col_vel = st.columns(2)
            with col_pos:
                st.markdown("**Posizione (km)**")
                sx = st.number_input('X', value=float(st.session_state.position[0]), format="%.1f", key="in_sx")
                sy = st.number_input('Y', value=float(st.session_state.position[1]), format="%.1f", key="in_sy")
                sz = st.number_input('Z', value=float(st.session_state.position[2]), format="%.1f", key="in_sz")
            with col_vel:
                st.markdown("**Velocità (km/s)**")
                vx = st.number_input('VX', value=float(st.session_state.velocity[0]), format="%.3f", key="in_vx")
                vy = st.number_input('VY', value=float(st.session_state.velocity[1]), format="%.3f", key="in_vy")
                vz = st.number_input('VZ', value=float(st.session_state.velocity[2]), format="%.3f", key="in_vz")

        with st.expander("2. Parametri Simulazione", expanded=True):
            st.session_state.mode = st.radio('Modalità di avanzamento', ['Anomalia', 'Tempo'], 
                                             index=0 if st.session_state.mode=='Anomalia' else 1, horizontal=True)
            
            delta_M = st.slider('Step Anomalia (rad)', 0.001, 0.05, 0.01, step=0.001, disabled=st.session_state.mode!='Anomalia')
            dt = st.slider('Step Tempo (s)', 10.0, 5000.0, 100.0, step=10.0, disabled=st.session_state.mode!='Tempo')
            
            st.markdown("---")
            use_perturbation = st.toggle('Attiva Perturbazione J2', value=False)
            if use_perturbation and st.session_state.mode == 'Anomalia':
                st.error("Per usare J2, passa alla modalità 'Tempo'.")

        with st.expander("3. Vista e Azioni", expanded=True):
            max_range = st.number_input('Scala Visuale (km)', value=8000, min_value=7000, step=1000)
            
            st.markdown("##### Comandi")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            # Aggiunto type="primary" a tutti i bottoni per farli diventare rossi!
            start_pressed = btn_col1.button('▶️ START', type="primary", use_container_width=True)
            stop_pressed = btn_col2.button('⏸️ STOP', type="primary", use_container_width=True)
            reset_trace_pressed = btn_col3.button('🔄 TRACCIA', type="primary", use_container_width=True)

# ================== PANNELLO DESTRO (Visualizzazione e Appendice) ==================
    with right_panel:
        if stop_pressed: 
            st.session_state.trajectory_data = None # Resetta i dati
            
        if reset_trace_pressed: 
            st.session_state.trajectory_data = None
        
        # Inizializziamo una variabile nel session state per tenere in memoria la traiettoria pre-calcolata
        if 'trajectory_data' not in st.session_state:
            st.session_state.trajectory_data = None
            
        if start_pressed:
            st.session_state.position = [sx, sy, sz]
            st.session_state.velocity = [vx, vy, vz]
            try:
                # 1. Calcola gli elementi kepleriani iniziali
                kep_el = ol.car2kep(st.session_state.position, st.session_state.velocity, MU_EARTH)
                st.session_state.elements = kep_el
                
                with st.spinner("🚀 Pre-calcolo dell'orbita in corso... (attendere)"):
                    if use_perturbation:
                        # Calcola 100 orbite perturbate
                        x_h, y_h, z_h, vx_h, vy_h, vz_h, t_sol, y_sol = ol.propagate_perturbed_orbit(
                            kepElements=kep_el, mu=MU_EARTH, num_steps=5000, num_orbits=100
                        )
                    else:
                        # Calcola 1 singola orbita osculatrice periodica
                        x_h, y_h, z_h, vx_h, vy_h, vz_h, orbEl_hist = ol.osculating_orbit(
                            kepElements=kep_el, mu=MU_EARTH
                        )
                    
                    # Salviamo i risultati nel session state
                    st.session_state.trajectory_data = (x_h, y_h, z_h)
                    
            except Exception as ex:
                st.error(f"Errore nel calcolo orbitale: {ex}")

        # --- Visualizzazione Grafico 3D Animato ---
        if st.session_state.trajectory_data is not None:
            # Se abbiamo i dati, costruiamo il grafico animato
            x_h, y_h, z_h = st.session_state.trajectory_data
            fig = build_animated_figure(x_h, y_h, z_h, max_range)
        else:
            # Grafico di default (vuoto/solo posizione iniziale)
            fig = build_figure(max_range) # Usiamo la tua vecchia funzione solo come placeholder iniziale
            
        st.plotly_chart(fig, use_container_width=True, key="grafico_3d_main")
        
        # NOTA: Essendo un'animazione gestita dal browser, il tempo in secondi non si 
        # aggiornerà in tempo reale tramite Streamlit senza farlo sfarfallare. 
        # Rimuoviamo il container del tempo che si aggiorna in loop per ora.
        
        # --- Sezione Appendice Personale (Statica) ---
        # ... (Mantieni qui la tua st.divider() e st.subheader("📌 Appendice..."))
        
    # ELIMINA COMPLETAMENTE IL BLOCCO: "CICLO DI AGGIORNAMENTO (IL MOTORE)"
    # Non abbiamo più bisogno di time.sleep né di st.rerun() in fondo al file!
    
        # Mostra il tempo
        info_container = st.container(border=True)
        with info_container:
            st.markdown(f"<h3 style='text-align: center; color: white; margin: 0;'>Tempo Simulato: {st.session_state.sim_time:.1f} s</h3>", unsafe_allow_html=True)
        
        # --- Sezione Appendice Personale (Statica) ---
        st.divider()
        st.subheader("📌 Appendice: Note e Considerazioni sul Progetto")
        
        # Qui puoi inserire tutto il testo, le liste o le presentazioni che desideri
        st.markdown("""
        **Presentazione del Lavoro:**
        Questo ambiente di simulazione orbitale è stato sviluppato con l'obiettivo di analizzare l'evoluzione 
        dinamica delle orbite terrestri. Il core matematico è costruito su propagatori numerici ad alta precisione 
        per integrare le *Equazioni Planetarie di Lagrange*.

        **Considerazioni sulla Perturbazione J2:**
        - L'asfericità del corpo centrale induce variazioni secolari sugli elementi orbitali.
        - L'effetto è particolarmente visibile sulla precessione dell'anomalia del nodo ascendente ($\Omega$) 
          e sull'argomento del pericentro ($\omega$).
        - Per le orbite LEO equatoriali o fortemente eccentriche, questi effetti dettano i requisiti di 
          station-keeping della missione.
          - La simulazione pre-calcola l'evoluzione orbitale con J2 per 100 orbite, consentendo un confronto 
            diretto tra il comportamento perturbato e quello ideale.
        """)

    # ================== CICLO DI AGGIORNAMENTO (IL MOTORE) ==================
    if st.session_state.running:
        if use_perturbation and st.session_state.mode == 'Tempo' and st.session_state.get('precomputed_sol') is not None:
            compute_step_from_precomputed(dt_slider=dt) 
        elif not use_perturbation:
            if st.session_state.mode == 'Anomalia': compute_orbit_step(delta_M=delta_M)
            else: compute_orbit_step(dt=dt)
            
        time.sleep(0.1) 
        st.rerun()

if __name__ == '__main__':
    main()
