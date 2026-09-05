import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import math
import orbital_lib as ol
import importlib
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Forza la ricarica della libreria
importlib.reload(ol)

# ---------------------- Configurazione Pagina ----------------------
st.set_page_config(
    page_title='AquinSpace',
    layout='wide',
    initial_sidebar_state="collapsed"
)

# ---------------------- Costanti & Corpi Celesti ----------------------
# I parametri (mu, raggio di riferimento, velocità di rotazione) vengono da
# orbital_lib, cosi' il sito usa sempre le stesse costanti del propagatore.
# I file originali EGM2008.gfc (~1-5 MB a seconda del grado) e soprattutto
# GRAIL (fino a ~200 MB, oltre il limite di 100MB/file di GitHub) NON vanno
# nella repo del sito: si usano solo una volta in locale, con
# truncate_coefficients.py, per generare due CSV già denormalizzati e
# troncati al grado massimo che l'interfaccia offre (MAX_SUPPORTED_DEGREE) -
# pochi KB/decine di KB invece di centinaia di MB. Vedi truncate_coefficients.py.
MAX_SUPPORTED_DEGREE = 50

BODY_PARAMS = {
    'Terra': dict(mu=ol.EARTH_MU, R_ref=ol.EARTH_R_REF, w_body=ol.EARTH_ANGULAR_VELOCITY,
                  coeffs_file=DATA_DIR / "egm2008_coeffs.csv", loader=ol.load_gravity_coefficients_csv),
    'Luna':  dict(mu=ol.MOON_MU, R_ref=ol.MOON_R_REF, w_body=ol.MOON_ANGULAR_VELOCITY,
                  coeffs_file=DATA_DIR / "grail_coeffs.csv", loader=ol.load_gravity_coefficients_csv),
}

MAX_TRAJ_POINTS = 6000
TRIM_TRAJ_TO = 4000
MAX_TABLE_ROWS = 6000
TRIM_TABLE_TO = 4000

EARTH_PRESETS = {
    'LEO 400km equatoriale': {'alt': 400.0, 'incl': 0.0},
    'LEO 800km polare':      {'alt': 800.0, 'incl': 90.0},
    'GTO':                   {'perigee_alt': 250.0, 'apogee_alt': 35786.0, 'incl': 0.0},
    'GEO':                   {'alt': 35786.0, 'incl': 0.0}
}

# Scenari Luna: presi direttamente dagli stessi casi di studio usati per
# validare il propagatore Cowell+Pines in MATLAB (r, v in km, km/s)
MOON_PRESETS = {
    'Inclinazione 45 deg':  {'r': [1798.0, 0.0, 0.0],  'v': [0.0, 1.1682, 1.1682]},
    'LRO Polare (50km)':    {'r': [1788.0, 0.0, 0.0],  'v': [0.0, 0.0, 1.6556]},
    'Apollo 11 (110km)':    {'r': [1848.0, 0.0, 0.0],  'v': [0.0, 1.6285, 0.0]},
    'Frozen Orbit':         {'r': [0.0, 125.1, 1789.0], 'v': [-1.6934, 0.0, 0.0]},
}

# ---------------------- Funzioni di Stato e Reset ----------------------
# NOTA IMPORTANTE sull'architettura dello stato: i campi numerici cartesiani
# (in_sx..in_vz) e kepleriani (in_kep_a..in_kep_M) sono le UNICHE fonti di
# verità per i rispettivi valori: una volta che una key esiste in
# st.session_state, Streamlit ignora il parametro value= di un widget con
# la stessa key e mostra sempre il valore in session_state. Per questo,
# preset e cambio di corpo/modalità NON scrivono solo su 'position'/
# 'velocity' (che sono un semplice specchio di comodo), ma direttamente
# sulle key dei widget, PRIMA che vengano istanziati nel run successivo.

def ensure_state():
    """Inizializza le variabili di stato se non esistono."""
    defaults = {
        'running': False, 'trajectory': [], 'sim_time': 0.0,
        'body': 'Terra', '_last_body': 'Terra',
        'input_mode': 'Cartesiane (r, v)', '_last_input_mode': 'Cartesiane (r, v)',
        'orbital_data_df': None, 'trajectory_data': None, 'body_mesh': {},
        'cont_state': None, 'cont_t': 0.0, 'cont_rows': [], 'cont_params': None,
        'position': [7000.0, 0.0, 0.0], 'velocity': [0.0, 7.5, 0.0],
        'in_sx': 7000.0, 'in_sy': 0.0, 'in_sz': 0.0,
        'in_vx': 0.0, 'in_vy': 7.5, 'in_vz': 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if 'in_kep_a' not in st.session_state:
        _sync_kep_from_cartesian(ol.EARTH_MU)


def _sync_kep_from_cartesian(mu):
    """Aggiorna i campi 'Elementi Kepleriani' a partire dallo stato cartesiano corrente."""
    try:
        r = [st.session_state.in_sx, st.session_state.in_sy, st.session_state.in_sz]
        v = [st.session_state.in_vx, st.session_state.in_vy, st.session_state.in_vz]
        a, ecc, I, w, Omega, M, _n, _tperi = ol.car2kep(r, v, mu)
        st.session_state.in_kep_a = float(a)
        st.session_state.in_kep_e = float(ecc)
        st.session_state.in_kep_i = float(np.degrees(I))
        st.session_state.in_kep_w = float(np.degrees(w)) % 360.0
        st.session_state.in_kep_Om = float(np.degrees(Omega)) % 360.0
        st.session_state.in_kep_M = float(np.degrees(M)) % 360.0
    except Exception:
        # orbita degenere (es. r=0): imposta valori di comodo senza far fallire il rerun
        st.session_state.setdefault('in_kep_a', 7000.0)
        st.session_state.setdefault('in_kep_e', 0.0)
        st.session_state.setdefault('in_kep_i', 0.0)
        st.session_state.setdefault('in_kep_w', 0.0)
        st.session_state.setdefault('in_kep_Om', 0.0)
        st.session_state.setdefault('in_kep_M', 0.0)


def _sync_cartesian_from_kep(mu):
    """Aggiorna i campi cartesiani a partire dagli elementi kepleriani correnti."""
    try:
        a, e = st.session_state.in_kep_a, st.session_state.in_kep_e
        i = np.radians(st.session_state.in_kep_i)
        w = np.radians(st.session_state.in_kep_w)
        Om = np.radians(st.session_state.in_kep_Om)
        M = np.radians(st.session_state.in_kep_M)
        r, v = ol.kep2car(a, e, i, w, Om, M, mu)
        st.session_state.in_sx, st.session_state.in_sy, st.session_state.in_sz = [float(c) for c in r]
        st.session_state.in_vx, st.session_state.in_vy, st.session_state.in_vz = [float(c) for c in v]
    except Exception:
        pass


def _set_cartesian_state(r, v):
    st.session_state.in_sx, st.session_state.in_sy, st.session_state.in_sz = [float(c) for c in r]
    st.session_state.in_vx, st.session_state.in_vy, st.session_state.in_vz = [float(c) for c in v]
    st.session_state.position = [float(c) for c in r]
    st.session_state.velocity = [float(c) for c in v]
    st.session_state.trajectory = [list(st.session_state.position)]


def apply_preset(name, body, mu, R_ref):
    """Applica i valori del preset allo stato (posizione/velocità cartesiane)."""
    if body == 'Terra':
        if name not in EARTH_PRESETS:
            return
        data = EARTH_PRESETS[name]
        if name == 'GEO':
            r_mag = R_ref + data['alt']
            v_mag = np.sqrt(mu / r_mag)
            r, v = [r_mag, 0.0, 0.0], [0.0, v_mag, 0.0]
        elif 'perigee_alt' in data:  # GTO
            rp = R_ref + data['perigee_alt']
            ra = R_ref + data['apogee_alt']
            a = (rp + ra) / 2
            vp = np.sqrt(mu * (2 / rp - 1 / a))
            r, v = [rp, 0.0, 0.0], [0.0, vp, 0.0]
        else:  # LEO
            r_mag = R_ref + data['alt']
            v_mag = np.sqrt(mu / r_mag)
            incl_rad = np.radians(data['incl'])
            r, v = [r_mag, 0.0, 0.0], [0.0, v_mag * np.cos(incl_rad), v_mag * np.sin(incl_rad)]
    else:
        if name not in MOON_PRESETS:
            return
        data = MOON_PRESETS[name]
        r, v = list(data['r']), list(data['v'])

    _set_cartesian_state(r, v)
    _sync_kep_from_cartesian(mu)


def get_initial_state(mu):
    """Restituisce (r0, v0) correnti in base alla modalità di inserimento scelta."""
    if st.session_state.input_mode == 'Cartesiane (r, v)':
        r0 = np.array(st.session_state.position, dtype=float)
        v0 = np.array(st.session_state.velocity, dtype=float)
    else:
        a = st.session_state.in_kep_a
        e = st.session_state.in_kep_e
        i = np.radians(st.session_state.in_kep_i)
        w = np.radians(st.session_state.in_kep_w)
        Om = np.radians(st.session_state.in_kep_Om)
        M = np.radians(st.session_state.in_kep_M)
        r0, v0 = ol.kep2car(a, e, i, w, Om, M, mu)
    return r0, v0


# ---------------------- Coefficienti Campo Gravitazionale ----------------------
@st.cache_resource(show_spinner=False)
def get_gravity_matrices(body, degree):
    """Carica (una sola volta per corpo/grado) e converte in matrici i coefficienti."""
    params = BODY_PARAMS[body]
    coeffs_table = params['loader'](params['coeffs_file'], degree)
    return ol.build_coeffs_matrix(coeffs_table, degree)


def get_perturbation_matrices(body, degree, use_perturbation):
    """Restituisce (Cmat, Smat, degree_effettivo) - matrici nulle se la perturbazione è disattivata."""
    if not use_perturbation:
        return np.zeros((3, 3)), np.zeros((3, 3)), 2
    return get_gravity_matrices(body, degree) + (degree,)


# ---------------------- Funzioni di Grafica ----------------------
def generate_body_mesh(body, radius):
    if body in st.session_state.body_mesh:
        return st.session_state.body_mesh[body]

    phi = np.linspace(0, np.pi, 45)
    theta = np.linspace(0, 2 * np.pi, 90)

    th_grid, ph_grid = np.meshgrid(theta, phi)
    x = radius * np.sin(ph_grid) * np.cos(th_grid)
    y = radius * np.sin(ph_grid) * np.sin(th_grid)
    z = radius * np.cos(ph_grid)
    shade = np.clip(np.sin(ph_grid) * np.cos(th_grid) * 0.8 + np.cos(ph_grid) * 0.4, 0, 1)
    st.session_state.body_mesh[body] = (x, y, z, shade)
    return x, y, z, shade


def _base_figure(body, radius, max_range, colorscale):
    x, y, z, shade = generate_body_mesh(body, radius)
    fig = go.Figure()

    fig.add_scatter3d(x=[-max_range, max_range], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=4), hoverinfo='none')
    fig.add_scatter3d(x=[0, 0], y=[-max_range, max_range], z=[0, 0], mode='lines', line=dict(color='green', width=4), hoverinfo='none')
    fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[-max_range, max_range], mode='lines', line=dict(color='blue', width=4), hoverinfo='none')

    fig.add_surface(x=x, y=y, z=z, surfacecolor=shade, colorscale=colorscale, showscale=False, opacity=0.95, hoverinfo='none')

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


def build_preview_figure(r0, v0, mu, body, radius, max_range):
    """Anteprima statica (baseline kepleriana non perturbata) dell'orbita per lo stato iniziale corrente."""
    fig = _base_figure(body, radius, max_range, 'Blues' if body == 'Terra' else 'Greys')

    try:
        kep_el = ol.car2kep(r0, v0, mu)
        x_h, y_h, z_h, *_ = ol.osculating_orbit(kep_el, mu)
        fig.add_scatter3d(x=x_h, y=y_h, z=z_h, mode='lines', line=dict(color='cyan', width=3), name='Anteprima Orbita')
    except Exception:
        pass  # orbita non definibile con i parametri correnti (es. iperbolica): mostra solo il punto

    fig.add_scatter3d(x=[r0[0]], y=[r0[1]], z=[r0[2]], mode='markers',
                       marker=dict(size=6, color='orange', symbol='diamond'), name='Posizione Iniziale')
    return fig


def build_running_figure(trajectory, position, body, radius, max_range):
    """Grafico statico (aggiornato ad ogni passo) per la propagazione 'ad oltranza'."""
    fig = _base_figure(body, radius, max_range, 'Blues' if body == 'Terra' else 'Greys')

    px, py, pz = position
    fig.add_scatter3d(x=[px], y=[py], z=[pz], mode='markers', marker=dict(size=6, color='orange', symbol='diamond'), name='Satellite')

    if len(trajectory) > 1:
        traj = np.array(trajectory)
        fig.add_scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode='lines', line=dict(color='yellow', width=3), name='Traiettoria')

    return fig


def build_animated_figure(x_hist, y_hist, z_hist, body, radius, max_range):
    """
    Crea un grafico Plotly 3D con la traiettoria completa e un'animazione
    fluida del satellite che la percorre.
    """
    fig = _base_figure(body, radius, max_range, 'Blues' if body == 'Terra' else 'Greys')

    fig.add_scatter3d(x=x_hist, y=y_hist, z=z_hist, mode='lines', line=dict(color='yellow', width=2), name='Traiettoria', hoverinfo='none')
    fig.add_scatter3d(x=[x_hist[0]], y=[y_hist[0]], z=[z_hist[0]], mode='markers', marker=dict(size=6, color='orange', symbol='diamond'), name='Satellite')

    num_frames = min(len(x_hist), 500)
    step = max(1, len(x_hist) // num_frames)

    frames = []
    for i in range(0, len(x_hist), step):
        frame = go.Frame(
            data=[go.Scatter3d(x=[x_hist[i]], y=[y_hist[i]], z=[z_hist[i]])],
            traces=[5]  # indice della traccia del satellite: assi=0,1,2, terra(surface)=3, traiettoria=4, satellite=5
        )
        frames.append(frame)

    fig.frames = frames

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=1.0,
            buttons=[
                dict(label="▶ Play Animazione",
                     method="animate",
                     args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pausa",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )]
    )
    return fig


# ================== MAIN APPLICATION ==================
def main():
    ensure_state()

    st.markdown("""
        <style>
        .stApp { background-color: #000000; }
        .stApp, h1, h2, h3, h4, h5, h6, p, label, span { color: #FFFFFF !important; }
        [data-testid="stExpander"] { background-color: #111111; border-color: #333333; }
        button[kind="primary"] {
            background-color: #ff4b4b !important;
            color: white !important;
            border: none;
            transition: 0.2s;
        }
        button[kind="primary"]:hover { background-color: #ff3333 !important; }
        button[kind="primary"]:active { background-color: #ffffff !important; color: #ff4b4b !important; }
        </style>
        """, unsafe_allow_html=True)

    col_logo, col_header_title = st.columns([1.5, 2.5], gap="large")
    with col_logo:
        try:
            st.image(str(BASE_DIR / "logo_scritta.jpg"), use_container_width=True)
        except Exception:
            st.warning("Logo non trovato")
    with col_header_title:
        st.markdown("<h1 style='font-size: 5rem; font-weight: 800; margin-top: 40px;'>AquinSpace</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color: #1a1a1a; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-top: 10px;'>
            <strong>Benvenuto In AquinSpace.</strong><br>
            Questo strumento simula la dinamica dei satelliti attorno alla Terra o alla Luna, integrando
            direttamente le equazioni del moto (formulazione di Cowell) con l'accelerazione perturbativa
            dovuta all'asfericità del corpo centrale calcolata con la formulazione di Pines - valida anche
            per orbite polari, senza singolarità. Imposta lo stato iniziale (in coordinate cartesiane o
            con gli elementi kepleriani), scegli corpo celeste e grado del campo gravitazionale, poi premi START.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    left_panel, right_panel = st.columns([1, 2], gap="medium")

    # ================== PANNELLO SINISTRO (Controlli) ==================
    with left_panel:
        st.subheader("Pannello di Controllo")

        with st.expander("1. Corpo Celeste e Stato Iniziale", expanded=True):
            body = st.radio('Corpo Celeste', ['Terra', 'Luna'],
                             index=0 if st.session_state.body == 'Terra' else 1, horizontal=True)
            mu = BODY_PARAMS[body]['mu']
            R_ref = BODY_PARAMS[body]['R_ref']

            # Cambio di corpo celeste: i valori cartesiani correnti (pensati per l'altro
            # corpo) non hanno più senso fisico (es. velocità da orbita terrestre attorno
            # alla Luna sarebbero iperboliche) - si applica automaticamente il primo
            # preset del nuovo corpo.
            if body != st.session_state._last_body:
                default_preset = next(iter(EARTH_PRESETS if body == 'Terra' else MOON_PRESETS))
                apply_preset(default_preset, body, mu, R_ref)
                st.session_state._last_body = body
            st.session_state.body = body

            presets = EARTH_PRESETS if body == 'Terra' else MOON_PRESETS
            preset_choice = st.selectbox('Scegli un Preset', ['(Personalizzato)'] + list(presets.keys()), key=f'preset_{body}')
            if preset_choice != '(Personalizzato)':
                if st.button(f'Applica {preset_choice}', type="primary", use_container_width=True):
                    apply_preset(preset_choice, body, mu, R_ref)
                    st.rerun()

            input_mode = st.radio('Modalità di inserimento',
                                   ['Cartesiane (r, v)', 'Elementi Kepleriani'],
                                   horizontal=True)

            # Cambio di modalità di inserimento: sincronizza i campi dell'altra
            # rappresentazione, cosi' i due modi restano sempre coerenti tra loro.
            if input_mode != st.session_state._last_input_mode:
                if input_mode == 'Elementi Kepleriani':
                    _sync_kep_from_cartesian(mu)
                else:
                    _sync_cartesian_from_kep(mu)
                st.session_state._last_input_mode = input_mode
            st.session_state.input_mode = input_mode

            if input_mode == 'Cartesiane (r, v)':
                col_pos, col_vel = st.columns(2)
                with col_pos:
                    st.markdown("**Posizione (km)**")
                    st.number_input('X', format="%.1f", key="in_sx")
                    st.number_input('Y', format="%.1f", key="in_sy")
                    st.number_input('Z', format="%.1f", key="in_sz")
                with col_vel:
                    st.markdown("**Velocità (km/s)**")
                    st.number_input('VX', format="%.3f", key="in_vx")
                    st.number_input('VY', format="%.3f", key="in_vy")
                    st.number_input('VZ', format="%.3f", key="in_vz")
                st.session_state.position = [st.session_state.in_sx, st.session_state.in_sy, st.session_state.in_sz]
                st.session_state.velocity = [st.session_state.in_vx, st.session_state.in_vy, st.session_state.in_vz]
            else:
                col_k1, col_k2 = st.columns(2)
                with col_k1:
                    st.number_input('Semiasse a (km)', min_value=1.0, format="%.2f", key="in_kep_a")
                    st.number_input('Eccentricità e', min_value=0.0, max_value=0.999, format="%.5f", key="in_kep_e")
                    st.number_input('Inclinazione i (deg)', min_value=0.0, max_value=180.0, format="%.3f", key="in_kep_i")
                with col_k2:
                    st.number_input('Arg. Pericentro w (deg)', format="%.3f", key="in_kep_w")
                    st.number_input('RAAN Omega (deg)', format="%.3f", key="in_kep_Om")
                    st.number_input('Anomalia Media M (deg)', format="%.3f", key="in_kep_M")

        with st.expander("2. Parametri Simulazione", expanded=True):
            use_perturbation = st.toggle('Attiva Perturbazione (campo gravitazionale reale)', value=False)
            degree = 2
            if use_perturbation:
                degree = st.slider('Grado del potenziale', min_value=2, max_value=MAX_SUPPORTED_DEGREE, value=10, step=1,
                                    help=f"Gradi più alti = maggiore fedeltà fisica ma integrazione più lenta. "
                                         f"I file dati inclusi nel sito arrivano al grado {MAX_SUPPORTED_DEGREE}.")

            st.markdown("---")
            prop_mode = st.radio('Modalità di propagazione', ['Numero di Orbite', 'Ad oltranza (fino a STOP)'])

            n_orbite, num_points, dt_step, n_substeps = 5, 3000, 60.0, 5
            if prop_mode == 'Numero di Orbite':
                n_orbite = st.number_input('Numero di Orbite da propagare', min_value=1, max_value=2000, value=5, step=1)
                num_points = st.number_input('Punti campionati', min_value=200, max_value=20000, value=3000, step=100)
            else:
                st.info("La propagazione prosegue indefinitamente, un intervallo di tempo per volta, "
                        "finché non premi STOP. Il grafico e la traiettoria mostrano solo gli ultimi punti calcolati.")
                dt_step = st.slider('Passo di Tempo per Aggiornamento (s)', 10.0, 5000.0, 100.0, step=10.0)
                n_substeps = st.slider('Sotto-campioni per Aggiornamento', 1, 20, 5,
                                        help="Punti intermedi campionati in ogni passo, per una traiettoria più liscia.")

        with st.expander("3. Vista e Azioni", expanded=True):
            max_range = st.number_input('Scala Visuale (km)', value=float(int(R_ref * 1.3)), min_value=float(R_ref), step=1000.0)

            st.markdown("##### Comandi")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            start_pressed = btn_col1.button('▶️ START', type="primary", use_container_width=True)
            stop_pressed = btn_col2.button('⏸️ STOP', type="primary", use_container_width=True)
            reset_trace_pressed = btn_col3.button('🔄 TRACCIA', type="primary", use_container_width=True)

        with st.expander("4. Dati Orbitali", expanded=True):
            st.markdown("Visualizza ed esporta i parametri orbitali propagati.")
            if st.session_state.orbital_data_df is not None:
                st.dataframe(st.session_state.orbital_data_df, height=200, use_container_width=True)
                csv_data = st.session_state.orbital_data_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Scarica Dati Orbitali (CSV)",
                    data=csv_data,
                    file_name='dati_orbitali.csv',
                    mime='text/csv',
                    type='primary',
                    use_container_width=True
                )
            else:
                st.info("Premi START per calcolare l'orbita e visualizzare i dati.")

    # ================== PANNELLO DESTRO (Visualizzazione) ==================
    with right_panel:
        r0, v0 = get_initial_state(mu)

        if stop_pressed:
            st.session_state.running = False

        if reset_trace_pressed:
            st.session_state.trajectory_data = None
            st.session_state.orbital_data_df = None
            st.session_state.cont_state = None
            st.session_state.cont_rows = []
            st.session_state.running = False

        if start_pressed:
            st.session_state.trajectory_data = None
            st.session_state.orbital_data_df = None
            st.session_state.cont_state = None
            st.session_state.cont_rows = []
            try:
                Cmat, Smat = get_perturbation_matrices(body, degree, use_perturbation)[0:2]
                degree_eff = degree if use_perturbation else 2
                w_body = BODY_PARAMS[body]['w_body']

                if prop_mode == 'Numero di Orbite':
                    with st.spinner("🚀 Propagazione dell'orbita in corso (Cowell + Pines)... (attendere)"):
                        kep_el = ol.car2kep(r0, v0, mu)
                        x_h, y_h, z_h, vx_h, vy_h, vz_h, t_sol, els = ol.propagate_orbit_cowell_pines(
                            kep_el, mu, R_ref, w_body, Cmat, Smat, degree_eff,
                            num_orbits=int(n_orbite), num_points=int(num_points)
                        )
                        st.session_state.trajectory_data = (x_h, y_h, z_h)
                        st.session_state.orbital_data_df = pd.DataFrame({
                            'Tempo (s)': t_sol,
                            'Semiasse a (km)': els[0],
                            'Eccentricità e': els[1],
                            'Inclinazione i (rad)': els[2],
                            'Arg. Pericentro w (rad)': els[3],
                            'RAAN Omega (rad)': els[4],
                            'Anomalia Media M (rad)': els[5],
                        })
                else:
                    st.session_state.cont_state = np.concatenate([r0, v0])
                    st.session_state.cont_t = 0.0
                    st.session_state.trajectory = [list(r0)]
                    st.session_state.cont_rows = []
                    st.session_state.cont_params = dict(Cmat=Cmat, Smat=Smat, degree=degree_eff, mu=mu,
                                                         R_ref=R_ref, w_body=w_body, dt=dt_step, n_substeps=int(n_substeps))
                    st.session_state.running = True
            except FileNotFoundError as ex:
                st.error(f"File dei coefficienti del campo gravitazionale non trovato: {ex}. "
                         f"Genera egm2008_coeffs.csv / grail_coeffs.csv con truncate_coefficients.py "
                         f"e mettili nella cartella 'data' accanto a interfaccia.py.")
            except Exception as ex:
                st.error(f"Errore nel calcolo orbitale: {ex}")

        # --- Passo di propagazione continua ("ad oltranza") ---
        if st.session_state.running and st.session_state.cont_state is not None:
            p = st.session_state.cont_params
            try:
                new_state, t_eval, r_hist, v_hist = ol.step_cowell_pines(
                    st.session_state.cont_state, st.session_state.cont_t, p['dt'],
                    p['Cmat'], p['Smat'], p['degree'], p['mu'], p['R_ref'], p['w_body'],
                    n_substeps=p['n_substeps']
                )
                for k in range(len(t_eval)):
                    st.session_state.trajectory.append(r_hist[k].tolist())
                    a_k, ecc_k, I_k, w_k, Omega_k, M_k, _n_k, _tp_k = ol.car2kep(r_hist[k], v_hist[k], p['mu'])
                    st.session_state.cont_rows.append({
                        'Tempo (s)': t_eval[k], 'Semiasse a (km)': a_k, 'Eccentricità e': ecc_k,
                        'Inclinazione i (rad)': I_k, 'Arg. Pericentro w (rad)': w_k,
                        'RAAN Omega (rad)': Omega_k, 'Anomalia Media M (rad)': M_k,
                    })

                if len(st.session_state.trajectory) > MAX_TRAJ_POINTS:
                    st.session_state.trajectory = st.session_state.trajectory[-TRIM_TRAJ_TO:]
                if len(st.session_state.cont_rows) > MAX_TABLE_ROWS:
                    st.session_state.cont_rows = st.session_state.cont_rows[-TRIM_TABLE_TO:]

                st.session_state.cont_state = new_state
                st.session_state.cont_t += p['dt']
                st.session_state.position = new_state[0:3].tolist()
                st.session_state.velocity = new_state[3:6].tolist()
                st.session_state.sim_time = st.session_state.cont_t
                st.session_state.orbital_data_df = pd.DataFrame(st.session_state.cont_rows)
            except Exception as ex:
                st.session_state.running = False
                st.error(f"Errore nella propagazione: {ex}")

        # --- Visualizzazione ---
        if st.session_state.trajectory_data is not None:
            x_h, y_h, z_h = st.session_state.trajectory_data
            fig = build_animated_figure(x_h, y_h, z_h, body, R_ref, max_range)
        elif st.session_state.cont_state is not None:
            fig = build_running_figure(st.session_state.trajectory, st.session_state.position, body, R_ref, max_range)
        else:
            fig = build_preview_figure(r0, v0, mu, body, R_ref, max_range)
            st.caption("Anteprima dell'orbita kepleriana non perturbata per lo stato iniziale corrente. Premi START per propagare.")

        st.plotly_chart(fig, use_container_width=True, key="grafico_3d_main")

        info_container = st.container(border=True)
        with info_container:
            st.markdown(f"<h3 style='text-align: center; color: white; margin: 0;'>Tempo Simulato: {st.session_state.sim_time:.1f} s</h3>", unsafe_allow_html=True)

        st.divider()
        st.subheader("📌 Appendice: Note e Considerazioni sul Progetto")
        st.markdown("""
        **Presentazione del Lavoro:**
        Questo ambiente di simulazione orbitale integra direttamente le equazioni del moto attorno alla
        Terra o alla Luna (formulazione di Cowell), sommando all'attrazione kepleriana l'accelerazione
        perturbativa dovuta all'asfericità del corpo centrale, calcolata con la formulazione di Pines
        (coseni direttori, priva di singolarità ai poli - adatta anche a orbite polari).

        **Modalità di propagazione:**
        - *Numero di Orbite*: l'intera traiettoria viene precalcolata e poi animata.
        - *Ad oltranza*: la propagazione prosegue indefinitamente, un intervallo alla volta, finché non
          si preme STOP - possibile grazie all'efficienza del propagatore numerico.

        **Corpo Celeste e Campo Gravitazionale:**
        - *Terra*: coefficienti EGM2008.
        - *Luna*: coefficienti GRGM900C (GRAIL).
        - Il grado del potenziale è impostabile: gradi più alti aumentano la fedeltà fisica a scapito
          del tempo di caricamento/integrazione.
        """)

    # ================== MOTORE DI AGGIORNAMENTO (solo modalità "Ad oltranza") ==================
    if st.session_state.running and st.session_state.cont_state is not None:
        time.sleep(0.05)
        st.rerun()


if __name__ == '__main__':
    main()
