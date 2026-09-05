"""
truncate_coefficients.py
=========================

Script da eseguire UNA SOLA VOLTA in locale (dove hai i file originali
EGM2008.gfc e GRAIL), per generare due CSV piccoli e già pronti per il sito:

    data/egm2008_coeffs.csv
    data/grail_coeffs.csv

Il motivo: il file GRAIL originale (jggrx_1800f_me_sha.tab.txt) pesa circa
190 MB - oltre il limite di 100 MB per file di GitHub, e comunque molto più
di quanto serva: l'interfaccia non propone mai un grado del potenziale
superiore a MAX_DEGREE (in interfaccia.py), quindi tutti i coefficienti a
gradi più alti sarebbero inutilizzati. Troncando a MAX_DEGREE e salvando
solo (l, m, C, S) già denormalizzati si ottiene un CSV di poche decine di
KB, che sta comodamente in git senza bisogno di Git LFS.

USO:
    1. Metti questo script nella cartella dove hai EGM2008.gfc e il file
       GRAIL (es. jggrx_1800f_me_sha.tab.txt), oppure modifica i path qui
       sotto.
    2. Esegui:  python3 truncate_coefficients.py
    3. Copia la cartella "data" generata accanto a interfaccia.py nel
       progetto del sito (o sovrascrivi quella già presente).
    4. NON committare i file originali .gfc/.tab.txt nella repo del sito -
       bastano i due CSV in data/.
"""

from pathlib import Path
import sys

# Se orbital_lib.py non è nella stessa cartella di questo script, aggiusta il path:
sys.path.insert(0, str(Path(__file__).resolve().parent))
import orbital_lib as ol

# --- CONFIGURAZIONE: aggiusta questi path ai tuoi file originali ---
EGM2008_GFC = Path("EGM2008.gfc")                      # es. GRACE Earth/EGM2008.gfc
GRAIL_TAB = Path("jggrx_1800f_me_sha.tab.txt")         # es. GRAIL moon/jggrx_1800f_me_sha.tab.txt

# Deve combaciare con MAX_SUPPORTED_DEGREE in interfaccia.py
MAX_DEGREE = 50

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if EGM2008_GFC.exists():
        print(f"Lettura {EGM2008_GFC} (grado massimo {MAX_DEGREE})...")
        earth_table = ol.load_gravity_coefficients_grace(EGM2008_GFC, MAX_DEGREE)
        out_path = OUTPUT_DIR / "egm2008_coeffs.csv"
        ol.save_gravity_coefficients_csv(earth_table, out_path)
        print(f"  -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(earth_table)} righe)")
    else:
        print(f"ATTENZIONE: {EGM2008_GFC} non trovato, salto la Terra.")

    if GRAIL_TAB.exists():
        print(f"Lettura {GRAIL_TAB} (grado massimo {MAX_DEGREE}) - può richiedere qualche secondo...")
        moon_table = ol.load_gravity_coefficients_grail(GRAIL_TAB, MAX_DEGREE)
        out_path = OUTPUT_DIR / "grail_coeffs.csv"
        ol.save_gravity_coefficients_csv(moon_table, out_path)
        print(f"  -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(moon_table)} righe)")
    else:
        print(f"ATTENZIONE: {GRAIL_TAB} non trovato, salto la Luna.")

    print("\nFatto. Copia la cartella 'data' accanto a interfaccia.py nel progetto del sito.")


if __name__ == '__main__':
    main()
