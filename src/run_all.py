"""Reproduce everything:  python src/run_all.py   (~15 min on a 2-core laptop)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import data, rhythm, roles, network, figures
from common import DATA

t0 = time.time()
if not os.path.exists(os.path.join(DATA, "tweets.parquet")):
    print("[1/5] building dataset");  data.build()
print("[2/5] rhythm forensics");      rhythm.run()
print("[3/5] role recovery");         roles.run()
print("[4/5] coordination network");  network.run()
print("[5/5] figures")
for f in (figures.fig1_waterfall, figures.fig2_clock, figures.fig3_dst, figures.fig4_holidays,
          figures.fig5_roles, figures.fig6_network):
    f()
print(f"done in {(time.time() - t0) / 60:.1f} min")
