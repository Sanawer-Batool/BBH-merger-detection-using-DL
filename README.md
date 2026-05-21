# O3 BBH Gravitational Wave Preprocessing Pipeline
==================================================
Based on: "A Machine-Learning Pipeline for Real-Time Detection of
Gravitational Waves from Compact Binary Coalescences" (arXiv:2403.18661)
 
How to run
----------
  pip install gwpy gwosc numpy torch scipy matplotlib
  python o3_bbh_preprocessing_v2.py
 
  Optional flags:
  --event      GW190521        (any GWTC-3 BBH name)
  --duration   16              (seconds around event centre)
  --plot                       (save a visualisation PNG)
