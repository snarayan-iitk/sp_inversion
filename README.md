# sp_inversion
This package performs Bayesian inversion of self‑potential (SP) anomalies to estimate the depth, geometry, and source parameters of subsurface bodies (spheres, cylinders, or dipping sheets). It uses the generalised SP anomaly model of Abdelrahman et al. (2006) and combines Differential Evolution for global optimisation with Markov Chain Monte Carlo (MCMC) for uncertainty quantification.

Features
Synthetic Fidelity Tests – verify the inversion on cylinder, sphere, and sheet models with known parameters (including corner plots).

Real‑Data Inversion – reads Excel/CSV files, automatically identifies SP and distance columns, detrends, and selects the optimal number of sources via BIC.

Enhanced Constraints – optional geological priors (massive sulphide, graphite) and borehole depth constraints.

MCMC Diagnostics – trace plots, Gelman‑Rubin R‑hat, effective sample size, correlation matrices, and posterior distributions (α, z, q).

Visualisation – publication‑ready plots: data fit with uncertainty bands, subsurface cross‑sections, corner plots, and convergence diagnostics.

Installation
Requirements
Python ≥ 3.7

Packages: numpy, pandas, matplotlib, scipy, emcee, corner, openpyxl (for Excel)

Install all dependencies using pip:

bash
pip install numpy pandas matplotlib scipy emcee corner openpyxl

All user‑adjustable settings are at the top of the script (USER CONFIGURATION section). Open the script in a text editor and modify:

Parameter	Description
FILE_PATH	Full path to your input Excel/CSV file.
OUTPUT_DIR	Directory where all plots and results will be saved.
MAX_SOURCES	Maximum number of overlapping bodies to test per profile.
MCMC_WALKERS, MCMC_STEPS, MCMC_BURN_IN	MCMC parameters (increase for more robust sampling).
CONFIDENCE_INTERVAL	Confidence level for uncertainty bands (e.g., 90).
GEOLOGY_TYPE	"massive_sulfide", "graphite", or "unknown" – constrains the shape factor q.
BOREHOLE_DEPTH	Known depth from borehole (in metres).
BOREHOLES_AVAILABLE	Set True to enable depth prior (penalises deviations from borehole depth).
Font sizes	TITLE_FONT, LABEL_FONT, TICK_FONT, LEGEND_FONT for plotting.
Note: The script automatically identifies SP and distance columns by common names (e.g., "PD Corrected Final SP (mV)", "Easting"). If your column names differ, update the identify_columns_robust function accordingly.

Running the Code
Simply execute the script:

bash
python your_script.py
You will be prompted:

text
Do you want to run Synthetic Fidelity Tests (Cylinder, Sphere & Sheet) with Corner Plots? (y/n):
Type y to run the synthetic tests first (recommended to validate the inversion).

Type n to skip directly to inverting your real data.

After the synthetic tests (or if skipped), the code reads your data file, processes each sheet/traverse, and performs inversion. Results (plots and summaries) are saved in OUTPUT_DIR.

Input Data Format
Excel (.xlsx) with multiple sheets, or CSV (single sheet).

Each sheet/table must contain:

A column with distance/easting (e.g., "Easting", "Station (X)").

A column with SP values (e.g., "PD Corrected Final SP (mV)", "SP (mV)").

Optional: a column grouping traverses (e.g., "Traverse (Y)"). If present, the code processes each traverse separately.

The code automatically removes linear trends and identifies the number of sources based on peak detection.

Outputs
For each processed profile, the following are generated in OUTPUT_DIR:

Main inversion plot

{profile_name}_Inversion.png – data fit with uncertainty bands (red), and a cross‑section showing inferred source(s) with depth, angle, and shape.

Convergence diagnostics

{profile_name}_Convergence.png – 9‑panel figure with R‑hat, ESS, correlation matrix, trace plots, and posterior histograms for z, α, q.

Corner plot (real data)

{profile_name}_Corner_alpha_z_q.png – pairwise posterior distributions for the key parameters of the first body.

Synthetic tests (if run)

Corner_Plot_Cylinder_alpha_z_q.png, etc. – corner plots for each synthetic model.

Synthetic_Cylinder_Sphere_Sheet_Tests.png – side‑by‑side comparison of data fits and subsurface reconstructions.

All plots are high‑resolution (300 dpi) and ready for publication.

Interpreting the Results
Shape factor (q)

q ≥ 1.3 → sphere/point source.

0.8 ≤ q < 1.3 → cylinder (horizontal).

q < 0.8 → dipping sheet.

Depth (z) – given in metres. The 90% credible interval is shown in the plots and printed in the console.

Angle (α) – polarisation angle (degrees), indicates the direction of polarisation.

R‑hat values – should be close to 1.0 (ideally < 1.1) for convergence. If not, increase MCMC_STEPS or MCMC_WALKERS.

Effective sample size (ESS) – should be > 100 for reliable parameter estimates.

Correlation matrix – high correlations (e.g., between K and z) indicate trade‑offs; this is typical and the MCMC should still provide valid uncertainties.

Customising for Your Data
Column mapping: If the automatic column identification fails, edit the identify_columns_robust function and add your exact column names to the lists.

Trend removal: The code uses a linear detrend. If your data has a more complex background, you can modify the remove_trend function (e.g., use polynomial or spline).

Geological constraints: Set GEOLOGY_TYPE to "massive_sulfide" or "graphite" to narrow the q bounds. If you have borehole information, set BOREHOLES_AVAILABLE = True and provide BOREHOLE_DEPTH – this adds a weak prior to guide depth estimation.

Troubleshooting
File not found – ensure FILE_PATH is correct and uses raw string (e.g., r"C:\path\to\file.xlsx").

Memory issues – reduce MCMC_WALKERS or MCMC_STEPS if running on a low‑resource machine.

Poor convergence – increase maxiter and popsize in the differential_evolution call (inside optimize_global). Also, increase MCMC_STEPS and MCMC_BURN_IN.

Depth overestimation – the script includes a diagnostic function that prints correlations and profile‑to‑depth ratios. If overestimation persists, check your profile length (should be several times the expected depth).


