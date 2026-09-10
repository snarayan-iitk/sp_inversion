### Quick User Guide

* **Install Python** and required packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `emcee`, `corner`, and `openpyxl`.
* **Prepare the SP data** in Excel/CSV format with station/easting/distance and SP values.
* **Set the input file path** in `FILE_PATH`.
* **Set the output folder** in `OUTPUT_DIR`.
* **Check inversion settings**, especially:

  * `MAX_SOURCES`
  * `MCMC_WALKERS`
  * `MCMC_STEPS`
  * `MCMC_BURN_IN`
  * `NOISE_FRACTION`
* **Enter borehole information**, if available, using `BOREHOLES_AVAILABLE` and `BOREHOLE_DEPTH`.
* **Select geological type** (`unknown`, `massive_sulfide`, or `graphite`) only when supported by independent geological information.
* **Run the Python script**.
* When asked about synthetic testing, select **`y`** for validation or **`n`** for direct processing of real data.
* The program **removes the background trend**, estimates the anomaly, and performs global optimization followed by **Bayesian MCMC inversion**.
* Check the estimated **X position, depth (Z), angle (α), and shape factor (q)**.
* Check **R², R-hat, ESS, trace plots, and posterior distributions** to assess the reliability of the inversion.
* Examine the **90% credible intervals** to understand parameter uncertainty.
* Use the saved **Inversion, Convergence, and Corner plots** for interpretation and reporting.
* Finally, **validate the interpreted source with geological/geophysical information** before drawing conclusion 
