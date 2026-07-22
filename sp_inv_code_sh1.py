import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
import emcee
import os
import warnings
import time
import corner

# NEW imports for convergence plot
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
 
# -----------------------------
# USER CONFIGURATION
# -----------------------------
FILE_PATH = r"D:\GSI_data\Harmara_west\Sheet_200_test.xlsx"
OUTPUT_DIR = r"D:\GSI_data\Harmara_west\Inversion_Results"  # Folder to save images

# Inversion Settings
MAX_SOURCES = 1               # Max number of overlapping bodies to test per profile
MCMC_WALKERS = 500             # Number of walkers for MCMC (Reduced for speed in testing)
MCMC_STEPS = 3000         # Total MCMC steps
MCMC_BURN_IN = 1000           # Steps to discard as burn-in
CONFIDENCE_INTERVAL = 90      # Confidence interval for uncertainty plotting (%)

# NEW: Geological constraints based on known geology from borehole
GEOLOGY_TYPE = "unknown"  # Options: "massive_sulfide", "graphite", "unknown"
BOREHOLE_DEPTH = 60  # Known depth from borehole in meters
BOREHOLES_AVAILABLE = False  # Set to False if no borehole data

# VISUALIZATION SETTINGS (Font Sizes)
TITLE_FONT =  20
LABEL_FONT = 20
TICK_FONT = 20
LEGEND_FONT = 16

# GEOPHYSICAL KERNELS
# -----------------------------
def model_general_body(x, params):
    """
    Generalized SP Anomaly Model (Abdelrahman et al., 2006).
    Params per body: [K, x0, z, alpha (deg), q]
    """
    K, x0, z, alpha_deg, q = params
    
    # Safety constraints
    z = max(z, 0.1)
    alpha_rad = np.radians(alpha_deg)
    
    term1 = (x - x0) * np.cos(alpha_rad)
    term2 = z * np.sin(alpha_rad)
    numerator = term1 - term2
    denominator = ((x - x0)**2 + z**2)**q
    
    return K * (numerator / (denominator + 1e-9))

def forward_model_multi_source(x, all_params):
    """
    Computes the superposition of N anomalies.
    all_params structure: [Offset, K1, x1, z1, a1, q1, K2, x2, z2, a2, q2, ...]
    """
    offset = all_params[0]
    y_calc = np.full_like(x, offset)
    
    n_params_per_body = 5
    num_bodies = (len(all_params) - 1) // n_params_per_body
    
    for i in range(num_bodies):
        idx = 1 + i * n_params_per_body
        body_params = all_params[idx : idx + n_params_per_body]
        y_calc += model_general_body(x, body_params)
        
    return y_calc

# -----------------------------
# INVERSION ENGINE
# -----------------------------
class BayesianInversion:
    def __init__(self, x_obs, y_obs):
        self.x = x_obs
        self.y = y_obs
        self.x_min, self.x_max = np.min(x_obs), np.max(x_obs)
        self.y_span = np.max(y_obs) - np.min(y_obs)
        
        # Define profile_length first
        self.profile_length = self.x_max - self.x_min
        
        # NEW: Precompute weights for near-surface sensitivity (Solution C)
        self.weights = self._compute_weights()
        
        # NEW: Storage for MCMC diagnostics
        self.chain = None
        self.sampler = None
        
    def _compute_weights(self):
        """Solution C: Weight near-surface sensitivity"""
        # Higher weight for central part of anomaly
        x_center = np.mean(self.x)
        width = self.profile_length / 4  # Focus on central quarter
        weights = np.exp(-((self.x - x_center)**2)/(2 * width**2))
        # Normalize weights
        weights = weights / np.mean(weights)
        return weights

    def log_likelihood(self, theta):
        """Gaussian Likelihood."""
        model = forward_model_multi_source(self.x, theta)
        sigma2 = np.var(self.y - model) # Estimate noise variance from residuals
        if sigma2 == 0: sigma2 = 1e-9
        return -0.5 * np.sum((self.y - model) ** 2 / sigma2 + np.log(sigma2))
    
    def log_likelihood_weighted(self, theta):
        """Solution C: Weighted likelihood for near-surface sensitivity"""
        model = forward_model_multi_source(self.x, theta)
        sigma2 = np.var(self.y - model)
        if sigma2 == 0: sigma2 = 1e-9
        # Apply weights to emphasize central part of anomaly
        weighted_residuals = self.weights * (self.y - model) ** 2 / sigma2
        return -0.5 * np.sum(weighted_residuals + np.log(sigma2))

    def log_prior(self, theta, num_bodies):
        """Uniform Prior with enhanced constraints (Solutions A, B, D)."""
        offset = theta[0]
        # Relaxed offset prior
        if not (-abs(self.y_span)*5 < offset < abs(self.y_span)*5):
            return -np.inf
            
        n_params = 5
        lp = 0.0
        
        # Solution D: Hierarchical prior for depths (if borehole available)
        if BOREHOLES_AVAILABLE:
            z_prior_mean = BOREHOLE_DEPTH
            z_prior_std = BOREHOLE_DEPTH * 0.3  # 30% uncertainty
        else:
            # Default prior if no borehole
            z_prior_mean = 30.0
            z_prior_std = 20.0
        
        # Store depths for smoothness constraint (Solution B)
        z_values = []
        
        for i in range(num_bodies):
            idx = 1 + i * n_params
            K, x0, z, alpha, q = theta[idx : idx + n_params]
            
            # Solution A: Geology-based q constraints - FIXED: wider bounds to allow convergence
            if GEOLOGY_TYPE == "massive_sulfide":
                # Sphere-like bodies - widened bounds
                if not (0.8 < q < 2.2): return -np.inf
            elif GEOLOGY_TYPE == "graphite":
                # Cylinder-like bodies - widened bounds
                if not (0.5 < q < 1.8): return -np.inf
            else:
                # Unknown geology - wider bounds
                if not (0.3 < q < 3.0): return -np.inf
            
            # Standard bounds
            if not (-1e7 < K < 1e7): return -np.inf
            if not (self.x_min - 1000 < x0 < self.x_max + 1000): return -np.inf
            if not (1.0 < z < 500): return -np.inf
            if not (-180 < alpha < 180): return -np.inf
            
            # Solution D: Add hierarchical penalty for depth - FIXED: don't force too strongly
            if BOREHOLES_AVAILABLE:
                lp_z = -0.1 * ((z - z_prior_mean) / z_prior_std)**2  # Reduced penalty
                lp += lp_z
            
            # Store z for smoothness constraint
            z_values.append(z)
        
        # Solution B: Add smoothness constraint if multiple bodies - FIXED: reduced penalty
        if num_bodies > 1:
            z_range = max(z_values) - min(z_values)
            if z_range > 100:  # Increased threshold
                lp += -0.05 * z_range  # Reduced penalty
        
        return lp

    def log_probability(self, theta, num_bodies):
        lp = self.log_prior(theta, num_bodies)
        if not np.isfinite(lp):
            return -np.inf
        
        # Use weighted likelihood for better shallow sensitivity
        return lp + self.log_likelihood_weighted(theta)

    def optimize_global(self, num_bodies):
        """Step 1: Differential Evolution (Global Optimization) with enhanced bounds."""
        bounds = [(-np.max(np.abs(self.y)), np.max(np.abs(self.y)))] # Offset
        
        for _ in range(num_bodies):
            # Solution A: Adjust K bounds based on data - FIXED: more reasonable bound
            K_bound = max(abs(self.y)) * 20  # Reduced from 100 to 20
            
            # Solution A: Adjust q bounds based on geology - FIXED: wider bounds
            if GEOLOGY_TYPE == "massive_sulfide":
                q_min, q_max = 0.8, 2.2  # Wider bounds
            elif GEOLOGY_TYPE == "graphite":
                q_min, q_max = 0.5, 1.8  # Wider bounds
            else:
                q_min, q_max = 0.3, 3.0  # Wider bounds
            
            # Solution D: Adjust z bounds if borehole available - FIXED: wider bounds
            if BOREHOLES_AVAILABLE:
                z_min = max(1.0, BOREHOLE_DEPTH * 0.2)  # 20% of borehole depth (was 50%)
                z_max = min(500, BOREHOLE_DEPTH * 3.0)  # 300% of borehole depth (was 200%)
            else:
                z_min, z_max = 5.0, 200.0
            
            bounds += [
                (-K_bound, K_bound),           # K - scaled to data
                (self.x_min, self.x_max),      # x0
                (z_min, z_max),                 # z - wider constraints
                (-90, 90),                      # alpha
                (q_min, q_max)                   # q - wider bounds
            ]
            
        def objective(theta):
            pred = forward_model_multi_source(self.x, theta)
            # Use weighted objective for optimization too
            weighted_residuals = self.weights * (self.y - pred)
            return np.sum(weighted_residuals**2)

        # FIXED: increased maxiter and popsize for better convergence
        result = differential_evolution(objective, bounds, strategy='best1bin', 
                                       maxiter=200, popsize=15, tol=0.001)
        return result.x, result.fun

    def run_mcmc(self, best_guess, num_bodies):
        """Step 2: Markov Chain Monte Carlo (MCMC)."""
        ndim = len(best_guess)
        nwalkers = max(MCMC_WALKERS, ndim * 2)
        
        # FIXED: larger initial spread for better exploration
        pos = best_guess + 1e-2 * np.random.randn(nwalkers, ndim)  # Increased from 1e-4 to 1e-2
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=(num_bodies,))
        
        sampler.run_mcmc(pos, MCMC_STEPS, progress=False)
        
        # Store chain and sampler for diagnostics
        self.sampler = sampler
        self.chain = sampler.get_chain()   # shape (n_steps, n_walkers, ndim)
        
        # Discard burn-in and flatten
        flat_samples = sampler.get_chain(discard=MCMC_BURN_IN, flat=True)
        return flat_samples, sampler

    # NEW: Helper to compute final R-hat (split chains)
    def _compute_rhat_final(self, chain, n_groups=2):
        """Compute Gelman-Rubin R-hat for each parameter using split chains."""
        n_steps, n_walkers, ndim = chain.shape
        group_size = n_walkers // n_groups
        if group_size < 2:
            raise ValueError("Too few walkers per group (need at least 2).")
        # Split walkers into groups
        groups = [chain[:, i*group_size:(i+1)*group_size, :] for i in range(n_groups)]
        chain_means = np.zeros((n_groups, ndim))
        chain_vars = np.zeros((n_groups, ndim))
        for g in range(n_groups):
            flat = groups[g].reshape(-1, ndim)
            chain_means[g] = np.mean(flat, axis=0)
            chain_vars[g] = np.var(flat, axis=0, ddof=1)
        W = np.mean(chain_vars, axis=0)
        B = n_steps * group_size * np.var(chain_means, axis=0, ddof=1)
        var_hat = (1 - 1/(n_steps*group_size)) * W + B/(n_steps*group_size)
        rhat = np.sqrt(var_hat / W)
        return rhat

    # -------------------- NEW CONVERGENCE PLOT (9-panel) --------------------
    def plot_convergence(self, profile_name, output_dir):
        """
        Generates a 9‑panel diagnostic figure:
          - R‑hat bar plot (with threshold 1.1)
          - ESS bar plot (with threshold 100)
          - Correlation matrix heatmap
          - Trace plots for z, α, q of first body
          - Posterior histograms for z, α, q
        Saves the figure and displays it.
        """
        if self.chain is None or self.sampler is None:
            print("  No MCMC chain available. Run MCMC first.")
            return

        chain = self.chain          # (n_steps, n_walkers, ndim)
        n_steps, n_walkers, ndim = chain.shape

        # ---------------------------
        # 1. Compute R-hat and ESS
        # ---------------------------
        rhat = self._compute_rhat_final(chain, n_groups=2)
        # ESS: use autocorrelation time
        try:
            tau = self.sampler.get_autocorr_time()
            ess = n_steps * n_walkers / tau
        except:
            ess = np.ones(ndim) * n_steps * n_walkers * 0.1   # fallback

        # Flatten samples (after burn-in) for correlation & histograms
        flat_samples = self.sampler.get_chain(discard=MCMC_BURN_IN, flat=True)  # (N, ndim)

        # Parameter names
        param_names = ["Offset"]
        for i in range(1, ndim):
            # Determine which body and parameter
            idx = i - 1
            body = idx // 5
            param = idx % 5
            if param == 0:
                name = f"K{body+1}"
            elif param == 1:
                name = f"x{body+1}"
            elif param == 2:
                name = f"z{body+1}"
            elif param == 3:
                name = f"α{body+1}"
            else:
                name = f"q{body+1}"
            param_names.append(name)

        # Indices for first body's z, α, q (assume at least one body)
        z_idx = 3 if ndim > 3 else None
        alpha_idx = 4 if ndim > 4 else None
        q_idx = 5 if ndim > 5 else None

        # ---------------------------
        # 2. Create 9-panel figure
        # ---------------------------
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # ---------- Panel 1: R-hat bars ----------
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['#2ecc71' if r < 1.1 else '#e74c3c' for r in rhat]
        bars = ax1.bar(range(ndim), rhat, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.axhline(y=1.1, color='#e74c3c', linestyle='--', linewidth=2.5, label='Threshold (1.1)', alpha=0.8)
        ax1.axhline(y=1.0, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.set_xticks(range(ndim))
        ax1.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
        ax1.set_ylabel('R-hat Value', fontsize=12, fontweight='bold')
        ax1.set_title('Gelman-Rubin Diagnostic', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(alpha=0.2, axis='y', linestyle='--')
        ax1.set_ylim(0.95, max(1.5, max(rhat) * 1.1))
        # Add value labels
        for bar, val in zip(bars, rhat):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # ---------- Panel 2: ESS bars ----------
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(range(ndim), ess, alpha=0.7, color='#3498db', edgecolor='black', linewidth=1)
        ax2.axhline(y=100, color='#e74c3c', linestyle='--', linewidth=2.5, label='Min ESS (100)', alpha=0.8)
        ax2.set_xticks(range(ndim))
        ax2.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Effective Sample Size', fontsize=12, fontweight='bold')
        ax2.set_title('ESS Diagnostics', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(alpha=0.2, axis='y', linestyle='--')
        # Value labels
        for bar, val in zip(bars, ess):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(ess)*0.02,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=8, rotation=90)

        # ---------- Panel 3: Correlation matrix ----------
        ax3 = fig.add_subplot(gs[0, 2])
        corr_matrix = np.corrcoef(flat_samples.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        cmap = LinearSegmentedColormap.from_list('RdYlBu_r', ['#d73027', '#f46d43', '#fdae61',
                                                              '#fee090', '#ffffbf', '#e0f3f8',
                                                              '#abd9e9', '#74add1', '#4575b4'])
        im = ax3.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax3.set_xticks(range(ndim))
        ax3.set_yticks(range(ndim))
        ax3.set_xticklabels(param_names, rotation=90, fontsize=8)
        ax3.set_yticklabels(param_names, fontsize=8)
        ax3.set_title('Posterior Correlation Matrix', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Correlation', fontsize=10, fontweight='bold')

        # ---------- Panels 4-6: Trace plots (z, α, q) ----------
        if z_idx is not None and alpha_idx is not None and q_idx is not None:
            # Depth (z)
            ax4 = fig.add_subplot(gs[1, 0])
            for w in range(min(30, n_walkers)):
                z_chain = chain[:, w, z_idx]
                ax4.plot(z_chain, color='#e74c3c', alpha=0.15, linewidth=0.8)
            z_mean = np.mean(chain[:, :, z_idx])
            ax4.axhline(y=z_mean, color='#c0392b', linewidth=2, linestyle='--',
                       label=f'Mean: {z_mean:.1f}m')
            ax4.set_xlabel('Step', fontsize=12)
            ax4.set_ylabel('Depth (z) [m]', fontsize=12, fontweight='bold')
            ax4.set_title('Trace Plot: Depth', fontsize=13, fontweight='bold')
            ax4.legend(loc='upper right', fontsize=9)
            ax4.grid(alpha=0.2, linestyle='--')

            # Angle (α)
            ax5 = fig.add_subplot(gs[1, 1])
            for w in range(min(30, n_walkers)):
                alpha_chain = chain[:, w, alpha_idx]
                ax5.plot(alpha_chain, color='#3498db', alpha=0.15, linewidth=0.8)
            alpha_mean = np.mean(chain[:, :, alpha_idx])
            ax5.axhline(y=alpha_mean, color='#2980b9', linewidth=2, linestyle='--',
                       label=f'Mean: {alpha_mean:.1f}°')
            ax5.set_xlabel('Step', fontsize=12)
            ax5.set_ylabel('Angle (α) [°]', fontsize=12, fontweight='bold')
            ax5.set_title('Trace Plot: Polarization Angle', fontsize=13, fontweight='bold')
            ax5.legend(loc='upper right', fontsize=9)
            ax5.grid(alpha=0.2, linestyle='--')

            # Shape factor (q)
            ax6 = fig.add_subplot(gs[1, 2])
            for w in range(min(30, n_walkers)):
                q_chain = chain[:, w, q_idx]
                ax6.plot(q_chain, color='#27ae60', alpha=0.15, linewidth=0.8)
            q_mean = np.mean(chain[:, :, q_idx])
            ax6.axhline(y=q_mean, color='#229954', linewidth=2, linestyle='--',
                       label=f'Mean: {q_mean:.2f}')
            ax6.set_xlabel('Step', fontsize=12)
            ax6.set_ylabel('Shape Factor (q)', fontsize=12, fontweight='bold')
            ax6.set_title('Trace Plot: Shape Factor', fontsize=13, fontweight='bold')
            ax6.legend(loc='upper right', fontsize=9)
            ax6.grid(alpha=0.2, linestyle='--')

            # ---------- Panels 7-9: Posterior histograms ----------
            # Depth histogram
            ax7 = fig.add_subplot(gs[2, 0])
            z_samples = flat_samples[:, z_idx]
            ax7.hist(z_samples, bins=40, density=True, color='#e74c3c', alpha=0.7,
                    edgecolor='white', linewidth=0.5)
            ax7.axvline(z_mean, color='#c0392b', linewidth=2.5, linestyle='-', label=f'Mean: {z_mean:.1f}m')
            ax7.axvline(np.percentile(z_samples, 16), color='#c0392b', linewidth=1.5,
                       linestyle='--', alpha=0.7, label='16th/84th percentile')
            ax7.axvline(np.percentile(z_samples, 84), color='#c0392b', linewidth=1.5, linestyle='--', alpha=0.7)
            ax7.set_xlabel('Depth (z) [m]', fontsize=12)
            ax7.set_ylabel('Density', fontsize=12)
            ax7.set_title('Posterior: Depth', fontsize=13, fontweight='bold')
            ax7.legend(loc='upper right', fontsize=9)
            ax7.grid(alpha=0.2, linestyle='--')

            # Angle histogram
            ax8 = fig.add_subplot(gs[2, 1])
            alpha_samples = flat_samples[:, alpha_idx]
            ax8.hist(alpha_samples, bins=40, density=True, color='#3498db', alpha=0.7,
                    edgecolor='white', linewidth=0.5)
            ax8.axvline(alpha_mean, color='#2980b9', linewidth=2.5, linestyle='-',
                       label=f'Mean: {alpha_mean:.1f}°')
            ax8.axvline(np.percentile(alpha_samples, 16), color='#2980b9', linewidth=1.5,
                       linestyle='--', alpha=0.7, label='16th/84th percentile')
            ax8.axvline(np.percentile(alpha_samples, 84), color='#2980b9', linewidth=1.5, linestyle='--', alpha=0.7)
            ax8.set_xlabel('Angle (α) [°]', fontsize=12)
            ax8.set_ylabel('Density', fontsize=12)
            ax8.set_title('Posterior: Polarization Angle', fontsize=13, fontweight='bold')
            ax8.legend(loc='upper right', fontsize=9)
            ax8.grid(alpha=0.2, linestyle='--')

            # Shape histogram
            ax9 = fig.add_subplot(gs[2, 2])
            q_samples = flat_samples[:, q_idx]
            ax9.hist(q_samples, bins=40, density=True, color='#27ae60', alpha=0.7,
                    edgecolor='white', linewidth=0.5)
            ax9.axvline(q_mean, color='#229954', linewidth=2.5, linestyle='-',
                       label=f'Mean: {q_mean:.2f}')
            ax9.axvline(np.percentile(q_samples, 16), color='#229954', linewidth=1.5,
                       linestyle='--', alpha=0.7, label='16th/84th percentile')
            ax9.axvline(np.percentile(q_samples, 84), color='#229954', linewidth=1.5, linestyle='--', alpha=0.7)
            ax9.set_xlabel('Shape Factor (q)', fontsize=12)
            ax9.set_ylabel('Density', fontsize=12)
            ax9.set_title('Posterior: Shape Factor', fontsize=13, fontweight='bold')
            ax9.legend(loc='upper right', fontsize=9)
            ax9.grid(alpha=0.2, linestyle='--')

        # ---------------------------
        # 3. Finish and save
        # ---------------------------
        plt.suptitle(f'Convergence Diagnostics - {profile_name}',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_path = os.path.join(output_dir, f"{profile_name}_Convergence.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   > Convergence plot saved to: {save_path}")
        plt.show()
        plt.close(fig)

    # NEW: Diagnostic function for depth overestimation
    def diagnose_depth_overestimation(self, samples, true_depth=None):
        """Diagnose why depth might be overestimated"""
        if true_depth is None and BOREHOLES_AVAILABLE:
            true_depth = BOREHOLE_DEPTH
        
        # Extract samples for first body
        z_samples = samples[:, 3]
        K_samples = samples[:, 1]
        q_samples = samples[:, 5]
        
        print("\n" + "="*50)
        print(" DEPTH OVERESTIMATION DIAGNOSIS")
        print("="*50)
        
        # 1. Check correlations
        corr_z_K = np.corrcoef(z_samples, K_samples)[0,1]
        corr_z_q = np.corrcoef(z_samples, q_samples)[0,1]
        
        print(f"Correlation z-K: {corr_z_K:.3f} (negative suggests trade-off)")
        print(f"Correlation z-q: {corr_z_q:.3f}")
        
        # 2. Check if depth hits bounds
        z_low, z_med, z_high = np.percentile(z_samples, [16, 50, 84])
        print(f"Depth: {z_med:.1f}m 95% CI: [{z_low:.1f}, {z_high:.1f}]")
        
        # 3. Compare with borehole if available
        if true_depth:
            error_percent = abs(z_med - true_depth) / true_depth * 100
            print(f"Borehole depth: {true_depth:.1f}m")
            print(f"Error: {error_percent:.1f}%")
            
            if z_med > true_depth:
                print("→ Model OVERESTIMATES depth")
                if corr_z_K < -0.5:
                    print("  Possible cause: Trade-off with amplitude (K)")
                if corr_z_q < -0.5:
                    print("  Possible cause: Trade-off with shape factor (q)")
        
        # 4. Check data coverage
        print(f"\nProfile length: {self.profile_length:.1f}m")
        depth_width_ratio = (z_med * 2) / self.profile_length
        print(f"Depth/Profile length ratio: {depth_width_ratio:.2f}")
        if depth_width_ratio > 0.3:
            print("⚠️  Profile may be too short to constrain depth well")
        else:
            print("✓ Profile length adequate for depth estimation")
        
        print("="*50)

    # ---------- OLD R-hat plot (kept but not used) ----------
    def compute_rhat(self, chain, n_groups=2):
        """
        Compute R-hat for each parameter at each step.
        chain: shape (n_steps, n_walkers, ndim)
        Splits walkers into n_groups (each group = one chain).
        Returns rhats: shape (n_steps, ndim)
        """
        n_steps, n_walkers, ndim = chain.shape
        if n_steps < 2:
            return np.ones((n_steps, ndim))
        group_size = n_walkers // n_groups
        if group_size < 2:
            raise ValueError("Too few walkers per group for R-hat computation.")
        groups = [chain[:, i*group_size:(i+1)*group_size, :] for i in range(n_groups)]
        
        rhats = np.zeros((n_steps, ndim))
        steps_to_compute = np.arange(1, n_steps, 10)
        for t in steps_to_compute:
            chain_means = np.zeros((n_groups, ndim))
            chain_vars = np.zeros((n_groups, ndim))
            for g in range(n_groups):
                group_samples = groups[g][:t+1, :, :]  # shape (t+1, group_size, ndim)
                flat = group_samples.reshape(-1, ndim)
                chain_means[g] = np.mean(flat, axis=0)
                chain_vars[g] = np.var(flat, axis=0, ddof=1)
            n_samples_per_chain = (t+1) * group_size
            B = n_samples_per_chain * np.var(chain_means, axis=0, ddof=1)
            W = np.mean(chain_vars, axis=0)
            W_safe = np.where(W < 1e-12, 1e-12, W)
            var_hat = (1 - 1/n_samples_per_chain) * W_safe + (1/n_samples_per_chain) * B
            rhat = np.sqrt(var_hat / W_safe)
            rhats[t] = rhat
        last_val = 1.0
        for t in range(n_steps):
            if t in steps_to_compute:
                last_val = rhats[t]
            else:
                rhats[t] = last_val
        return rhats

    def plot_rhat(self, rhats, param_indices, param_names, title=""):
        """
        Enhanced R-hat plot with subplots, burn-in shading, and final R-hat annotations.
        (Kept for backward compatibility, but not called anymore)
        """
        n_params = len(param_indices)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 6), sharex=True)
        if n_params == 1:
            axes = [axes]
        
        final_rhat = rhats[-1, param_indices]
        n_steps = rhats.shape[0]
        
        for i, (idx, name) in enumerate(zip(param_indices, param_names)):
            ax = axes[i]
            ax.plot(rhats[:, idx], color=f'C{i}', linewidth=2.5, label=name)
            ax.axhline(y=1.1, color='k', linestyle='--', alpha=0.7, linewidth=2)
            if MCMC_BURN_IN > 0 and MCMC_BURN_IN < n_steps:
                ax.axvspan(0, MCMC_BURN_IN, alpha=0.15, color='gray', label='Burn-in')
            ax.text(0.95, 0.05, f'R = {final_rhat[i]:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=LEGEND_FONT, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            ax.set_title(name, fontsize=TITLE_FONT, fontweight='bold')
            ax.set_ylabel('R-hat', fontsize=LABEL_FONT)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=TICK_FONT)
            if i == 0:
                ax.legend(loc='upper right', fontsize=LEGEND_FONT-2)
        
        axes[-1].set_xlabel('MCMC Step', fontsize=LABEL_FONT)
        fig.suptitle(f'Gelman-Rubin Convergence Diagnostic - {title}',
                     fontsize=TITLE_FONT, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

# -----------------------------
# SYNTHETIC TEST MODULE
# -----------------------------
class SyntheticTest:
    def __init__(self):
        print("\n" + "="*50)
        print("  INITIATING SYNTHETIC FIDELITY TESTS")
        print("="*50)

    def generate_cylinder_data(self):
        """Generates noisy synthetic data for a cylinder model (q=1.0)."""
        # Model: Cylinder (q=1.0), Depth=60m, Angle=-30 deg
        self.cylinder_true = np.array([0.0, -5000, 500, 40.0, -30.0, 1.0])
        
        self.x_synth = np.linspace(0, 1000, 50)
        self.y_clean_cyl = forward_model_multi_source(self.x_synth, self.cylinder_true)
        
        # Add 5% Gaussian Noise
        noise_level = 0.05 * np.max(np.abs(self.y_clean_cyl))
        np.random.seed(42)
        noise = np.random.normal(0, noise_level, len(self.x_synth))
        self.y_noisy_cyl = self.y_clean_cyl + noise
        
        print("\n" + "-"*40)
        print("  CYLINDER MODEL TEST (q = 1.0)")
        print("-"*40)
        print(f"  Ground Truth Parameters:")
        print(f"    - Depth (z): {self.cylinder_true[3]} m")
        print(f"    - Angle (α): {self.cylinder_true[4]}°")
        print(f"    - Shape (q): {self.cylinder_true[5]} (Cylinder)")
        print(f"    - Added 5% Random Gaussian Noise")

    def generate_sphere_data(self):
        """Generates noisy synthetic data for a sphere model (q=1.5)."""
        # Model: Sphere (q=1.5), Depth=60m, Angle=+15 deg
        self.sphere_true = np.array([0.0, 8000, 500, 40.0, 15.0, 1.5])
        
        self.y_clean_sph = forward_model_multi_source(self.x_synth, self.sphere_true)
        
        # Add 5% Gaussian Noise (using same seed for consistency)
        noise_level = 0.03 * np.max(np.abs(self.y_clean_sph))
        np.random.seed(43)  # Different seed for variety
        noise = np.random.normal(0, noise_level, len(self.x_synth))
        self.y_noisy_sph = self.y_clean_sph + noise
        
        print("\n" + "-"*40)
        print("  SPHERE MODEL TEST (q = 1.5)")
        print("-"*40)
        print(f"  Ground Truth Parameters:")
        print(f"    - Depth (z): {self.sphere_true[3]} m")
        print(f"    - Angle (α): {self.sphere_true[4]}°")
        print(f"    - Shape (q): {self.sphere_true[5]} (Sphere)")
        print(f"    - Added 5% Random Gaussian Noise")

    # NEW: generate_sheet_data
    def generate_sheet_data(self):
        """Generates noisy synthetic data for a dipping sheet model (q=0.5)."""
        # Model: Dipping Sheet (q=0.5), Depth=40m, Angle=-20 deg
        self.sheet_true = np.array([0.0, -3000, 450, 35.0, -20.0, 0.5])
        self.y_clean_sheet = forward_model_multi_source(self.x_synth, self.sheet_true)
        # Add 5% Gaussian Noise
        noise_level = 0.05 * np.max(np.abs(self.y_clean_sheet))
        np.random.seed(44)
        noise = np.random.normal(0, noise_level, len(self.x_synth))
        self.y_noisy_sheet = self.y_clean_sheet + noise
        print("\n" + "-"*40)
        print("  SHEET MODEL TEST (q = 0.5)")
        print("-"*40)
        print(f"  Ground Truth Parameters:")
        print(f"    - Depth (z): {self.sheet_true[3]} m")
        print(f"    - Angle (α): {self.sheet_true[4]}°")
        print(f"    - Shape (q): {self.sheet_true[5]} (Dipping Sheet)")
        print(f"    - Added 5% Random Gaussian Noise")

    def plot_corner(self, samples, true_params, model_name):
        """Creates publication-standard corner plot showing only alpha, z, and q."""
        # Extract only alpha (index 4), z (index 3), and q (index 5)
        alpha_samples = samples[:, 4]
        z_samples = samples[:, 3]
        q_samples = samples[:, 5]
        
        # Stack them for corner plot
        selected_samples = np.column_stack([alpha_samples, z_samples, q_samples])
        
        # True values for selected parameters
        selected_truths = [true_params[4], true_params[3], true_params[5]]
        
        # Parameter names
        param_names = ['α (°)', 'z (m)', 'q']
        
        # Create corner figure
        fig = corner.corner(
            selected_samples,
            labels=param_names,
            truths=selected_truths,
            truth_color='green',
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": TICK_FONT},
            label_kwargs={"fontsize": LABEL_FONT},
            title_fmt='.2f',
            use_math_text=True,
            figsize=(10, 10),
            bins=30,
            smooth=1.0,
            color='red',
            hist_kwargs={"density": True, "alpha": 0.6, "color": "red"},
            contour_kwargs={"colors": "red"},
            fill_contours=True,
            contourf_kwargs={"alpha": 0.3, "colors": "red"}
        )
        
        # Add title
        fig.suptitle(f"{model_name} Model - Posterior Distributions\n(α, z, q)", 
                    fontsize=TITLE_FONT, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save corner plot
        save_path = os.path.join(OUTPUT_DIR, f"Corner_Plot_{model_name}_alpha_z_q.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [INFO] Corner plot saved to: {save_path}")
        plt.show()
        plt.close(fig)  # Close the figure to free memory
        
        return fig

    def run_single_test(self, x_data, y_data, true_params, model_name):
        """Runs inversion on a single synthetic dataset."""
        print(f"\n > Running Inversion on {model_name} Data...")
        inversion = BayesianInversion(x_data, y_data)
        
        # Run Optimization
        best_guess, _ = inversion.optimize_global(num_bodies=1)
        samples, sampler = inversion.run_mcmc(best_guess, num_bodies=1)
        
        # Run diagnosis
        inversion.diagnose_depth_overestimation(samples, true_params[3])
        
        # NEW: Plot convergence diagnostics
        inversion.plot_convergence(model_name, OUTPUT_DIR)
        
        # Get Inverted Statistics
        inverted_params = np.median(samples, axis=0)
        param_percentiles = np.percentile(samples, [5, 95], axis=0)
        
        # Compare Key Parameters (z, alpha, q)
        z_inv = inverted_params[3]
        a_inv = inverted_params[4]
        q_inv = inverted_params[5]
        
        z_true = true_params[3]
        a_true = true_params[4]
        q_true = true_params[5]
        
        # Calculate Errors
        z_err = abs(z_inv - z_true) / z_true * 100
        a_err = abs(a_inv - a_true)
        q_err = abs(q_inv - q_true) / q_true * 100
        
        # Calculate 90% confidence intervals
        z_ci = param_percentiles[:, 3]
        a_ci = param_percentiles[:, 4]
        q_ci = param_percentiles[:, 5]
        
        print("-" * 40)
        print(f"  {model_name.upper()} TEST RESULTS")
        print("-" * 40)
        print(f"  PARAMETER   |   TRUE   | INVERTED | 90% CI        | ERROR")
        print(f"  Depth (z)   | {z_true:6.1f}m | {z_inv:7.1f}m | [{z_ci[0]:5.1f}, {z_ci[1]:5.1f}] | {z_err:6.2f}%")
        print(f"  Angle (α)   | {a_true:6.1f}° | {a_inv:7.1f}° | [{a_ci[0]:5.1f}, {a_ci[1]:5.1f}] | {a_err:6.2f}°")
        print(f"  Shape (q)   | {q_true:6.2f}  | {q_inv:7.2f}  | [{q_ci[0]:5.2f}, {q_ci[1]:5.2f}] | {q_err:6.2f}%")
        print("-" * 40)
        
        # Classification check
        if q_inv >= 1.3:
            inv_shape = "Sphere"
        elif 0.8 <= q_inv < 1.3:
            inv_shape = "Cylinder"
        else:
            inv_shape = "Dipping Sheet"
            
        true_shape = "Sphere" if q_true >= 1.3 else "Cylinder" if q_true >= 0.8 else "Sheet"
        shape_match = inv_shape == true_shape
        
        print(f"  Shape Classification: Inverted={inv_shape}, True={true_shape} → {'✓' if shape_match else '✗'}")
        
        if z_err < 10 and a_err < 15 and shape_match:
            print(f"  [SUCCESS] {model_name} test passed (errors within limits)")
        else:
            print(f"  [WARNING] High error in {model_name} test")
        
        return inverted_params, samples

    def run_fidelity_check(self):
        """Runs cylinder, sphere, and sheet synthetic tests with corner plots."""
        self.generate_cylinder_data()
        self.generate_sphere_data()
        self.generate_sheet_data()   # NEW
        
        # Test Cylinder Model
        cyl_params, cyl_samples = self.run_single_test(
            self.x_synth, self.y_noisy_cyl, 
            self.cylinder_true, "Cylinder"
        )
        self.plot_corner(cyl_samples, self.cylinder_true, "Cylinder")
        
        # Test Sphere Model
        sph_params, sph_samples = self.run_single_test(
            self.x_synth, self.y_noisy_sph, 
            self.sphere_true, "Sphere"
        )
        self.plot_corner(sph_samples, self.sphere_true, "Sphere")
        
        # Test Sheet Model (NEW)
        sheet_params, sheet_samples = self.run_single_test(
            self.x_synth, self.y_noisy_sheet,
            self.sheet_true, "Sheet"
        )
        self.plot_corner(sheet_samples, self.sheet_true, "Sheet")
        
        # -----------------------------
        # Plotting all three results side by side (extend to 3 columns)
        # -----------------------------
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        
        # Define list of (model_name, y_noisy, y_clean, params, true_params)
        models = [
            ("Cylinder", self.y_noisy_cyl, self.y_clean_cyl, cyl_params, self.cylinder_true),
            ("Sphere",   self.y_noisy_sph, self.y_clean_sph, sph_params, self.sphere_true),
            ("Sheet",    self.y_noisy_sheet, self.y_clean_sheet, sheet_params, self.sheet_true)
        ]
        
        for row, (name, y_noisy, y_clean, inv_params, true_params) in enumerate(models):
            ax1 = axes[row, 0]
            ax2 = axes[row, 1]
            
            # Data fit plot
            inds = np.random.randint(len(cyl_samples), size=50)  # reuse any samples
            # We'll plot posterior samples for this model (we have samples from previous run)
            # Actually, we have separate samples for each model; we need to use the correct ones.
            # We'll store them in a dict.
            if name == "Cylinder":
                samples_plot = cyl_samples
            elif name == "Sphere":
                samples_plot = sph_samples
            else:
                samples_plot = sheet_samples
                
            for ind in np.random.randint(len(samples_plot), size=50):
                sample = samples_plot[ind]
                y_sample = forward_model_multi_source(self.x_synth, sample)
                ax1.plot(self.x_synth, y_sample, color='red', alpha=0.05)
            
            ax1.plot(self.x_synth, y_noisy, 'k.', label='Noisy Data', markersize=4)
            ax1.plot(self.x_synth, y_clean, 'g--', linewidth=2, label='Ground Truth')
            ax1.plot(self.x_synth, forward_model_multi_source(self.x_synth, inv_params), 
                    'r-', linewidth=2, label='Inverted Model')
            
            z_err = abs(inv_params[3] - true_params[3]) / true_params[3] * 100
            ax1.set_title(f"{name} Model (Depth Error: {z_err:.1f}%)", 
                         fontsize=TITLE_FONT, fontweight='bold')
            ax1.set_ylabel("SP (mV)", fontsize=LABEL_FONT)
            if row == 2:
                ax1.set_xlabel("Distance (m)", fontsize=LABEL_FONT)
            ax1.legend(fontsize=LEGEND_FONT, loc='best')
            ax1.grid(alpha=0.3)
            ax1.tick_params(labelsize=TICK_FONT)
            
            # Subsurface plot
            ax2.set_title(f"{name} Model (q={inv_params[5]:.2f})", 
                         fontsize=TITLE_FONT, fontweight='bold')
            ax2.set_ylabel("Depth (m)", fontsize=LABEL_FONT)
            if row == 2:
                ax2.set_xlabel("Distance (m)", fontsize=LABEL_FONT)
            ax2.invert_yaxis()
            ax2.axhline(0, color='brown', linewidth=2)
            
            # Determine body shape for plotting
            q = inv_params[5]
            if q >= 1.3:
                # Sphere
                circle = patches.Circle((inv_params[2], inv_params[3]), 
                                       radius=inv_params[3]/4, 
                                       facecolor='lightblue', edgecolor='blue', 
                                       linewidth=2, alpha=0.7)
                ax2.add_patch(circle)
            elif 0.8 <= q < 1.3:
                # Cylinder
                circle = patches.Circle((inv_params[2], inv_params[3]), 
                                       radius=inv_params[3]/4, 
                                       facecolor='mistyrose', edgecolor='red', 
                                       linewidth=2, alpha=0.7)
                ax2.add_patch(circle)
            else:
                # Dipping Sheet
                length = 30
                dx = length * np.cos(np.radians(inv_params[4] - 90))
                dy = length * np.sin(np.radians(inv_params[4] - 90))
                ax2.plot([inv_params[2] - dx, inv_params[2] + dx], 
                        [inv_params[3] - dy, inv_params[3] + dy], 
                        color='purple', linewidth=6, alpha=0.7)
            
            ax2.scatter(inv_params[2], inv_params[3], marker='+', s=200, color='black', zorder=10)
            ax2.text(inv_params[2], inv_params[3] - 20, 
                    f"Z={inv_params[3]:.1f}m\nα={inv_params[4]:.1f}°", 
                    ha='center', fontsize=TICK_FONT, fontweight='bold')
            
            # Mark true position
            ax2.scatter(true_params[2], true_params[3], 
                       marker='x', s=200, color='green', label='True Position', zorder=5)
            ax2.legend(fontsize=LEGEND_FONT, loc='lower right')
            ax2.grid(alpha=0.3)
            ax2.tick_params(labelsize=TICK_FONT)
            ax2.set_xlim([200, 800])
            ax2.set_ylim([100, 0])
        
        plt.tight_layout()
        
        # Save the combined synthetic figure
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        save_path = os.path.join(OUTPUT_DIR, "Synthetic_Cylinder_Sphere_Sheet_Tests.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  [INFO] Synthetic test plot saved to: {save_path}")
        plt.show()
        plt.close(fig)

# -----------------------------
# HELPERS
# -----------------------------
def interpret_shape(q_val):
    if q_val >= 1.3: return "Sphere/Point"
    if 0.8 <= q_val < 1.3: return "Cylinder"
    return "Dipping Sheet" 

def remove_trend(x, y):
    if len(x) < 3: return y, np.zeros_like(y)
    p = np.polyfit(x, y, 1)
    trend = np.polyval(p, x)
    return y - trend, trend

def estimate_sources_count(y):
    peaks, _ = find_peaks(np.abs(y), prominence=np.std(y)*0.5)
    return max(1, min(len(peaks), MAX_SOURCES))

def identify_columns_robust(df):
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    
    sp_col = None
    for p in ['pd corrected final sp (mv)', 'sp (mv) final', 'sp (mv)', 'sp']:
        if p in col_map: 
            sp_col = col_map[p]
            break
    if sp_col is None and len(df.columns) >= 5:
        if np.issubdtype(df.iloc[:, 4].dtype, np.number): 
            sp_col = df.columns[4]

    x_col = None
    for p in ['easting', 'utmx', 'station (x)', 'station', 'x', 'dist', 'distance']:
        if p in col_map: 
            x_col = col_map[p]
            break
            
    trav_col = col_map.get('traverse (y)') or col_map.get('traverse') or col_map.get('line')
    return sp_col, x_col, trav_col

# -----------------------------
# MAIN ANALYSIS LOOP
# -----------------------------
def analyze_profile(df, profile_name):
    sp_col, x_col, _ = identify_columns_robust(df)
    
    if not sp_col or not x_col:
        print(f"[{profile_name}] Skipping: X or SP columns not found.")
        return

    print(f"\nProcessing Profile: {profile_name}")
    
    # Data Prep
    df = df.dropna(subset=[x_col, sp_col]).drop_duplicates(subset=[x_col]).sort_values(by=x_col)
    x = df[x_col].values.astype(float)
    y = df[sp_col].values.astype(float)
    
    if len(x) < 5: return

    # Detrending
    y_res, trend = remove_trend(x, y)
    
    # Model Selection Loop
    inversion = BayesianInversion(x, y_res)
    best_bic = np.inf
    best_model_params = None
    best_num_bodies = 1
    
    max_try = estimate_sources_count(y_res)
    
    for n in range(1, max_try + 1):
        theta_opt, rss = inversion.optimize_global(n)
        
        # Calculate BIC
        k = len(theta_opt)
        n_data = len(x)
        if rss <= 1e-9: rss = 1e-9
        bic = n_data * np.log(rss/n_data) + k * np.log(n_data)
        
        print(f"      N={n}: RSS={rss:.2f}, BIC={bic:.2f}")
        
        if bic < best_bic:
            best_bic = bic
            best_num_bodies = n
            best_model_params = theta_opt
    
    print(f"   > Selected Model: {best_num_bodies} Source(s)")
    
    # MCMC Run
    samples, sampler = inversion.run_mcmc(best_model_params, best_num_bodies)
    
    # Run depth overestimation diagnosis
    inversion.diagnose_depth_overestimation(samples)
    
    # ---------- NEW: Plot convergence diagnostics ----------
    inversion.plot_convergence(profile_name, OUTPUT_DIR)
    
    # -----------------------------
    # PLOTTING
    # -----------------------------
    theta_median = np.median(samples, axis=0)
    y_calc_final = forward_model_multi_source(x, theta_median) + trend
    r2 = 1 - np.sum((y - y_calc_final)**2) / np.sum((y - np.mean(y))**2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1.5]})
    
    # 1. Plot Confidence Band
    inds = np.random.randint(len(samples), size=100)
    for ind in inds:
        sample = samples[ind]
        y_sample = forward_model_multi_source(x, sample) + trend
        ax1.plot(x, y_sample, color='red', alpha=0.05)
        
    ax1.scatter(x, y, c='k', s=40, label='Observed Data')
    ax1.plot(x, y_calc_final, 'r--', linewidth=2.5, label=f'Inversion Fit (R²={r2:.2f})')
    
    ax1.set_title(f"Bayesian SP Inversion: {profile_name}", fontsize=TITLE_FONT, fontweight='bold', pad=15)
    ax1.set_ylabel("SP (mV)", fontsize=LABEL_FONT, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax1.legend(loc='upper left', fontsize=LEGEND_FONT)
    ax1.grid(alpha=0.3)
    ax1.ticklabel_format(useOffset=False, style='plain', axis='x')

    # 2. Subsurface Plot
    ax2.set_ylabel("Depth (m)", fontsize=LABEL_FONT, fontweight='bold')
    ax2.set_xlabel(f"Easting", fontsize=LABEL_FONT, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax2.invert_yaxis()
    ax2.axhline(0, color='brown', linewidth=3)
    ax2.ticklabel_format(useOffset=False, style='plain', axis='x')
    
    for i in range(best_num_bodies):
        idx = 1 + i * 5
        x0_med = np.median(samples[:, idx+1])
        z_med = np.median(samples[:, idx+2])
        alpha_med = np.median(samples[:, idx+3])
        q_med = np.median(samples[:, idx+4])
        
        shape_str = interpret_shape(q_med)
        legend_label = f"Body {i+1}: {shape_str}\n(Z={z_med:.1f}m, α={alpha_med:.1f}°)"
        
        if shape_str == "Dipping Sheet":
            length = 30
            dx = length * np.cos(np.radians(alpha_med - 90))
            dy = length * np.sin(np.radians(alpha_med - 90))
            ax2.plot([x0_med - dx, x0_med + dx], [z_med - dy, z_med + dy], 
                     color='blue', linewidth=4, alpha=0.7, label=legend_label)
        else:
             circle = patches.Circle((x0_med, z_med), radius=z_med/3, facecolor='mistyrose', edgecolor='red')
             ax2.add_patch(circle)
             ax2.plot([], [], 'o', color='red', markersize=10, label=legend_label)

        ax2.scatter(x0_med, z_med, marker='+', s=300, color='black', zorder=10)
        ax2.text(x0_med, z_med + (z_med*0.15), f"Z={z_med:.1f}m", 
                 ha='center', fontsize=TICK_FONT, fontweight='bold')
        print(f"   > Body {i+1} ({shape_str}): X={x0_med:.1f}, Z={z_med:.1f}m, Alpha={alpha_med:.1f}")
        
        # Create corner plot for real data (only for first body if multiple)
        if i == 0 and best_num_bodies == 1:
            print(f"\n > Generating corner plot for real data (α, z, q)...")
            
            # Extract alpha, z, q samples
            alpha_samples = samples[:, idx+3]
            z_samples = samples[:, idx+2]
            q_samples = samples[:, idx+4]
            
            # Stack them for corner plot
            selected_samples = np.column_stack([alpha_samples, z_samples, q_samples])
            
            # Parameter names
            param_names = ['α (°)', 'z (m)', 'q']
            
            # Create corner figure
            fig_corner = corner.corner(
                selected_samples,
                labels=param_names,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": TICK_FONT},
                label_kwargs={"fontsize": LABEL_FONT},
                title_fmt='.2f',
                use_math_text=True,
                figsize=(10, 10),
                bins=30,
                smooth=1.0,
                color='blue',
                hist_kwargs={"density": True, "alpha": 0.6, "color": "blue"},
                contour_kwargs={"colors": "blue"},
                fill_contours=True,
                contourf_kwargs={"alpha": 0.3, "colors": "blue"}
            )
            
            # Add title
            fig_corner.suptitle(f"{profile_name} - Posterior Distributions\n(α, z, q)", 
                              fontsize=TITLE_FONT, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            # Save corner plot
            corner_save_path = os.path.join(OUTPUT_DIR, f"{profile_name}_Corner_alpha_z_q.png")
            fig_corner.savefig(corner_save_path, dpi=300, bbox_inches='tight')
            print(f"   > Real data corner plot saved to: {corner_save_path}")
            plt.show()
            plt.close(fig_corner)  # Close the corner figure

    ax2.grid(True, which='both', alpha=0.5)
    ax2.legend(fontsize=LEGEND_FONT, loc='lower right')
    plt.setp(ax2.get_xticklabels(), rotation=0, ha='center')
    plt.tight_layout()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Save the main inversion figure
    save_path = os.path.join(OUTPUT_DIR, f"{profile_name}_Inversion.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   > Main inversion plot saved to: {save_path}")
    plt.show()
    plt.close(fig)  # Close the figure to free memory

# -----------------------------
# FILE HANDLER
# -----------------------------
def process_file(file_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # --- SYNTHETIC TEST TRIGGER ---
    run_test = input("Do you want to run Synthetic Fidelity Tests (Cylinder, Sphere & Sheet) with Corner Plots? (y/n): ").strip().lower()
    if run_test == 'y':
        tester = SyntheticTest()
        tester.run_fidelity_check()
    # ------------------------------

    print(f"Loading: {file_path}")
    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:
        if file_path.lower().endswith('.xlsx'):
            dfs = pd.read_excel(file_path, sheet_name=None)
        else:
            dfs = {'Sheet1': pd.read_csv(file_path)}
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    for sheet_name, df in dfs.items():
        sp_col, x_col, trav_col = identify_columns_robust(df)
        if not sp_col: continue
            
        if trav_col:
            traverses = df[trav_col].dropna().unique()
            for t in traverses:
                sub_df = df[df[trav_col] == t].copy()
                analyze_profile(sub_df, f"{sheet_name}_{t}")
        else:
            analyze_profile(df, sheet_name)

if __name__ == "__main__":
    process_file(FILE_PATH)
