"""
Generate all camera-ready figures for the paper.

Usage (from the AISTATS2026/ directory):
    python code/generate_figures.py

Outputs:
    src/case1_tau.png, src/case2_tau.png         -- Example trajectory figures (Figs 1-2)
    src/prob_2.png                                -- Allocation probability (Fig 3)
    src/ATE_ney.png, src/ATE_adj.png             -- Running ATE (Figs 4a-4b)
    src/sd_2.png                                  -- MC standard deviation (Fig 5)
    src/CI.png                                    -- Confidence sequence (Fig 6)
    src/appendix/prob_09.png, ATE_*_09.png, ...  -- Appendix figures (beta=0.9)
    src/appendix/typeI_07.png, typeI_09.png      -- Type I error figures

Requirements: numpy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Core Functions (from smoother_figures.py)
# ============================================================

def compute_oracle_variance(beta, q, r):
    return q * r / (q + r * (1 - beta)**2)

def kalman_smoother(Y_obs, beta, q, r, mu_prior=0.0, P_prior=None):
    T = len(Y_obs)
    mu_filt = np.zeros(T)
    P_filt = np.zeros(T)
    mu_pred_seq = np.zeros(T)
    P_pred_seq = np.zeros(T)
    if P_prior is None:
        P_prior = q / (1 - beta**2) if abs(beta) < 1 else 1.0
    for t in range(T):
        if t == 0:
            mu_pred = beta * mu_prior
            P_pred = beta**2 * P_prior + q
        else:
            mu_pred = beta * mu_filt[t-1]
            P_pred = beta**2 * P_filt[t-1] + q
        mu_pred_seq[t] = mu_pred
        P_pred_seq[t] = P_pred
        K = P_pred / (P_pred + r)
        mu_filt[t] = mu_pred + K * (Y_obs[t] - mu_pred)
        P_filt[t] = (1 - K) * P_pred
    mu_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
    mu_smooth[T-1] = mu_filt[T-1]
    P_smooth[T-1] = P_filt[T-1]
    for t in range(T-2, -1, -1):
        J = P_filt[t] * beta / P_pred_seq[t+1]
        mu_smooth[t] = mu_filt[t] + J * (mu_smooth[t+1] - mu_pred_seq[t+1])
        P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t+1] - P_pred_seq[t+1])
    return mu_smooth, P_smooth, mu_filt, P_filt

def compute_model_adjusted_pi(sigma_1, sigma_0, q_1, q_0, beta_1, beta_0, n):
    pi = 0.5
    for _ in range(200):
        r_1 = sigma_1**2 / (n * pi)
        r_0 = sigma_0**2 / (n * (1 - pi))
        f_1 = q_1 / (q_1 + r_1 * (1 - beta_1)**2)
        f_0 = q_0 / (q_0 + r_0 * (1 - beta_0)**2)
        pi_new = f_1 * sigma_1 / (f_1 * sigma_1 + f_0 * sigma_0)
        pi_new = np.clip(pi_new, 0.01, 0.99)
        if abs(pi_new - pi) < 1e-10:
            break
        pi = pi_new
    return pi

def simulate_experiment(beta, q_1, q_0, sigma2_1, sigma2_0,
                        mu_init_1, mu_init_0, n, T, T0, allocation_rule, seed=None):
    """Simulate a batched adaptive experiment with individual-level data.

    Generates n individual outcomes per period, computes batch means for
    the Kalman filter/smoother, and uses the pooled within-period variance
    estimator for sigma_hat (matching the paper's formula).
    """
    rng = np.random.RandomState(seed)
    sigma_1, sigma_0 = np.sqrt(sigma2_1), np.sqrt(sigma2_0)
    # Generate latent state paths
    mu_1 = np.zeros(T + 1); mu_0 = np.zeros(T + 1)
    mu_1[0] = mu_init_1; mu_0[0] = mu_init_0
    for t in range(1, T + 1):
        mu_1[t] = beta * mu_1[t-1] + rng.normal(0, np.sqrt(q_1))
        mu_0[t] = beta * mu_0[t-1] + rng.normal(0, np.sqrt(q_0))
    tau = mu_1[1:] - mu_0[1:]

    pi_seq = np.zeros(T); Y_1 = np.zeros(T); Y_0 = np.zeros(T)
    # Pooled within-period variance: accumulate SS and df
    SS_1 = 0.0; SS_0 = 0.0; df_1 = 0; df_0 = 0
    sigma_hat_1 = sigma_1; sigma_hat_0 = sigma_0  # init for burn-in

    for t in range(T):
        # --- Allocation decision (uses past data only) ---
        if t < T0:
            pi_t = 0.5
        elif allocation_rule == 'fixed':
            pi_t = 0.5
        elif allocation_rule == 'neyman':
            pi_t = sigma_hat_1 / (sigma_hat_0 + sigma_hat_1)
        elif allocation_rule == 'corrected':
            pi_t = compute_model_adjusted_pi(sigma_hat_1, sigma_hat_0,
                                              q_1, q_0, beta, beta, n)
        else:
            pi_t = 0.5
        pi_t = np.clip(pi_t, 0.05, 0.95)
        pi_seq[t] = pi_t

        # --- Generate individual-level data ---
        n_1_t = max(1, int(round(n * pi_t)))
        n_0_t = max(1, n - n_1_t)
        Y_ind_1 = mu_1[t+1] + rng.normal(0, sigma_1, size=n_1_t)
        Y_ind_0 = mu_0[t+1] + rng.normal(0, sigma_0, size=n_0_t)

        # Batch means (what the Kalman filter/smoother sees)
        Y_1[t] = np.mean(Y_ind_1)
        Y_0[t] = np.mean(Y_ind_0)

        # --- Update pooled within-period variance estimator ---
        SS_1 += np.sum((Y_ind_1 - Y_1[t])**2)
        SS_0 += np.sum((Y_ind_0 - Y_0[t])**2)
        df_1 += n_1_t - 1
        df_0 += n_0_t - 1
        if df_1 > 1:
            sigma_hat_1 = np.sqrt(SS_1 / df_1)
        if df_0 > 1:
            sigma_hat_0 = np.sqrt(SS_0 / df_0)

    # --- Post-experiment: smoother estimation ---
    naive_running = np.cumsum(Y_1 - Y_0) / np.arange(1, T+1)
    # Use estimated sigma_hat for r_avg (no oracle knowledge)
    r_1_avg = np.mean(sigma_hat_1**2 / (n * pi_seq))
    r_0_avg = np.mean(sigma_hat_0**2 / (n * (1 - pi_seq)))
    ms1, _, _, _ = kalman_smoother(Y_1, beta, q_1, r_1_avg, mu_prior=mu_init_1, P_prior=1.0)
    ms0, _, _, _ = kalman_smoother(Y_0, beta, q_0, r_0_avg, mu_prior=mu_init_0, P_prior=1.0)
    tau_running = np.cumsum(tau) / np.arange(1, T+1)
    return {
        'tau': tau, 'tau_running': tau_running,
        'naive_running': naive_running,
        'smoother_estimate': np.mean(ms1 - ms0),
        'mu_smooth_1': ms1, 'mu_smooth_0': ms0,
        'pi_seq': pi_seq, 'Y_1': Y_1, 'Y_0': Y_0,
        'r_1_avg': r_1_avg, 'r_0_avg': r_0_avg,
        'sigma_hat_1': sigma_hat_1, 'sigma_hat_0': sigma_hat_0,
    }


def generate_figures(beta, q_1, q_0, sigma2_1, sigma2_0,
                     mu_init_1, mu_init_0, n, T, T0,
                     out_dir, N_REPS=500):
    """Generate all figures for a given beta value.

    Global visual scheme:
      Color = estimator:  red=truth, blue=naive, green=smoother
      Linetype = allocation: dashed=Neyman, solid=Model-adjusted
      No theoretical/oracle lines.
    """
    os.makedirs(out_dir, exist_ok=True)
    sigma_1, sigma_0 = np.sqrt(sigma2_1), np.sqrt(sigma2_0)

    # -------------------------------------------------------
    # Figure: Allocation Probability (no Fixed, 2 rules only)
    # -------------------------------------------------------
    print(f"  Figure: Allocation probabilities...")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for rule, ls, label in [('neyman', '--', 'Classical Neyman'),
                             ('corrected', '-', 'Model-adjusted')]:
        res = simulate_experiment(beta, q_1, q_0, sigma2_1, sigma2_0,
                                  mu_init_1, mu_init_0, n, T, T0, rule, seed=42)
        ax.plot(range(1, T+1), res['pi_seq'], color='black', label=label,
                linewidth=1.5, linestyle=ls, alpha=0.9)
    ax.set_xlabel('Period $t$')
    ax.set_ylabel(r'$\pi_t$')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir + 'prob_09.png' if 'appendix' in out_dir else out_dir + 'prob_2.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # Figure: Running Average ATE — two SEPARATE figures
    # -------------------------------------------------------
    checkpoints = list(range(9, T, 10)) + [T-1]
    for rule, title, suffix in [('neyman', 'Neyman', 'ney'),
                                 ('corrected', 'Model-adjusted', 'adj')]:
        print(f"  Figure: Running average ATE ({title})...")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        res = simulate_experiment(beta, q_1, q_0, sigma2_1, sigma2_0,
                                  mu_init_1, mu_init_0, n, T, T0, rule, seed=42)
        t_range = np.arange(1, T+1)
        sm_running = np.full(T, np.nan)
        for t_end in checkpoints:
            ms1, _, _, _ = kalman_smoother(res['Y_1'][:t_end+1], beta, q_1, res['r_1_avg'],
                                            mu_prior=mu_init_1, P_prior=1.0)
            ms0, _, _, _ = kalman_smoother(res['Y_0'][:t_end+1], beta, q_0, res['r_0_avg'],
                                            mu_prior=mu_init_0, P_prior=1.0)
            sm_running[t_end] = np.mean(ms1 - ms0)
        ax.plot(t_range, res['tau_running'], color='red', ls='-', linewidth=1.5,
                label=r'True $\bar{\tau}_t$')
        valid = ~np.isnan(sm_running)
        ax.plot(t_range[valid], sm_running[valid], color='green', ls='-',
                linewidth=1.5, label='Smoother')
        ax.plot(t_range, res['naive_running'], color='blue', ls='--',
                linewidth=1.0, alpha=0.7, label='Naive')
        ax.set_title(title)
        ax.set_xlabel('Period $t$')
        ax.set_ylabel(r'$\bar{\tau}_t$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if 'appendix' in out_dir:
            fname = f'ATE_{suffix}_09.png'
        else:
            fname = f'ATE_{suffix}.png'
        plt.savefig(out_dir + fname, dpi=150, bbox_inches='tight')
        print(f"    Saved {fname}")
        plt.close()

    # -------------------------------------------------------
    # Figure: MC Standard Deviation (1 combined panel, no theory)
    # Color=estimator, linetype=allocation
    # -------------------------------------------------------
    print(f"  Figure: MC standard deviation (combined)...")
    checkpoints_sd = list(range(9, T, 10)) + [T-1]
    fig, ax = plt.subplots(figsize=(6, 4))

    for rule, ls, alloc_label in [('neyman', '--', 'Neyman'),
                                   ('corrected', '-', 'Model-adj.')]:
        naive_ests = np.zeros((N_REPS, T))
        sm_ests_cp = np.zeros((N_REPS, len(checkpoints_sd)))
        true_runs = np.zeros((N_REPS, T))

        for rep in range(N_REPS):
            res = simulate_experiment(beta, q_1, q_0, sigma2_1, sigma2_0,
                                      mu_init_1, mu_init_0, n, T, T0, rule, seed=rep)
            naive_ests[rep] = res['naive_running']
            true_runs[rep] = res['tau_running']
            for ci, t_end in enumerate(checkpoints_sd):
                ms1, _, _, _ = kalman_smoother(res['Y_1'][:t_end+1], beta, q_1,
                                               res['r_1_avg'], mu_prior=mu_init_1, P_prior=1.0)
                ms0, _, _, _ = kalman_smoother(res['Y_0'][:t_end+1], beta, q_0,
                                               res['r_0_avg'], mu_prior=mu_init_0, P_prior=1.0)
                sm_ests_cp[rep, ci] = np.mean(ms1 - ms0)

        t_range = np.arange(1, T+1)
        naive_errors = naive_ests - true_runs
        naive_sd = np.std(naive_errors, axis=0) * np.sqrt(t_range)

        sm_sd_cp = np.zeros(len(checkpoints_sd))
        t_cp = np.array(checkpoints_sd) + 1
        for ci, t_end in enumerate(checkpoints_sd):
            true_tau_cp = true_runs[:, t_end]
            sm_sd_cp[ci] = np.std(sm_ests_cp[:, ci] - true_tau_cp) * np.sqrt(t_end + 1)

        # Plot: color=estimator, linetype=allocation
        ax.plot(t_range[9:], naive_sd[9:], color='blue', ls=ls, linewidth=1.5,
                label=f'Naive ({alloc_label})')
        ax.plot(t_cp, sm_sd_cp, color='green', ls=ls, linewidth=1.5,
                marker='o' if ls == '-' else 's', markersize=3,
                label=f'Smoother ({alloc_label})')
        print(f"    {rule}: naive_sd_final={naive_sd[-1]:.4f}, sm_sd_final={sm_sd_cp[-1]:.4f}")

    ax.set_xlabel('Period $t$')
    ax.set_ylabel(r'$\sqrt{t}\times\mathrm{SD}$')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = 'sd_09.png' if 'appendix' in out_dir else 'sd_2.png'
    plt.savefig(out_dir + fname, dpi=150, bbox_inches='tight')
    print(f"    Saved {fname}")
    plt.close()


def generate_ci_figure(beta, q_1, q_0, sigma2_1, sigma2_0,
                       mu_init_1, mu_init_0, n, T, T0, out_path):
    """Generate confidence sequence figure."""
    print(f"  Figure: Confidence sequence...")
    res = simulate_experiment(beta, q_1, q_0, sigma2_1, sigma2_0,
                              mu_init_1, mu_init_0, n, T, T0, 'neyman', seed=42)
    t_range = np.arange(1, T+1)
    alpha = 0.05; rho_cs = 0.5

    # Naive CS
    r_1_avg = res['r_1_avg']; r_0_avg = res['r_0_avg']
    sigma2_cs = r_1_avg + r_0_avg
    naive_hw = np.zeros(T)
    for t in range(T):
        tt = t + 1
        S = tt * sigma2_cs
        naive_hw[t] = np.sqrt(sigma2_cs) * np.sqrt(
            2 * (S * rho_cs**2 + 1) / (tt**2 * rho_cs**2) *
            np.log(np.sqrt(S * rho_cs**2 + 1) / alpha))

    # Smoother CS at checkpoints every 10
    V_smooth = compute_oracle_variance(beta, q_1, r_1_avg) + \
               compute_oracle_variance(beta, q_0, r_0_avg)
    sm_centers = np.full(T, np.nan); sm_hw = np.full(T, np.nan)
    checkpoints = list(range(9, T, 10)) + [T-1]
    for t_end in checkpoints:
        ms1, _, _, _ = kalman_smoother(res['Y_1'][:t_end+1], beta, q_1, r_1_avg,
                                        mu_prior=mu_init_1, P_prior=1.0)
        ms0, _, _, _ = kalman_smoother(res['Y_0'][:t_end+1], beta, q_0, r_0_avg,
                                        mu_prior=mu_init_0, P_prior=1.0)
        tt = t_end + 1
        sm_centers[t_end] = np.mean(ms1 - ms0)
        sm_hw[t_end] = np.sqrt(V_smooth / tt) * np.sqrt(
            2 * (1 + 1/(tt*rho_cs**2)) * np.log(np.sqrt(tt*rho_cs**2+1)/alpha))

    fig, ax = plt.subplots(figsize=(6, 4))
    start = 20
    valid = ~np.isnan(sm_centers)
    # True effect: red solid
    ax.plot(t_range[start:], res['tau_running'][start:], color='red', ls='-', lw=1.5,
            label=r'True $\bar{\tau}_t$')
    # Naive CS: blue shaded + blue dashed center
    ax.fill_between(t_range[start:], res['naive_running'][start:] - naive_hw[start:],
                     res['naive_running'][start:] + naive_hw[start:],
                     color='blue', alpha=0.15, label='Naive CS')
    ax.plot(t_range[start:], res['naive_running'][start:], color='blue', ls='--',
            lw=1, alpha=0.6)
    # Smoother CS: green shaded + green solid center
    t_v = t_range[valid]; mask = t_v > start
    ax.fill_between(t_v[mask], sm_centers[valid][mask] - sm_hw[valid][mask],
                     sm_centers[valid][mask] + sm_hw[valid][mask],
                     color='green', alpha=0.2, label='Smoother CS')
    ax.plot(t_v[mask], sm_centers[valid][mask], color='green', ls='-', lw=1.5)
    ax.set_xlabel('Period $t$')
    ax.set_ylabel(r'$\bar{\tau}_t$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    Saved {os.path.basename(out_path)}")
    plt.close()


def generate_typeI_figure(beta, q_1, q_0, sigma2_1, sigma2_0,
                          mu_init_1, mu_init_0, n, T, T0, out_path, N_REPS=500):
    """Generate cumulative miscoverage plot under continuous monitoring (peeking).

    For each MC replication, we track whether the CI has EVER failed to cover
    the true tau_bar up to time t. The cumulative miscoverage at t is the
    fraction of replications where coverage was violated at any time <= t.
    This is the correct "peeking" diagnostic: a valid CS should stay below alpha.
    """
    print(f"  Figure: Cumulative miscoverage (peeking)...")
    alpha = 0.05; rho_cs = 0.5
    start_t = 10  # start checking from period 10

    # For each rep, track whether naive CI / smoother CS has ever failed
    # up to each time t. Shape: (N_REPS, T)
    naive_ever_failed = np.zeros((N_REPS, T), dtype=bool)
    cs_ever_failed = np.zeros((N_REPS, T), dtype=bool)

    checkpoints = list(range(9, T, 10)) + [T-1]
    checkpoint_set = set(checkpoints)

    for rep in range(N_REPS):
        res = simulate_experiment(beta, q_1, q_0, sigma2_1, sigma2_0,
                                  mu_init_1, mu_init_0, n, T, T0, 'neyman', seed=rep)
        r_1_avg = res['r_1_avg']; r_0_avg = res['r_0_avg']
        sigma2_cs = r_1_avg + r_0_avg
        V_smooth = compute_oracle_variance(beta, q_1, r_1_avg) + \
                   compute_oracle_variance(beta, q_0, r_0_avg)

        # Naive CLT-based CI: check at every t, track if EVER failed
        naive_failed_so_far = False
        for t in range(T):
            tt = t + 1
            naive_se = np.sqrt(sigma2_cs / tt)
            hw = 1.96 * naive_se
            if abs(res['naive_running'][t] - res['tau_running'][t]) > hw:
                naive_failed_so_far = True
            naive_ever_failed[rep, t] = naive_failed_so_far

        # Smoother CS: check at checkpoints, track if EVER failed
        cs_failed_so_far = False
        last_cp = -1
        for t in range(T):
            if t in checkpoint_set:
                ms1, _, _, _ = kalman_smoother(res['Y_1'][:t+1], beta, q_1, r_1_avg,
                                                mu_prior=mu_init_1, P_prior=1.0)
                ms0, _, _, _ = kalman_smoother(res['Y_0'][:t+1], beta, q_0, r_0_avg,
                                                mu_prior=mu_init_0, P_prior=1.0)
                tt = t + 1
                sm_est = np.mean(ms1 - ms0)
                hw = np.sqrt(V_smooth / tt) * np.sqrt(
                    2 * (1 + 1/(tt*rho_cs**2)) * np.log(np.sqrt(tt*rho_cs**2+1)/alpha))
                if abs(sm_est - res['tau_running'][t]) > hw:
                    cs_failed_so_far = True
            cs_ever_failed[rep, t] = cs_failed_so_far

    # Cumulative miscoverage: fraction of reps where CI ever failed up to t
    naive_cum_miscover = np.mean(naive_ever_failed, axis=0)
    cs_cum_miscover = np.mean(cs_ever_failed, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    t_range = np.arange(1, T+1)
    ax.plot(t_range[start_t:], naive_cum_miscover[start_t:], color='blue', ls='--',
            lw=1.5, label='Naive CI (peeking)')
    ax.plot(t_range[start_t:], cs_cum_miscover[start_t:], color='green', ls='-',
            lw=1.5, label='Smoother CS (peeking)')
    ax.axhline(y=alpha, color='red', ls=':', lw=1.5,
               label=rf'Nominal $\alpha={alpha}$')
    ax.set_xlabel('Period $t$')
    ax.set_ylabel('Cumulative miscoverage rate')
    ax.set_ylim(-0.01, max(0.25, naive_cum_miscover[-1] + 0.05))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    Saved {os.path.basename(out_path)}")
    print(f"      Naive peeking miscover at T: {naive_cum_miscover[-1]:.3f}")
    print(f"      CS peeking miscover at T: {cs_cum_miscover[-1]:.3f}")
    plt.close()


# ============================================================
# Example trajectory figures (Figures 1 and 2 in the paper)
# ============================================================

def generate_example_figures(out_dir):
    """Generate Figures 1-2: example AR(1) and AR(1)+seasonal treatment effect trajectories."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    T = 200

    # --- Figure 1: AR(1) latent state ---
    print("  Figure: AR(1) example trajectory...")
    beta = 0.9; q_1 = 0.015; q_0 = 0.01
    mu_1 = np.zeros(T+1); mu_0 = np.zeros(T+1)
    mu_1[0] = 5.0; mu_0[0] = 3.0
    for t in range(1, T+1):
        mu_1[t] = beta * mu_1[t-1] + rng.normal(0, np.sqrt(q_1))
        mu_0[t] = beta * mu_0[t-1] + rng.normal(0, np.sqrt(q_0))
    tau = mu_1[1:] - mu_0[1:]
    tau_running = np.cumsum(tau) / np.arange(1, T+1)
    t_range = np.arange(1, T+1)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_range, tau, color='black', ls='-', lw=1.0, alpha=0.6, label=r'$\tau_t$')
    ax.plot(t_range, tau_running, color='red', ls='--', lw=1.5, label=r'$\bar{\tau}_t$')
    ax.set_xlabel('Period $t$')
    ax.set_ylabel('Treatment effect')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir + 'case1_tau.png', dpi=150, bbox_inches='tight')
    print("    Saved case1_tau.png")
    plt.close()

    # --- Figure 2: AR(1) + seasonal ---
    print("  Figure: AR(1) + seasonal example trajectory...")
    rng2 = np.random.RandomState(123)
    beta_s = 0.9; q_level = 0.002; q_seas = 0.001; S = 14
    level_1 = np.zeros(T+1); level_0 = np.zeros(T+1)
    level_1[0] = 4.0; level_0[0] = 2.0
    # Initialize seasonal components with larger amplitude for visibility
    seas_1 = np.zeros(T+S); seas_0 = np.zeros(T+S)
    seas_1[:S] = 0.5 * np.sin(2*np.pi*np.arange(S)/S) + rng2.normal(0, 0.05, S)
    seas_1[:S] -= seas_1[:S].mean()
    seas_0[:S] = 0.3 * np.sin(2*np.pi*np.arange(S)/S + 1.0) + rng2.normal(0, 0.05, S)
    seas_0[:S] -= seas_0[:S].mean()
    for t in range(1, T+1):
        level_1[t] = beta_s * level_1[t-1] + rng2.normal(0, np.sqrt(q_level))
        level_0[t] = beta_s * level_0[t-1] + rng2.normal(0, np.sqrt(q_level))
        seas_1[t+S-1] = -np.sum(seas_1[t:t+S-1]) + rng2.normal(0, np.sqrt(q_seas))
        seas_0[t+S-1] = -np.sum(seas_0[t:t+S-1]) + rng2.normal(0, np.sqrt(q_seas))
    mu_1_s = level_1[1:] + seas_1[S:]
    mu_0_s = level_0[1:] + seas_0[S:]
    tau_s = mu_1_s - mu_0_s
    tau_s_running = np.cumsum(tau_s) / np.arange(1, T+1)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_range, tau_s, color='black', ls='-', lw=1.0, alpha=0.6, label=r'$\tau_t$')
    ax.plot(t_range, tau_s_running, color='red', ls='--', lw=1.5, label=r'$\bar{\tau}_t$')
    ax.set_xlabel('Period $t$')
    ax.set_ylabel('Treatment effect')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir + 'case2_tau.png', dpi=150, bbox_inches='tight')
    print("    Saved case2_tau.png")
    plt.close()

print("=" * 50)
print("Generating example trajectory figures")
print("=" * 50)
generate_example_figures('src/')

# ============================================================
# Run for beta=0.7 (main paper)
# ============================================================
print("=" * 50)
print("Generating figures for beta=0.7 (main paper)")
print("=" * 50)
beta = 0.7
q_0, q_1 = 0.01, 0.015
sigma2_0, sigma2_1 = 1.0, 16.0
sigma_0, sigma_1 = np.sqrt(sigma2_0), np.sqrt(sigma2_1)
mu_init_0, mu_init_1 = 3.0, 8.0
n = 30; T = 200; T0 = 10

OUT_DIR = 'src/'
generate_figures(beta, q_1, q_0, sigma2_1, sigma2_0,
                 mu_init_1, mu_init_0, n, T, T0, OUT_DIR, N_REPS=500)

pi_neyman = sigma_1 / (sigma_1 + sigma_0)
r_1_ney = sigma2_1 / (n * pi_neyman)
r_0_ney = sigma2_0 / (n * (1 - pi_neyman))
V_naive_theory = r_1_ney + r_0_ney
V_smooth_theory = (compute_oracle_variance(beta, q_1, r_1_ney) +
                   compute_oracle_variance(beta, q_0, r_0_ney))
generate_ci_figure(beta, q_1, q_0, sigma2_1, sigma2_0,
                   mu_init_1, mu_init_0, n, T, T0, OUT_DIR + 'CI.png')
generate_typeI_figure(beta, q_1, q_0, sigma2_1, sigma2_0,
                      mu_init_1, mu_init_0, n, T, T0,
                      'src/appendix/typeI_07.png', N_REPS=500)

print(f"\nTheory (beta={beta}):")
print(f"  V_Naive = {V_naive_theory:.6f}, sqrt = {np.sqrt(V_naive_theory):.4f}")
print(f"  V_Smooth = {V_smooth_theory:.6f}, sqrt = {np.sqrt(V_smooth_theory):.4f}")
print(f"  Efficiency gain: {(1 - V_smooth_theory/V_naive_theory)*100:.1f}%")

# ============================================================
# Run for beta=0.9 (appendix)
# ============================================================
print("\n" + "=" * 50)
print("Generating figures for beta=0.9 (appendix)")
print("=" * 50)
beta09 = 0.9
OUT_DIR_APP = 'src/appendix/'
generate_figures(beta09, q_1, q_0, sigma2_1, sigma2_0,
                 mu_init_1, mu_init_0, n, T, T0, OUT_DIR_APP, N_REPS=500)

generate_ci_figure(beta09, q_1, q_0, sigma2_1, sigma2_0,
                   mu_init_1, mu_init_0, n, T, T0, OUT_DIR_APP + 'CI_09.png')
generate_typeI_figure(beta09, q_1, q_0, sigma2_1, sigma2_0,
                      mu_init_1, mu_init_0, n, T, T0,
                      OUT_DIR_APP + 'typeI_09.png', N_REPS=500)

V_naive09 = r_1_ney + r_0_ney  # same Neyman allocation
V_smooth09 = (compute_oracle_variance(beta09, q_1, r_1_ney) +
              compute_oracle_variance(beta09, q_0, r_0_ney))
print(f"\nTheory (beta={beta09}):")
print(f"  V_Naive = {V_naive09:.6f}, sqrt = {np.sqrt(V_naive09):.4f}")
print(f"  V_Smooth = {V_smooth09:.6f}, sqrt = {np.sqrt(V_smooth09):.4f}")
print(f"  Efficiency gain: {(1 - V_smooth09/V_naive09)*100:.1f}%")

print("\nDone! All figures saved.")
