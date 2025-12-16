# -*- coding: gbk -*-
import os, sys, argparse
import re
import hashlib

# ===================================================================
# 1. Parallel Configuration (Before importing numpy/scipy/statsmodels)
# ===================================================================

def _preparse_int(argv, keys, default):
    """ Helper to extract integer args before full argparse """
    val = None
    for i, a in enumerate(argv):
        if a in keys and i+1 < len(argv):
            try:
                val = int(argv[i+1])
            except Exception:
                pass
    return val if val is not None else default

DEFAULT_JOBS = max(1, int(0.9 * (os.cpu_count() or 4)))
DEFAULT_BLAS_THREADS = 1

JOBS = _preparse_int(sys.argv, ("--jobs","-j"), DEFAULT_JOBS)
BLAS_THREADS = _preparse_int(sys.argv, ("--blas-threads",), DEFAULT_BLAS_THREADS)

for var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[var] = str(BLAS_THREADS)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from lifelines import CoxPHFitter
from joblib import Parallel, delayed
import joblib
from contextlib import contextmanager

from statsmodels.stats.mediation import Mediation
from statsmodels.duration.hazard_regression import PHReg as _SM_PHReg

from patsy import bs  # noqa: F401

# --- FIXED TQDM FALLBACK ---
try:
    from tqdm.auto import tqdm
except ImportError:
    # Fix: Define a dummy class that handles both iterables and manual 'total' usage
    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        
        def __iter__(self):
            # If used as a loop wrapper: for i in tqdm(range(...))
            if self.iterable is not None:
                yield from self.iterable
            else:
                return iter([])

        def update(self, n=1):
            # If used with joblib: tqdm_object.update()
            pass
        
        def close(self):
            pass
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

@contextmanager
def tqdm_joblib(tqdm_object):
    """ Tqdm callback for Joblib """
    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

# ===================================================================
# 2. Global Configuration & Variables
# ===================================================================

INPUT_FILE = "/home/data/laiyuxuan/wc_pro_Batch.csv"  #pro
INPUT_SEP = ","

PROTEIN_START_COL_INDEX_1_BASED = 2
PROTEIN_END_COL_INDEX_1_BASED = 2921

WC_VARS = ["WC"]

CONTINUOUS_COVS = ['Age', 'TDI', 'Time_TV', 'MET Physical activity', 'sample_age_days','Fasting_time','BMI']
DISCRETE_COVS = [
    'Sex', 'Smoking status', 'Alcohol intake frequency',
    'Oily fish intake', 'Processed meat intake', 'Education',
    'Fruit intake', 'vegetable intake', 'Red meat intake',
    'Ethnic background', 'UK Biobank assessment centre | Instance 0','Batch'
]
ALL_COVS = CONTINUOUS_COVS + DISCRETE_COVS

ZHEN_DIAG_COL = "zhen_need_diagnosis"
ZHEN_TIME_COL = "zhen_ten_need_time"

DIAG_COL = "new_diagnosis_after_baseline"
DIAG_TIME_COL = "time_new_diagnosis_after_baseline"

DEATH_ICD_COL = "newp_s_alldead"
DEATH_DATE_COL = "new2025516_dead_data"
BASELINE_COL = "date_attending_assessment_centre"
CENSOR_DATE = pd.to_datetime("2024-07-08")
EVENT_COL = "event"

DISEASE_LIST_FILE = "/home/data/laiyuxuan/categorized_disease.csv"
DISEASE_CODE_COLUMN = "Disease_Code"

MEDIATION_SIMS = 2000
MEDIATION_ALPHA = 0.05
MEDIATION_SEED = 42
MEDIATION_TREAT_MODE = "sd"     # "sd" or "delta"
MEDIATION_TREAT_DELTA = 1.0     # only used when treat_mode == "delta"
MEDIATION_MIN_EVENTS = 10
MEDIATION_SCALE = True

# ===================================================================
# 3. Helper Functions (Sanitize unique, Imputation, ICD Matching)
# ===================================================================

def clean_col_name(name):
    if pd.isna(name): return "Unknown"
    name = str(name).strip()
    name = re.sub(r'[ \(\)\-\/\.,]', '_', name)
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name

def _make_unique(names):
    counts = {}
    out = []
    for n in names:
        k = counts.get(n, 0) + 1
        counts[n] = k
        out.append(n if k == 1 else f"{n}_{k}")
    return out

def sanitize_columns_list(cols):
    cleaned = [clean_col_name(c) for c in cols]
    return _make_unique(cleaned)

def sanitize_df_columns(df):
    df.columns = sanitize_columns_list(list(df.columns))
    return df

def resolve_name_from_header(raw_name, header_raw, header_sanitized_unique):
    try:
        idx = header_raw.index(raw_name)
        return header_sanitized_unique[idx]
    except Exception:
        return clean_col_name(raw_name)

def apply_full_header_sanitized(df, header_raw, header_sanitized_unique):
    df.columns = [resolve_name_from_header(c, header_raw, header_sanitized_unique) for c in df.columns]
    return df

def find_site_column(df):
    candidates_raw = ['UK Biobank assessment centre | Instance 0']
    base = clean_col_name(candidates_raw[0])
    if base in df.columns:
        return base
    for c in df.columns:
        if c == base or c.startswith(base + "_"):
            return c
    return None

def _find_by_base(df, base_raw):
    base = clean_col_name(base_raw)
    if base in df.columns:
        return base
    for c in df.columns:
        if c == base or c.startswith(base + "_"):
            return c
    return None

def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Continuous: Median fill
    - TDI: Median fill by Assessment Centre (if available), else global median
    - BMI: Median fill by Sex (if available), else global median   <<< Key change(1)
    - Discrete: Mode fill
    - Ethnic background: Fill with 1
    """
    df = df.copy()

    # ---- Continuous: TDI special + others median ----
    targets = [c for c in CONTINUOUS_COVS if c in df.columns]
    for cov in targets:
        tdi_base = clean_col_name('TDI')
        if cov == tdi_base or cov.startswith(tdi_base + "_"):
            df[cov] = pd.to_numeric(df[cov], errors='coerce')
            site_col = find_site_column(df)
            if site_col and site_col in df.columns:
                site_meds = df.groupby(site_col, dropna=False)[cov].median().to_dict()
                overall = df[cov].median()
                def _fill_tdi(row):
                    v = row[cov]
                    if pd.isna(v):
                        return site_meds.get(row.get(site_col), overall)
                    return v
                df[cov] = df.apply(_fill_tdi, axis=1)
            else:
                df[cov] = df[cov].fillna(df[cov].median())
        else:
            series = pd.to_numeric(df[cov], errors='coerce')
            med = series.median()
            if pd.isna(med): med = 0
            df[cov] = series.fillna(med)

    # ---- BMI: median by Sex (your requested logic, vectorized) ----
    bmi_col = _find_by_base(df, 'BMI')
    sex_col = _find_by_base(df, 'Sex')
    if bmi_col and bmi_col in df.columns:
        df[bmi_col] = pd.to_numeric(df[bmi_col], errors='coerce')
        if sex_col and sex_col in df.columns:
            tmp = df[[bmi_col, sex_col]].copy()
            tmp[sex_col] = pd.to_numeric(tmp[sex_col], errors='coerce')
            bmi_medians = tmp.groupby(sex_col, dropna=False)[bmi_col].median()
            overall_median = tmp[bmi_col].median()
            fill_vals = tmp[sex_col].map(bmi_medians).fillna(overall_median)
            df[bmi_col] = df[bmi_col].fillna(fill_vals)
        else:
            df[bmi_col] = df[bmi_col].fillna(df[bmi_col].median())

    # ---- Discrete ----
    eth_col = _find_by_base(df, 'Ethnic background')
    if eth_col and eth_col in df.columns:
        df[eth_col] = df[eth_col].fillna(1)

    for cov in DISCRETE_COVS:
        cov_s = clean_col_name(cov)
        if cov_s not in df.columns:
            continue
        if df[cov_s].isnull().any():
            try:
                m = df[cov_s].mode(dropna=True)
                mode_val = m.iloc[0] if len(m) > 0 else df[cov_s].dropna().iloc[0]
            except Exception:
                mode_val = 0
            df[cov_s] = df[cov_s].fillna(mode_val)

    return df

def parse_first_date(s):
    if pd.isna(s) or str(s).strip() == "":
        return pd.NaT
    parts = [p.strip() for p in str(s).split("|") if p.strip() != ""]
    if not parts:
        return pd.NaT
    dates = pd.to_datetime(parts, errors="coerce", dayfirst=False, yearfirst=True)
    dates = dates.dropna()
    return dates.min() if len(dates) else pd.NaT

def is_event_match(patient_code_str: str, target_disease_code: str) -> bool:
    if not patient_code_str or not target_disease_code:
        return False
    try:
        code = patient_code_str.strip().upper().replace(".", "")
        target = target_disease_code.strip().upper()
        if not code or not target:
            return False
        return code.startswith(target)
    except Exception:
        return False

def build_survival_event(df_in: pd.DataFrame, target_disease_code: str) -> pd.DataFrame:
    df = df_in.copy()

    base_dt = pd.to_datetime(df[BASELINE_COL], errors="coerce", dayfirst=False, yearfirst=True)
    death_dt = df[DEATH_DATE_COL].apply(parse_first_date)

    censor_dt = pd.Series(CENSOR_DATE, index=df.index)
    mask_death_early = death_dt.notna() & (death_dt < censor_dt)
    censor_dt[mask_death_early] = death_dt[mask_death_early]

    idx_to_exclude = set()
    zhen_diag_series = df[ZHEN_DIAG_COL].fillna("").astype(str).str.split('|')
    zhen_time_series = df[ZHEN_TIME_COL].fillna("").astype(str).str.split('|')

    for idx in df.index:
        codes = zhen_diag_series.loc[idx]
        dates_str = zhen_time_series.loc[idx]
        b_dt = base_dt.loc[idx]

        if pd.isna(b_dt):
            idx_to_exclude.add(idx)
            continue

        dates = pd.to_datetime(dates_str, errors='coerce', dayfirst=False, yearfirst=True)
        for code, dt in zip(codes, dates):
            code_clean = str(code).strip()
            if pd.notna(dt) and code_clean and dt < b_dt:
                if is_event_match(code_clean, target_disease_code):
                    idx_to_exclude.add(idx)
                    break

    first_event_dt = pd.Series(pd.NaT, index=df.index)
    diag_series = df[DIAG_COL].fillna("").astype(str).str.split('|')
    time_series = df[DIAG_TIME_COL].fillna("").astype(str).str.split('|')

    for idx in df.index:
        if idx in idx_to_exclude:
            continue
        codes = diag_series.loc[idx]
        dates_str = time_series.loc[idx]
        b_dt = base_dt.loc[idx]
        dates = pd.to_datetime(dates_str, errors='coerce', dayfirst=False, yearfirst=True)

        valid_event_dts = []
        for code, dt in zip(codes, dates):
            code_clean = str(code).strip()
            if pd.notna(dt) and code_clean and dt > b_dt:
                if is_event_match(code_clean, target_disease_code):
                    valid_event_dts.append(dt)
        if valid_event_dts:
            first_event_dt.loc[idx] = min(valid_event_dts)

    out = df.copy()
    out["event"] = 0
    out["time"] = np.nan

    for idx in df.index:
        if idx in idx_to_exclude:
            continue

        b_dt = base_dt.loc[idx]
        e_dt = first_event_dt.loc[idx]
        c_dt = censor_dt.loc[idx]
        if pd.isna(b_dt) or pd.isna(c_dt):
            continue

        observed_dt = c_dt
        event = 0
        if pd.notna(e_dt) and e_dt <= c_dt:
            observed_dt = e_dt
            event = 1

        time_days_diff = (observed_dt - b_dt)
        if pd.notna(time_days_diff):
            time_days = time_days_diff.total_seconds() / (24.0 * 3600.0)
            if time_days > 0:
                out.loc[idx, "time"] = time_days
                out.loc[idx, "event"] = event

    out = out[out["time"].notna() & (out["time"] > 0)]
    return out

def build_binary_event(df_in: pd.DataFrame, target_disease_code: str) -> pd.DataFrame:
    # Used for Table0 counts
    df = df_in.copy()
    base_dt = pd.to_datetime(df[BASELINE_COL], errors="coerce", dayfirst=False, yearfirst=True)
    death_dt = df[DEATH_DATE_COL].apply(parse_first_date)

    censor_dt = pd.Series(CENSOR_DATE, index=df.index)
    mask_death_early = death_dt.notna() & (death_dt < censor_dt)
    censor_dt[mask_death_early] = death_dt[mask_death_early]

    idx_to_exclude = set()
    zhen_diag_series = df[ZHEN_DIAG_COL].fillna("").astype(str).str.split('|')
    zhen_time_series = df[ZHEN_TIME_COL].fillna("").astype(str).str.split('|')

    for idx in df.index:
        codes = zhen_diag_series.loc[idx]
        dates_str = zhen_time_series.loc[idx]
        b_dt = base_dt.loc[idx]
        if pd.isna(b_dt):
            idx_to_exclude.add(idx)
            continue
        dates = pd.to_datetime(dates_str, errors='coerce', dayfirst=False, yearfirst=True)
        for code, dt in zip(codes, dates):
            code_clean = str(code).strip()
            if pd.notna(dt) and code_clean and dt < b_dt:
                if is_event_match(code_clean, target_disease_code):
                    idx_to_exclude.add(idx)
                    break

    event = pd.Series(0, index=df.index, dtype=float)
    diag_series = df[DIAG_COL].fillna("").astype(str).str.split('|')
    time_series = df[DIAG_TIME_COL].fillna("").astype(str).str.split('|')

    for idx in df.index:
        if idx in idx_to_exclude:
            continue
        codes = diag_series.loc[idx]
        dates_str = time_series.loc[idx]
        b_dt = base_dt.loc[idx]
        c_dt = censor_dt.loc[idx]
        if pd.isna(b_dt) or pd.isna(c_dt):
            continue
        dates = pd.to_datetime(dates_str, errors='coerce', dayfirst=False, yearfirst=True)
        for code, dt in zip(codes, dates):
            code_clean = str(code).strip()
            if pd.notna(dt) and code_clean and dt > b_dt and dt <= c_dt:
                if is_event_match(code_clean, target_disease_code):
                    event.loc[idx] = 1
                    break

    out = df.copy()
    event.loc[list(idx_to_exclude)] = np.nan
    event.loc[base_dt.isna()] = np.nan
    out[EVENT_COL] = event
    return out

# ===================================================================
# 4. Step 1: OLS (WC vs Proteins)
# ===================================================================

def run_step1_ols(input_file, sep, wc_col, protein_indices,
                  all_covs_list, discrete_covs_list, out_file):
    print(f"  [Step 1] Reading Data: {input_file} (for {wc_col})")
    df_raw = pd.read_csv(input_file, sep=sep, dtype=str)
    df_raw = sanitize_df_columns(df_raw)

    wc_col = clean_col_name(wc_col)

    # proteins by index
    pro_start = protein_indices[0] - 1
    pro_end   = protein_indices[1]
    protein_cols = list(df_raw.columns[pro_start:pro_end])
    p_total = len(protein_cols)
    print(f"  [Step 1] Found {p_total} proteins.")

    # keep covs + site(for TDI imputation)
    keep_cols = [wc_col] + [clean_col_name(c) for c in all_covs_list if clean_col_name(c) in df_raw.columns]
    site_col = find_site_column(df_raw)
    if site_col and site_col in df_raw.columns:
        keep_cols.append(site_col)
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df_raw.columns]))

    df_cov = df_raw[keep_cols].copy()

    # --- WC + Covariates to numeric & site kept raw ---
    cont_cols = [c for c in CONTINUOUS_COVS if c in df_cov.columns]
    num_cols = list(dict.fromkeys([wc_col] + cont_cols))
    for c in num_cols:
        df_cov[c] = pd.to_numeric(df_cov[c], errors="coerce")
    # -----------------------------------------------------------------

    # drop missing WC
    df_cov = df_cov[~df_cov[wc_col].isna()].copy()

    # impute covariates
    df_cov = impute_data(df_cov)

    # build design matrix for covariates (excluding WC)
    disc_cols = [clean_col_name(c) for c in discrete_covs_list]
    disc_cols = [c for c in disc_cols if c in df_cov.columns]
    cont_cols = [c for c in cont_cols if c in df_cov.columns]

    X_parts = []
    if cont_cols:
        X_parts.append(df_cov[cont_cols].apply(pd.to_numeric, errors="coerce"))
    if disc_cols:
        X_parts.append(pd.get_dummies(df_cov[disc_cols], prefix=disc_cols, drop_first=True))

    if X_parts:
        X_cov = pd.concat(X_parts, axis=1)
    else:
        X_cov = pd.DataFrame(index=df_cov.index)

    X_cov = X_cov.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    n = df_cov.shape[0]
    C = np.column_stack([np.ones(n, dtype=np.float64), X_cov.to_numpy(np.float64)])
    wc = df_cov[wc_col].to_numpy(np.float64)

    try:
        Q, R = np.linalg.qr(C, mode="reduced")
        wc_coef = np.linalg.solve(R, Q.T @ wc)
        wc_hat = C @ wc_coef
    except Exception:
        wc_coef, *_ = np.linalg.lstsq(C, wc, rcond=None)
        wc_hat = C @ wc_coef

    wc_res = wc - wc_hat
    denom = float(np.dot(wc_res, wc_res))
    if not np.isfinite(denom) or denom <= 0:
        print("  [Step 1][ERROR] WC has zero variance after covariate adjustment.")
        out = pd.DataFrame({"protein": protein_cols, "Beta": np.nan, "SE": np.nan, "P_value": np.nan})
        out["P_bonferroni"] = np.nan
        out["Protein_count"] = p_total
        out.to_csv(out_file, index=False)
        return out_file

    p_full = C.shape[1] + 1
    df_resid = n - p_full
    if df_resid <= 1:
        print("  [Step 1][ERROR] Not enough degrees of freedom.")
        df_resid = max(2, df_resid)

    print("  [Step 1] Processing Proteins...")
    df_pro = df_raw.loc[df_cov.index, protein_cols].apply(pd.to_numeric, errors="coerce")
    medians = df_pro.median(axis=0, skipna=True)
    df_pro = df_pro.fillna(medians).fillna(0.0)

    results = []
    block = 256

    for i in tqdm(range(0, p_total, block), desc=f"Step 1 OLS-matrix ({wc_col})", unit="block"):
        cols_blk = protein_cols[i:i+block]
        Y = df_pro[cols_blk].to_numpy(np.float64)

        try:
            if "Q" in locals():
                B = np.linalg.solve(R, Q.T @ Y)
                Y_hat = C @ B
            else:
                B, *_ = np.linalg.lstsq(C, Y, rcond=None)
                Y_hat = C @ B
        except Exception:
            B, *_ = np.linalg.lstsq(C, Y, rcond=None)
            Y_hat = C @ B

        Y_res = Y - Y_hat

        num = (wc_res[:, None] * Y_res).sum(axis=0)
        beta = num / denom

        resid = Y_res - wc_res[:, None] * beta
        rss = (resid ** 2).sum(axis=0)
        sigma2 = rss / df_resid
        se = np.sqrt(sigma2 / denom)
        tval = beta / se
        pval = 2.0 * stats.t.sf(np.abs(tval), df=df_resid)

        for j, prot in enumerate(cols_blk):
            results.append([prot, float(beta[j]), float(se[j]), float(pval[j])])

    out = pd.DataFrame(results, columns=["protein", "Beta", "SE", "P_value"])
    out["P_bonferroni"] = (out["P_value"] * p_total).clip(upper=1.0)
    out["Protein_count"] = p_total
    out = out.sort_values("P_value").reset_index(drop=True)
    out.to_csv(out_file, index=False)
    print(f"  [Step 1] Saved -> {out_file}")
    return out_file

# ===================================================================
# 5. Step 2: Cox (Protein vs Outcome)  (lifelines)
# ===================================================================

def run_cox_one(prot, df_base, covar_cols):
    try:
        cols = ["time", "event"] + covar_cols + [prot]
        dd = df_base[cols].copy()
        for c in covar_cols + [prot]:
            dd[c] = pd.to_numeric(dd[c], errors='coerce')
        dd = dd.dropna()

        if dd.shape[0] < 50 or dd["event"].sum() < 10:
            return [prot, np.nan, np.nan, np.nan, np.nan]

        cph = CoxPHFitter()
        cph.fit(dd, duration_col="time", event_col="event", show_progress=False)

        if prot not in cph.summary.index:
            return [prot, np.nan, np.nan, np.nan, np.nan]

        row = cph.summary.loc[prot]
        hr   = float(row.get('exp(coef)', np.exp(row['coef'])))
        ci_l = float(row.get('exp(coef) lower 95%', np.exp(row['coef lower 95%'])))
        ci_u = float(row.get('exp(coef) upper 95%', np.exp(row['coef upper 95%'])))
        p    = float(row['p'])
        return [prot, hr, ci_l, ci_u, p]
    except Exception:
        return [prot, np.nan, np.nan, np.nan, np.nan]

def run_step2_cox(input_file, sep, table_a_csv, target_disease_code, all_covs_list, out_file):
    try:
        tableA = pd.read_csv(table_a_csv)
        prot_col = 'protein' if 'protein' in tableA.columns else 'Protein'
        p_col = 'P_bonferroni'
        cand_prots = (tableA.loc[tableA[p_col] < 0.05, prot_col]
                      .dropna().astype(str).tolist())
        cand_prots = list(dict.fromkeys(cand_prots))
    except Exception as e:
        print(f"  [Step 2] Error reading TableA: {e}")
        return None

    if not cand_prots:
        print(f"  [Step 2] No significant proteins in {table_a_csv}. Skipping.")
        return None

    print(f"  [Step 2] Found {len(cand_prots)} candidates.")
    print(f"  [Step 2] Reading Data for {target_disease_code}")
    df_raw = pd.read_csv(input_file, sep=sep, dtype=str)
    df_raw = sanitize_df_columns(df_raw)

    print(f"  [Step 2] Building Survival Data...")
    df_surv = build_survival_event(df_raw, target_disease_code)

    df_surv = impute_data(df_surv)

    present_prots = [p for p in cand_prots if p in df_surv.columns]
    for p in present_prots:
        s = pd.to_numeric(df_surv[p], errors='coerce')
        med = np.nanmedian(s)
        df_surv[p] = np.where(np.isnan(s), med, s) if np.isfinite(med) else s.fillna(0)

    if not present_prots:
        return None

    sanitized_all_covs = [clean_col_name(c) for c in all_covs_list]
    sanitized_discrete = [clean_col_name(c) for c in DISCRETE_COVS]

    keep_cols = set(["time","event"]) | set(sanitized_all_covs) | set(present_prots)
    df_base = df_surv.loc[:, [c for c in df_surv.columns if c in keep_cols]].copy()

    final_covar_cols = []
    for c in sanitized_all_covs:
        if c not in df_base.columns:
            continue
        if c in sanitized_discrete:
            dummies = pd.get_dummies(df_base[c], prefix=c, drop_first=True)
            for d_col in dummies.columns:
                df_base[d_col] = dummies[d_col]
                final_covar_cols.append(d_col)
        else:
            df_base[c] = pd.to_numeric(df_base[c], errors='coerce')
            final_covar_cols.append(c)

    print(f"  [Step 2] Parallel Cox (n_jobs={JOBS}) ...")
    with tqdm_joblib(tqdm(total=len(present_prots), desc=f"Step 2 Cox ({target_disease_code})", unit="prot")):
        results = Parallel(n_jobs=JOBS, backend="loky")(
            delayed(run_cox_one)(p, df_base, final_covar_cols) for p in present_prots
        )

    out = pd.DataFrame(results, columns=["protein", "HR", "CI_low", "CI_high", "P_value"])
    m = max(1, len(present_prots))
    out["P_bonf"] = (out["P_value"] * m).clip(upper=1.0)
    out["HR_95CI"] = out.apply(
        lambda r: f"{r.HR:.3f} ({r.CI_low:.3f}-{r.CI_high:.3f})" if pd.notna(r.HR) else "",
        axis=1
    )
    out = out.sort_values("P_value", na_position="last").reset_index(drop=True)
    out.to_csv(out_file, index=False)
    print(f"  [Step 2] Saved -> {out_file}")
    return out_file

# ===================================================================
# 6. Step 3: Mediation (WC -> Protein -> Outcome)  (PHReg + statsmodels.Mediation)
# ===================================================================

class PHRegPredOnly(_SM_PHReg):
    """
    Override to make PHReg.predict(params, exog) return ndarray directly.
    Statsmodels Mediation.fit expects this, but PHReg usually returns a bunch.
    """
    def predict(self, params, exog=None, cov_params=None, endog=None,
                strata=None, offset=None, pred_type="lhr", pred_only=False):
        return super().predict(
            params, exog=exog, cov_params=cov_params, endog=endog,
            strata=strata, offset=offset, pred_type="lhr", pred_only=True
        )

def _pvalue(vec):
    vec = np.asarray(vec)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        return np.nan
    return 2 * min(np.sum(vec > 0), np.sum(vec < 0)) / float(len(vec))

def mediate_one_cox_sm(prot, dfb, wc_col, cov_cols_numeric,
                       sims, alpha, seed, treat_mode, treat_delta,
                       min_events, scale_mediator):
    """
    Step3: Cox PHReg + statsmodels Mediation
    Effect scale: log-hazard (lhr) differences.
    """
    need_cols = ["time", "event", wc_col] + cov_cols_numeric + [prot]
    dd = dfb.loc[:, [c for c in need_cols if c in dfb.columns]].copy()
    dd = dd.dropna()

    n_eff = int(dd.shape[0])
    if n_eff < 200 or dd["event"].sum() < min_events:
        return [prot] + [np.nan]*12 + [n_eff]

    # ---- mediator z-score (optional) ----
    M = pd.to_numeric(dd[prot], errors="coerce")
    if scale_mediator:
        mu = M.mean()
        sd = M.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return [prot] + [np.nan]*12 + [n_eff]
        M = (M - mu) / sd

    # ---- exposure scaling so that 0 vs 1 == (mean vs mean+1SD) or (mean vs mean+delta) ----
    X_raw = pd.to_numeric(dd[wc_col], errors="coerce")
    mu_x = X_raw.mean()
    if treat_mode == "delta":
        denom = float(treat_delta)
        if not np.isfinite(denom) or denom == 0:
            return [prot] + [np.nan]*12 + [n_eff]
        X = (X_raw - mu_x) / denom
    else:
        sd_x = X_raw.std(ddof=1)
        if not np.isfinite(sd_x) or sd_x == 0:
            return [prot] + [np.nan]*12 + [n_eff]
        X = (X_raw - mu_x) / sd_x

    # numeric covs
    COVS = dd[cov_cols_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # time/event
    T = pd.to_numeric(dd["time"], errors="coerce")
    E = pd.to_numeric(dd["event"], errors="coerce")

    # final assemble
    dat = pd.DataFrame({"X": X, "M": M, "time": T, "event": E}, index=dd.index)
    dat = pd.concat([dat, COVS], axis=1).dropna()

    n_eff = int(dat.shape[0])
    if n_eff < 200 or dat["event"].sum() < min_events:
        return [prot] + [np.nan]*12 + [n_eff]

    try:
        # reproducible per-protein
        h = int(hashlib.md5(prot.encode("utf-8")).hexdigest()[:8], 16)
        np.random.seed(seed + (h % 1_000_000))

        # mediator model: OLS with intercept
        mex = sm.add_constant(dat[["X"] + cov_cols_numeric], has_constant="add")
        mediator_model = sm.OLS(dat["M"].to_numpy(np.float64), mex.to_numpy(np.float64))

        # outcome model: PHReg (NO intercept!)   exog order: [X, M, covs]
        oex = dat[["X", "M"] + cov_cols_numeric].to_numpy(np.float64)
        outcome_model = PHRegPredOnly(dat["time"].to_numpy(np.float64), oex,
                                      status=dat["event"].to_numpy(np.int8),
                                      ties="breslow")

        # exposure positions: outcome exog X is col0; mediator exog has const then X so X is col1
        tx_pos = [0, 1]
        # mediator position in outcome exog: M is col1
        med_pos = 1

        med = Mediation(outcome_model, mediator_model, tx_pos, med_pos).fit(
            method="parametric", n_rep=int(sims)
        )

        # Use "average" effects
        ADE_vec  = np.asarray(med.ADE_avg,  dtype=float)
        ACME_vec = np.asarray(med.ACME_avg, dtype=float)
        PM_vec   = np.asarray(med.prop_med_avg, dtype=float)

        # point estimates: ADE/ACME mean; PM median (same as statsmodels summary behavior)
        ADE_hat  = float(np.nanmean(ADE_vec))
        ACME_hat = float(np.nanmean(ACME_vec))
        PM_hat   = float(np.nanmedian(PM_vec))

        def ci(arr, use_median=False):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.nan, np.nan, np.nan, np.nan
            lo, hi = np.percentile(arr, [100*alpha/2, 100*(1-alpha/2)])
            p = _pvalue(arr)
            est = float(np.median(arr)) if use_median else float(np.mean(arr))
            return est, float(lo), float(hi), float(p)

        _, ade_lo,  ade_hi,  ade_p  = ci(ADE_vec,  use_median=False)
        _, acme_lo, acme_hi, acme_p = ci(ACME_vec, use_median=False)
        _, pm_lo,   pm_hi,   pm_p   = ci(PM_vec,   use_median=True)

        # total effect = ACME + ADE (on lhr scale, averaged)
        TE_hat = float(np.nanmean(np.asarray(med.total_effect, dtype=float))) if hasattr(med, "total_effect") else np.nan
        if not np.isfinite(TE_hat) or TE_hat == 0:
            # PM already computed from statsmodels ratio; keep it
            pass

        return [prot,
                ADE_hat,  ade_lo,  ade_hi,  ade_p,
                ACME_hat, acme_lo, acme_hi, acme_p,
                PM_hat,   pm_lo,   pm_hi,   pm_p,
                n_eff]
    except Exception:
        return [prot] + [np.nan]*12 + [n_eff]

def run_step3_mediation(input_file, sep, table_b_csv, wc_col, target_disease_code,
                        all_covs_list, out_file):
    try:
        tableB = pd.read_csv(table_b_csv)
        prot_col = 'protein' if 'protein' in tableB.columns else 'Protein'
        p_col = 'P_bonf'
        cand_prots = (tableB.loc[tableB[p_col] < 0.05, prot_col]
                      .dropna().astype(str).tolist())
        cand_prots = list(dict.fromkeys(cand_prots))
    except Exception as e:
        print(f"  [Step 3] Error reading TableB: {e}")
        return None

    if not cand_prots:
        print(f"  [Step 3] No significant proteins in {table_b_csv}. Skipping.")
        return None

    print(f"  [Step 3] Reading Data...")
    df_raw = pd.read_csv(input_file, sep=sep, dtype=str)
    df_raw = sanitize_df_columns(df_raw)
    wc_col = clean_col_name(wc_col)

    # >>> survival data Cox time+event <<<
    print(f"  [Step 3] Building Survival Event (time/event) ...")
    df_surv = build_survival_event(df_raw, target_disease_code)
    df_surv = impute_data(df_surv)

    present_prots = [p for p in cand_prots if p in df_surv.columns]
    if not present_prots:
        print("  [Step 3] No candidate proteins found in data columns.")
        return None

    sanitized_all_covs = [clean_col_name(c) for c in all_covs_list]
    sanitized_discrete = [clean_col_name(c) for c in DISCRETE_COVS]

    # proteins numeric + median impute
    print(f"  [Step 3] Converting Proteins to Numeric & Imputing...")
    for p in present_prots:
        s = pd.to_numeric(df_surv[p], errors='coerce')
        medv = np.nanmedian(s)
        df_surv[p] = np.where(np.isnan(s), medv, s) if np.isfinite(medv) else s.fillna(0)

    if wc_col not in df_surv.columns:
        print(f"  [Step 3] Exposure {wc_col} missing.")
        return None

    keep_cols = ["time", "event", wc_col] + sanitized_all_covs + present_prots
    dfb = df_surv.loc[:, [c for c in keep_cols if c in df_surv.columns]].copy()
    dfb[wc_col] = pd.to_numeric(dfb[wc_col], errors='coerce')

    # covariates: one-hot for discrete, numeric for continuous
    final_covar_cols_numeric = []
    for c in sanitized_all_covs:
        if c not in dfb.columns:
            continue
        if c in sanitized_discrete:
            dummies = pd.get_dummies(dfb[c], prefix=c, drop_first=True)
            for d_col in dummies.columns:
                dfb[d_col] = dummies[d_col]
                final_covar_cols_numeric.append(d_col)
        else:
            dfb[c] = pd.to_numeric(dfb[c], errors='coerce')
            final_covar_cols_numeric.append(c)

    print(f"  [Step 3] Parallel Mediation (PHReg + statsmodels.Mediation) (n_jobs={JOBS}) ...")
    with tqdm_joblib(tqdm(total=len(present_prots), desc=f"Step 3 Med (Cox) ({wc_col})", unit="prot")):
        results = Parallel(n_jobs=JOBS, backend="loky")(
            delayed(mediate_one_cox_sm)(
                p, dfb, wc_col, final_covar_cols_numeric,
                sims=MEDIATION_SIMS, alpha=MEDIATION_ALPHA, seed=MEDIATION_SEED,
                treat_mode=MEDIATION_TREAT_MODE, treat_delta=MEDIATION_TREAT_DELTA,
                min_events=MEDIATION_MIN_EVENTS, scale_mediator=MEDIATION_SCALE
            )
            for p in present_prots
        )

    cols = ["protein", "ADE", "ADE_low", "ADE_high", "ADE_p",
            "ACME","ACME_low","ACME_high","ACME_p",
            "PM",  "PM_low",  "PM_high",  "PM_p", "N"]
    out = pd.DataFrame(results, columns=cols)

    m = max(1, len(present_prots))
    out["ACME_p_bonf"] = (out["ACME_p"] * m).clip(upper=1.0)

    def fmt_ci(v, lo, hi):
        if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi): return ""
        return f"{v:.4e} ({lo:.4e},{hi:.4e})"
    def fmt_pm(pm, lo, hi):
        if not np.isfinite(pm) or not np.isfinite(lo) or not np.isfinite(hi): return ""
        return f"{pm*100:.2f}% ({lo*100:.2f}%,{hi*100:.2f}%)"

    out["ADE_str"]  = out.apply(lambda r: fmt_ci(r["ADE"],  r["ADE_low"],  r["ADE_high"]), axis=1)
    out["ACME_str"] = out.apply(lambda r: fmt_ci(r["ACME"], r["ACME_low"], r["ACME_high"]), axis=1)
    out["PM_str"]   = out.apply(lambda r: fmt_pm(r["PM"],   r["PM_low"],   r["PM_high"]), axis=1)

    out = out.sort_values(["ACME_p","ADE_p"], na_position="last").reset_index(drop=True)
    out.to_csv(out_file, index=False)
    print(f"  [Step 3] Saved -> {out_file}")
    return out_file

# ===================================================================
# 7. Main Execution
# ===================================================================

def main():
    global BASELINE_COL, DEATH_DATE_COL, ZHEN_DIAG_COL, ZHEN_TIME_COL, DIAG_COL, DIAG_TIME_COL, DEATH_ICD_COL
    global CONTINUOUS_COVS, DISCRETE_COVS, ALL_COVS, WC_VARS

    print("==========================================================")
    print(" Executing Combined Analysis (Table A, B, C) - [Updated Covariates]")
    print(f" Parallel: n_jobs={JOBS}, blas_threads={BLAS_THREADS}")
    print(f" Input File: {INPUT_FILE}")
    print("==========================================================")

    header_raw = pd.read_csv(INPUT_FILE, sep=INPUT_SEP, nrows=0).columns.tolist()
    header_sanitized_unique = sanitize_columns_list(header_raw)

    BASELINE_COL    = resolve_name_from_header(BASELINE_COL,    header_raw, header_sanitized_unique)
    DEATH_DATE_COL  = resolve_name_from_header(DEATH_DATE_COL,  header_raw, header_sanitized_unique)
    ZHEN_DIAG_COL   = resolve_name_from_header(ZHEN_DIAG_COL,   header_raw, header_sanitized_unique)
    ZHEN_TIME_COL   = resolve_name_from_header(ZHEN_TIME_COL,   header_raw, header_sanitized_unique)
    DIAG_COL        = resolve_name_from_header(DIAG_COL,        header_raw, header_sanitized_unique)
    DIAG_TIME_COL   = resolve_name_from_header(DIAG_TIME_COL,   header_raw, header_sanitized_unique)
    DEATH_ICD_COL   = resolve_name_from_header(DEATH_ICD_COL,   header_raw, header_sanitized_unique)

    CONTINUOUS_COVS = [clean_col_name(c) for c in CONTINUOUS_COVS]
    DISCRETE_COVS   = [clean_col_name(c) for c in DISCRETE_COVS]
    ALL_COVS        = CONTINUOUS_COVS + DISCRETE_COVS
    WC_VARS         = [clean_col_name(c) for c in WC_VARS]

    print("\n[--- Loading Disease Definitions ---]")
    try:
        disease_df = pd.read_csv(DISEASE_LIST_FILE, dtype=str)
        disease_df.columns = [c.strip() for c in disease_df.columns]
        if DISEASE_CODE_COLUMN not in disease_df.columns:
            print(f"[ERROR] Column '{DISEASE_CODE_COLUMN}' not found in {DISEASE_LIST_FILE}.")
            return
        OUTCOME_DISEASE_CODES = disease_df[DISEASE_CODE_COLUMN].dropna().unique().tolist()
        OUTCOME_DISEASE_CODES = [str(code).strip() for code in OUTCOME_DISEASE_CODES if str(code).strip()]
        print(f"  Found {len(OUTCOME_DISEASE_CODES)} unique disease codes.")
    except Exception as e:
        print(f"[ERROR] Could not read disease list: {e}")
        return

    print("\n[--- Counting Events per Outcome ---]")
    try:
        cols_to_load_raw = [
            "date_attending_assessment_centre",
            "new2025516_dead_data",
            "zhen_need_diagnosis",
            "zhen_ten_need_time",
            "new_diagnosis_after_baseline",
            "time_new_diagnosis_after_baseline",
        ]
        present_raw = [c for c in cols_to_load_raw if c in header_raw]
        df_counts = pd.read_csv(INPUT_FILE, sep=INPUT_SEP, usecols=present_raw, dtype=str)
        df_counts = apply_full_header_sanitized(df_counts, header_raw, header_sanitized_unique)

        event_count_results = []
        for disease_code in tqdm(OUTCOME_DISEASE_CODES, desc="Counting Events"):
            df_e = build_binary_event(df_counts, disease_code)
            count = int(df_e[EVENT_COL].sum())
            n_valid = int(df_e[EVENT_COL].notna().sum())
            event_count_results.append((disease_code, count, n_valid))

        stats_df = pd.DataFrame(event_count_results, columns=["Disease_Code", "Event_Count", "Valid_N"])
        stats_df.to_csv("Table0_Event_Counts_Specific_Diseases.csv", index=False)
        print("  [Counts] Saved Table0.")
    except Exception as e:
        print(f"[ERROR] Counting failed: {e}")
        return

    step1_results = {}
    step2_results = {}
    protein_indices = (PROTEIN_START_COL_INDEX_1_BASED, PROTEIN_END_COL_INDEX_1_BASED)

    print("\n[--- Starting Step 1: OLS ---]")
    for wc_var in WC_VARS:
        out_file = f"TableA_{wc_var}_vs_Proteins_OLS.csv"
        try:
            path = run_step1_ols(
                INPUT_FILE, INPUT_SEP, wc_var, protein_indices,
                ALL_COVS, DISCRETE_COVS, out_file
            )
            step1_results[wc_var] = path
        except Exception as e:
            print(f"[ERROR] Step 1 ({wc_var}) failed: {e}")

    print("\n[--- Starting Step 2: Cox ---]")
    for wc_var in WC_VARS:
        table_a_file = step1_results.get(wc_var)
        if not table_a_file:
            continue
        for disease_code in OUTCOME_DISEASE_CODES:
            out_file = f"TableB_{wc_var}_vs_{disease_code}_Cox.csv"
            try:
                path = run_step2_cox(
                    INPUT_FILE, INPUT_SEP, table_a_file,
                    target_disease_code=disease_code,
                    all_covs_list=ALL_COVS,
                    out_file=out_file
                )
                if path:
                    step2_results[(wc_var, disease_code)] = path
            except Exception as e:
                print(f"[ERROR] Step 2 ({disease_code}) failed: {e}")

    print("\n[--- Starting Step 3: Mediation (Cox) ---]")
    for wc_var in WC_VARS:
        for disease_code in OUTCOME_DISEASE_CODES:
            table_b_file = step2_results.get((wc_var, disease_code))
            if not table_b_file:
                continue
            out_file = f"TableC_{wc_var}_Mediation_{disease_code}.csv"
            try:
                run_step3_mediation(
                    INPUT_FILE, INPUT_SEP, table_b_file,
                    wc_col=wc_var,
                    target_disease_code=disease_code,
                    all_covs_list=ALL_COVS,
                    out_file=out_file
                )
            except Exception as e:
                print(f"[ERROR] Step 3 ({disease_code}) failed: {e}")

    print("\n==========================================================")
    print(" Analysis Complete.")
    print("==========================================================")

if __name__ == "__main__":
    main()
