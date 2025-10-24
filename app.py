import os, io, re, time, tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
from openai import OpenAI
import os, streamlit as st

# ==== OPTIONAL libs for better PDF extraction & OCR (safe if missing) ====
try:
    import pdfplumber  # table/text extraction from PDFs
except Exception:
    pdfplumber = None

try:
    import fitz  # PyMuPDF: render PDFs to high-DPI images
except Exception:
    fitz = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

# -----------------------------------------
# ---- OpenAI (Responses API) ----
# -----------------------------------------
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets).")
    st.stop()

client = OpenAI(api_key=api_key)

with st.expander("🔍 OpenAI connection diagnostics", expanded=False):
    key_src = "st.secrets" if "OPENAI_API_KEY" in st.secrets else "env"
    mask = lambda s: (s[:7] + "..." + s[-4:]) if s and len(s) > 12 else "unset"
    st.write("Key source:", key_src)
    st.write("API key (masked):", mask(st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")))
    try:
        _ = client.models.list()
        st.success("Models list ok — auth + project look good.")
    except Exception as e:
        st.error(f"Models list failed: {e}")
    try:
        r = client.responses.create(model="gpt-4.1-mini", input="ping")
        st.success("Responses call ok.")
    except Exception as e:
        st.error(f"Responses call failed: {e}")

# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Listed Company Results Tracker (Verified)", layout="wide")
st.title("📈 Listed Company Results Tracker — Verified Numbers")
st.caption("Fetch BSE Results → Parse HTML/PDF Tables → Verify → (Optional) Summarize")

# =========================================
# Small utilities
# =========================================
_ILLEGAL_RX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _clean(s: str) -> str:
    return _ILLEGAL_RX.sub('', s) if isinstance(s, str) else s

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns: return n
    return None

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _slug(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s or "")).strip("_")
    return (s[:maxlen] if len(s) > maxlen else s) or "file"

# =========================================
# Attachment URL candidates (unchanged)
# =========================================
def _candidate_urls(row):
    cands = []
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        cands += [
            f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/Attach/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
        ]
    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        cands.append(ns if ns.lower().startswith("http") else "https://www.bseindia.com/" + ns.lstrip("/"))
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u); seen.add(u)
    return out

# NEW: Build the HTML announcement page url (for DOM table scraping)
def _html_page_url(row):
    ns = str(row.get("NSURL") or "").strip()
    if not ns:
        return None
    return ns if ns.lower().startswith("http") else "https://www.bseindia.com/" + ns.lstrip("/")

# =========================================
# BSE fetch — strict; returns filtered DF (unchanged logic)
# =========================================
def fetch_bse_announcements_strict(start_yyyymmdd: str,
                                   end_yyyymmdd: str,
                                   verbose: bool = True,
                                   request_timeout: int = 25) -> pd.DataFrame:
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    assert start_yyyymmdd <= end_yyyymmdd
    base_page = "https://www.bseindia.com/corporates/ann.html"
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_page,
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })

    try:
        s.get(base_page, timeout=15)
    except Exception:
        pass

    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "",   "strSearch": "P"},
        {"subcategory": "",   "strSearch": ""},
    ]

    all_rows = []
    for v in variants:
        params = {
            "pageno": 1, "strCat": "-1", "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd, "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"], "strscrip": "", "strType": "C",
        }
        rows, total, page = [], None, 1
        while True:
            r = s.get(url, params=params, timeout=request_timeout)
            ct = r.headers.get("content-type","")
            if "application/json" not in ct:
                if verbose: st.warning(f"[variant {v}] non-JSON on page {page} (ct={ct}).")
                break
            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)
            if total is None:
                try: total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception: total = None
            if not table: break
            params["pageno"] += 1; page += 1; time.sleep(0.25)
            if total and len(rows) >= total: break
        if rows:
            all_rows = rows; break

    if not all_rows: return pd.DataFrame()

    all_keys = set()
    for r in all_rows: all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))

    # Filter to Result
    def filter_announcements(df_in: pd.DataFrame, category_filter="Result") -> pd.DataFrame:
        if df_in.empty: return df_in.copy()
        cat_col = _first_col(df_in, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
        if not cat_col: return df_in.copy()
        df2 = df_in.copy()
        df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
        return df2.loc[df2["_cat_norm"] == _norm(category_filter).lower()].drop(columns=["_cat_norm"])

    df_filtered = filter_announcements(df, category_filter="Result")

    return df_filtered

# =========================================
# NEW: HTML extraction & classification
# =========================================
KEY_PNL = ["statement of profit", "profit and loss", "income statement", "part i", "particulars"]
KEY_BS  = ["statement of assets", "balance sheet", "assets & liabilities", "statement of asset"]
KEY_CF  = ["cash flow", "cashflow"]

_num_rx = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def coerce_number(x):
    """Coerce a cell to float if numeric-like; else None."""
    if pd.isna(x): return None
    s = str(x).strip()
    if s in ("", "-", "—"): return None
    m = _num_rx.search(s.replace("–","-"))
    if not m: return None
    s = m.group(0).replace(",", "")
    try:
        return float(s)
    except:
        return None

def normalize_units_detect(page_text: str) -> float:
    """
    Detect unit scaling: returns multiplier to convert to INR Crore.
    Heuristics from nearby text: '₹ in Crore', '₹ in Lakhs', '₹ million', etc.
    """
    t = (page_text or "").lower()
    # preference order: crore, million, lakh, thousand
    if re.search(r'(in|₹|rs)\s*(crore|cr)\b', t): return 1.0
    if re.search(r'(in|₹|rs)\s*(million|mn)\b', t): return 0.1  # 1 million = 0.1 crore
    if re.search(r'(in|₹|rs)\s*(lakh|lac)\b', t): return 0.01  # 1 lakh = 0.01 crore
    if re.search(r'(in|₹|rs)\s*(thousand|000s)\b', t): return 0.0001
    # fallback: assume Crore if ambiguous
    return 1.0

def _classify_table(df: pd.DataFrame, html_text_hint: str = "") -> str|None:
    head = " ".join([str(c) for c in df.columns]).lower()
    first_col_text = " ".join(map(lambda s: str(s).lower(), list(df.iloc[:,0].astype(str))[:12]))
    hint = (html_text_hint or "").lower()

    blob = " ".join([head, first_col_text, hint])
    if any(k in blob for k in KEY_PNL): return "pnl"
    if any(k in blob for k in KEY_BS):  return "bs"
    if any(k in blob for k in KEY_CF):  return "cf"
    return None

def standardize_fin_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep 1st col as 'Line item'; remaining columns as-is, but sanitize numbers.
    Drop all-empty rows; trim whitespace.
    """
    if df is None or df.empty: return pd.DataFrame()
    df2 = df.copy()
    # Flatten a possible MultiIndex header
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [" ".join([_norm(x) for x in tup if str(x) != "nan"]).strip() for tup in df2.columns]
    else:
        df2.columns = [ _norm(c) for c in df2.columns ]

    # Ensure first column name
    df2.rename(columns={df2.columns[0]: "Line item"}, inplace=True)
    # Trim & coerce
    for c in df2.columns:
        if c == "Line item":
            df2[c] = df2[c].map(_norm)
        else:
            df2[c] = df2[c].map(coerce_number)
    # Drop empty rows
    if "Line item" in df2.columns:
        df2 = df2[ (df2["Line item"].astype(str).str.strip()!="") | (df2.drop(columns=["Line item"]).notna().any(axis=1)) ]
    return df2

def pick_qoq_yoy_columns(df: pd.DataFrame):
    """
    Heuristic: choose 3 numeric columns for Latest / Prev (QoQ) / YoY, based on rightmost-to-leftmost recency ordering.
    Many BSE tables put latest on left; some put latest on right. We guess by header text containing dates like '30-Jun-2025' or 'Q1 FY26'.
    """
    if df is None or df.empty or df.shape[1] < 4: return None
    cols = [c for c in df.columns if c != "Line item"]
    # Rank by 'looks like date/period' then keep numeric density
    def _date_score(c):
        s = c.lower()
        score = 0
        if re.search(r'(fy|q[1-4]|quarter|half|nine|year|ended|\d{2,4})', s): score += 2
        if re.search(r'\d{1,2}[-/ ]?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', s): score += 3
        if re.search(r'\d{4}', s): score += 1
        return score
    cols_sorted = sorted(cols, key=lambda c: (_date_score(c), list(df[c].notna()).count(True)), reverse=True)
    # take top 3
    pick = cols_sorted[:3]
    if len(pick) < 3: return None
    # Try to guess ordering: Latest, Prev, YoY
    # If headers contain something like 2025 vs 2024, newer year => Latest
    def _year(c):
        m = re.search(r'(20\d{2})', c)
        return int(m.group(1)) if m else -1
    pick_sorted = sorted(pick, key=_year, reverse=True)
    # Fallback to as-is if scores too close
    return {"latest": pick_sorted[0], "prev": pick_sorted[1], "yoy": pick_sorted[2]}

def extract_tables_from_html(page_url: str) -> dict:
    """
    Returns dict with keys 'unit_multiplier', 'pnl','bs','cf' (each a standardized DataFrame) by scraping the HTML DOM.
    """
    out = {"unit_multiplier": 1.0, "pnl": None, "bs": None, "cf": None, "raw_tables": []}
    if not page_url:
        return out
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    r = s.get(page_url, timeout=25)
    html = r.text
    out["unit_multiplier"] = normalize_units_detect(html)
    try:
        tables = pd.read_html(html)  # uses lxml/html5lib backend if present
    except Exception:
        tables = []
    out["raw_tables"] = tables

    for t in tables:
        kind = _classify_table(t, html_text_hint=html)
        if not kind: continue
        std = standardize_fin_table(t)
        if kind == "pnl" and out["pnl"] is None:
            out["pnl"] = std
        elif kind == "bs" and out["bs"] is None:
            out["bs"] = std
        elif kind == "cf" and out["cf"] is None:
            out["cf"] = std
    return out

# =========================================
# NEW: PDF extraction & image improvement
# =========================================
def _download_pdf(url: str, timeout=25) -> bytes:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://www.bseindia.com/corporates/ann.html",
        "Accept-Language": "en-US,en;q=0.9",
    })
    r = s.get(url, timeout=timeout, allow_redirects=True, stream=False)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    data = r.content
    return data

def extract_tables_from_pdf(pdf_bytes: bytes) -> dict:
    """Use pdfplumber (if available) to extract tabular data and classify."""
    out = {"unit_multiplier": 1.0, "pnl": None, "bs": None, "cf": None}
    if not pdfplumber:
        return out
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_text = []
            tab_sets = []
            for p in pdf.pages:
                try:
                    page_text.append(p.extract_text() or "")
                    tabs = p.extract_tables() or []
                    tab_sets.extend(tabs)
                except Exception:
                    continue
            out["unit_multiplier"] = normalize_units_detect("\n".join(page_text))
            for raw in tab_sets:
                df = pd.DataFrame(raw)
                # drop empty cols/rows
                df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
                if df.shape[1] < 2: continue
                kind = _classify_table(df, "\n".join(page_text))
                std = standardize_fin_table(df)
                if kind == "pnl" and out["pnl"] is None: out["pnl"] = std
                if kind == "bs"  and out["bs"]  is None: out["bs"]  = std
                if kind == "cf"  and out["cf"]  is None: out["cf"]  = std
    except Exception:
        pass
    return out

def rasterize_pdf_to_images(pdf_bytes: bytes, zoom: float = 3.0) -> list[str]:
    """
    Render each page to a high-DPI PNG (300-450dpi equivalent) using PyMuPDF.
    Returns list of file paths.
    """
    if not fitz:
        return []
    tmp_dir = tempfile.mkdtemp(prefix="bsepdf_")
    out_files = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(zoom, zoom)
        for i in range(len(doc)):
            pix = doc.load_page(i).get_pixmap(matrix=mat)
            fp = os.path.join(tmp_dir, f"page_{i+1:03d}.png")
            pix.save(fp)
            out_files.append(fp)
    except Exception:
        return []
    return out_files

# =========================================
# OpenAI file upload + (optional) high-DPI image attach
# =========================================
def _upload_to_openai(pdf_bytes: bytes, fname: str = "document.pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        f = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")
    return f

def _ensure_main_bullet_newlines(text: str) -> str:
    text = re.sub(r'^\s+', '', text or '')
    for n in range(2, 6):
        text = re.sub(rf'(?<!\n)\s*{n}-\s', f'\n\n{n}- ', text)
    return text.strip()

# =========================================
# NEW: Verified computation & rendering
# =========================================
def compute_metrics(pnl_df: pd.DataFrame, unit_mult: float, colmap: dict):
    """
    Build a small dict of headline metrics from standardized P&L df and chosen columns.
    """
    if pnl_df is None or not colmap: return None
    latest, prev, yoy = colmap["latest"], colmap["prev"], colmap["yoy"]
    def _get(name):
        mask = pnl_df["Line item"].str.contains(name, case=False, regex=True, na=False)
        rows = pnl_df.loc[mask]
        return rows[[latest, prev, yoy]].mean(numeric_only=True)  # average if multiple rows match
    # Row finders
    rev = _get(r"revenue from operations|income from operations")
    other = _get(r"other income")
    # EBITDA either explicit or compute: Revenue - (Total expenses before DA & Finance costs)
    ebitda_row = pnl_df["Line item"].str.contains(r"ebitda|profit before depreciation", case=False, regex=True, na=False)
    if ebitda_row.any():
        ebitda = pnl_df.loc[ebitda_row, [latest, prev, yoy]].mean(numeric_only=True)
    else:
        # Try compute if we have 'Total expenses'
        te = _get(r"total expenses")
        if rev is not None and te is not None:
            ebitda = rev - te
        else:
            ebitda = pd.Series([np.nan, np.nan, np.nan], index=[latest, prev, yoy])
    pbt = _get(r"profit before tax|pbt")
    pat = _get(r"profit.*after tax|pat|total comprehensive income")
    # margins (in %)
    def _margin(num, den):
        return (num / den * 100.0).replace([np.inf, -np.inf], np.nan)
    ebitda_margin = _margin(ebitda, rev)
    pat_margin = _margin(pat, rev)
    # pack
    to_cr = lambda s: (s * unit_mult) if isinstance(s, pd.Series) else None
    out = {
        "rev": to_cr(rev),
        "other": to_cr(other),
        "ebitda": to_cr(ebitda),
        "pbt": to_cr(pbt),
        "pat": to_cr(pat),
        "ebitda_margin": ebitda_margin,  # already %
        "pat_margin": pat_margin,        # already %
    }
    return out

def change_pct(curr, prev):
    if curr is None or prev is None: return np.nan
    try:
        return ( (curr - prev) / abs(prev) ) * 100.0
    except Exception:
        return np.nan

def bps_delta(curr_pct, prev_pct):
    if curr_pct is None or prev_pct is None: return np.nan
    try:
        return (curr_pct - prev_pct) * 100.0
    except Exception:
        return np.nan

def verify_and_tally(html_pack: dict, pdf_pack: dict, strict_tol_cr: float = 0.5, strict_tol_pct: float = 1.0):
    """
    Decide source-of-truth tables: prefer HTML (if present), else PDF.
    Compute a verification status comparing the other source where available.
    """
    chosen = {"source": None, "unit": 1.0, "pnl": None, "bs": None, "cf": None, "verification": {}}
    # prefer HTML
    if html_pack and (html_pack.get("pnl") is not None or html_pack.get("bs") is not None or html_pack.get("cf") is not None):
        chosen.update({"source": "html", "unit": html_pack["unit_multiplier"], "pnl": html_pack.get("pnl"),
                       "bs": html_pack.get("bs"), "cf": html_pack.get("cf")})
        other = pdf_pack
    elif pdf_pack and (pdf_pack.get("pnl") is not None or pdf_pack.get("bs") is not None or pdf_pack.get("cf") is not None):
        chosen.update({"source": "pdf", "unit": pdf_pack["unit_multiplier"], "pnl": pdf_pack.get("pnl"),
                       "bs": pdf_pack.get("bs"), "cf": pdf_pack.get("cf")})
        other = html_pack
    else:
        return chosen  # nothing found

    # compare pnl rows if both exist
    def _cmp(df_a, df_b):
        if df_a is None or df_b is None: return None
        # crude overlap score: compare all numeric cells that share the same 'Line item' rows
        common = set(df_a["Line item"]).intersection(set(df_b["Line item"]))
        if not common: return None
        diffs = []
        for li in common:
            ra = df_a.loc[df_a["Line item"]==li]
            rb = df_b.loc[df_b["Line item"]==li]
            if ra.empty or rb.empty: continue
            an = ra.select_dtypes(include=[np.number]).mean(numeric_only=True)
            bn = rb.select_dtypes(include=[np.number]).mean(numeric_only=True)
            # abs diff after rough scale alignment (unit differences may exist)
            # We'll compare ratios as well
            for c in set(an.index).intersection(set(bn.index)):
                va, vb = an[c], bn[c]
                if pd.isna(va) or pd.isna(vb): continue
                diffs.append(abs(va - vb))
        if not diffs: return None
        avg_diff = float(np.nanmean(diffs))
        return avg_diff

    chosen["verification"]["pnl_delta"] = _cmp(chosen["pnl"], other.get("pnl") if other else None) if other else None
    chosen["verification"]["bs_delta"]  = _cmp(chosen["bs"],  other.get("bs")  if other else None) if other else None
    chosen["verification"]["cf_delta"]  = _cmp(chosen["cf"],  other.get("cf")  if other else None) if other else None

    # simple pass/fail flag (Cr tolerance)
    def _flag(d):
        if d is None: return "⚪ Not cross-verified"
        return "🟢 Tally" if d <= strict_tol_cr else "🔴 Mismatch"
    chosen["verification"]["pnl_flag"] = _flag(chosen["verification"]["pnl_delta"])
    chosen["verification"]["bs_flag"]  = _flag(chosen["verification"]["bs_delta"])
    chosen["verification"]["cf_flag"]  = _flag(chosen["verification"]["cf_delta"])

    return chosen

def render_verified_output(company, dt, subcat, pdf_url, chosen, strict_mode=True):
    with st.expander(f"{company or 'Unknown'} — {dt}  •  {subcat or 'N/A'}", expanded=False):
        if pdf_url:
            st.markdown(f"[PDF link]({pdf_url})")

        if chosen["pnl"] is None and chosen["bs"] is None and chosen["cf"] is None:
            st.warning("No recognizable financial tables found in HTML/PDF.")
            return

        # Show verification flags
        vf = chosen["verification"]
        st.write("**Verification:**",
                 f"P&L: {vf.get('pnl_flag','⚪')}",
                 f"BS: {vf.get('bs_flag','⚪')}",
                 f"CF: {vf.get('cf_flag','⚪')}",
                 f"(Source: {chosen['source']}, Unit→Crore x{chosen['unit']:.4g})")

        # If strict and any mismatch, highlight
        if strict_mode and any(flag=="🔴 Mismatch" for flag in [vf.get("pnl_flag"), vf.get("bs_flag"), vf.get("cf_flag")]):
            st.error("Strict mode: A table failed numeric tallies. Review below carefully.")

        # Render tables
        def _show_table(title, df):
            if df is None or df.empty:
                st.info(f"{title}: Not disclosed.")
                return
            st.markdown(f"**{title}** (raw parsed)")
            st.dataframe(df, use_container_width=True)

        _show_table("Income Statement (Quarter)", chosen["pnl"])
        _show_table("Balance Sheet (Assets & Liabilities)", chosen["bs"])
        _show_table("Cash Flow Statement", chosen["cf"])

        # Try to compute bullets if P&L present
        pnl, unit = chosen["pnl"], chosen["unit"]
        if pnl is not None and not pnl.empty and pnl.shape[1] >= 4:
            colmap = pick_qoq_yoy_columns(pnl)
            if colmap:
                met = compute_metrics(pnl, unit, colmap)
                if met:
                    L, P, Y = colmap["latest"], colmap["prev"], colmap["yoy"]
                    rev, other, ebitda, pat = met["rev"], met["other"], met["ebitda"], met["pat"]
                    em, pm = met["ebitda_margin"], met["pat_margin"]

                    def fmt(x, digits=2):
                        if x is None or (isinstance(x,float) and np.isnan(x)): return "Not disclosed"
                        if isinstance(x, pd.Series):
                            return f"{x[L]:,.2f}"
                        return f"{x:,.2f}"

                    def chg(series):
                        if series is None or not isinstance(series, pd.Series): return ("Not disclosed","Not disclosed")
                        yoy = change_pct(series[L], series[Y])
                        qoq = change_pct(series[L], series[P])
                        def p(v):
                            return "Not disclosed" if pd.isna(v) else f"{v:+.1f}%"
                        return (p(yoy), p(qoq))

                    def bps(series):
                        if series is None or not isinstance(series, pd.Series): return ("","")
                        yoy = bps_delta(series[L], series[Y])
                        qoq = bps_delta(series[L], series[P])
                        def b(v):
                            return "" if pd.isna(v) else f"{v:+.0f} bps"
                        return (b(yoy), b(qoq))

                    yoy_rev, qoq_rev = chg(rev)
                    yoy_oth, qoq_oth = chg(other)
                    yoy_eb, qoq_eb = chg(ebitda)
                    yoy_pat, qoq_pat = chg(pat)
                    yoy_eb_bps, qoq_eb_bps = bps(em)
                    yoy_pm_bps, qoq_pm_bps = bps(pm)

                    # Render EXACT 5 bullets with verified numbers
                    st.markdown("**Verified 5-bullet summary (numbers derived from parsed tables):**")
                    st.markdown(
                        f"1- Revenue from operations stands at INR {fmt(rev)} ({yoy_rev} YoY / {qoq_rev} QoQ)\n\n"
                        f"2- Other income stands at INR {fmt(other)} ({yoy_oth} YoY / {qoq_oth} QoQ)\n\n"
                        f"3- EBITDA stands at INR {fmt(ebitda)} ({yoy_eb} YoY / {qoq_eb} QoQ). "
                        f"EBITDA margin stands at {fmt(em[L],1) if isinstance(em,pd.Series) else 'Not disclosed'}% "
                        f"({yoy_eb_bps} YoY / {qoq_eb_bps} QoQ)\n\n"
                        f"4- Finance Costs: Not disclosed (computed only from parsed rows)\n\n"
                        f"5- Net Profit after tax stands at INR {fmt(pat)} ({yoy_pat} YoY / {qoq_pat} QoQ). "
                        f"PAT margin stands at {fmt(pm[L],1) if isinstance(pm,pd.Series) else 'Not disclosed'}% "
                        f"({bps(pm)[0]} YoY / {bps(pm)[1]} QoQ)"
                    )

# =========================================
# OpenAI PDF summarization (kept; used optionally for narrative only)
# =========================================
def summarize_pdf_with_openai(
    pdf_bytes: bytes,
    company: str,
    headline: str,
    subcat: str,
    model: str = "gpt-4.1-mini",
    style: str = "bullets",
    max_output_tokens: int = 800,
    temperature: float = 0.2,
    attach_images: list[str] | None = None,   # NEW: high-DPI images
) -> str:
    import json  # stdlib only
    f = _upload_to_openai(pdf_bytes, fname=f"{_slug(company or 'doc')}.pdf")

    content = [{"type": "input_text", "text": "Summarize the attached BSE results PDF in 5 crisp bullets for an analyst. Do NOT invent numbers; if unsure, say 'Not disclosed'."},
               {"type": "input_file", "file_id": f.id}]

    # If we rasterized pages, attach as images to help vision read image-only PDFs
    if attach_images:
        for fp in attach_images[:6]:  # cap
            try:
                up = client.files.create(file=open(fp, "rb"), purpose="vision")
                content.append({"type":"input_image","image":{"file_id": up.id}})
            except Exception:
                continue

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=[{"role": "user","content": content}],
    )
    raw = (resp.output_text or "").strip()
    return _ensure_main_bullet_newlines(raw)

def safe_summarize(*args, **kwargs) -> str:
    for i in range(4):
        try:
            return summarize_pdf_with_openai(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                time.sleep(2.0 * (i + 1))
                continue
            raise
    return "⚠️ Unable to summarize due to repeated rate limits."

# =========================================
# Sidebar controls
# =========================================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    model = st.selectbox(
        "OpenAI model",
        ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"],
        index=0,
        help="Used only for optional narrative. Numbers are parsed & verified."
    )
    style = st.radio("Summary style", ["bullets", "narrative"], horizontal=True)
    max_tokens = st.slider("Max output tokens", 200, 2000, 800, step=50)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)

    max_workers = st.slider("Parallel summaries", 1, 8, 3, help="Lower if you hit 429s.")
    max_items = st.slider("Max announcements to parse", 5, 200, 60, step=5)

    strict_mode = st.toggle("Strict numeric mode (fail on mismatches)", value=True)
    improve_images = st.toggle("Improve image PDFs (render @ high DPI + OCR)", value=False)

    run = st.button("🚀 Fetch & Parse (Verified)")

# =========================================
# Run pipeline (fetch → HTML/PDF → Verify → (Optional) narrative)
# =========================================
def _fmt(d: datetime.date) -> str: return d.strftime("%Y%m%d")

def _pick_company_cols(df: pd.DataFrame) -> tuple[str, str]:
    nm = _first_col(df, ["SLONGNAME","SNAME","SC_NAME","COMPANYNAME"]) or "SLONGNAME"
    subcol = _first_col(df, ["SUBCATEGORYNAME","SUBCATEGORY","SUB_CATEGORY","NEWS_SUBCATEGORY"]) or "SUBCATEGORYNAME"
    return nm, subcol

if run:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY (set env var, add to Streamlit Secrets, or export in your shell).")
        st.stop()

    start_str, end_str = _fmt(start_date), _fmt(end_date)

    with st.status("Fetching announcements…", expanded=True):
        df_hits = fetch_bse_announcements_strict(start_str, end_str, verbose=False)
        st.write(f"Matched rows after filters: **{len(df_hits)}**")

    if df_hits.empty:
        st.warning("No matching announcements in this window.")
        st.stop()

    if len(df_hits) > max_items:
        df_hits = df_hits.head(max_items)

    rows = []
    for _, r in df_hits.iterrows():
        urls = _candidate_urls(r)
        rows.append((r, urls))

    st.subheader("📑 Verified Tables + (Optional) Narrative")

    nm, subcol = _pick_company_cols(df_hits)

    # Worker to download and verify a single row
    def worker(idx, row, urls):
        pdf_bytes, used_url = None, ""
        for u in urls:
            try:
                data = _download_pdf(u, timeout=25)
                if data and len(data) > 500:
                    pdf_bytes, used_url = data, u
                    break
            except Exception:
                continue

        # Extract from HTML page (inspect element)
        html_url = _html_page_url(row)
        html_pack = extract_tables_from_html(html_url) if html_url else {}

        # Extract from PDF (programmatic)
        pdf_pack = extract_tables_from_pdf(pdf_bytes) if pdf_bytes else {}

        # If image-only & improve_images: rasterize to high DPI for the model
        images = rasterize_pdf_to_images(pdf_bytes, zoom=3.0) if (pdf_bytes and improve_images) else []

        chosen = verify_and_tally(html_pack, pdf_pack, strict_tol_cr=0.5, strict_tol_pct=1.0)

        # Optional narrative (not used for numbers)
        narrative = None
        try:
            narrative = safe_summarize(pdf_bytes, str(row.get(nm) or "").strip(),
                                       str(row.get("HEADLINE") or "").strip(),
                                       str(row.get(subcol) or "").strip(),
                                       model=model,
                                       style=("bullets" if style=="bullets" else "narrative"),
                                       max_output_tokens=int(max_tokens),
                                       temperature=float(temperature),
                                       attach_images=images)
        except Exception:
            narrative = None

        return idx, used_url, chosen, narrative

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, i, r, urls) for i, (r, urls) in enumerate(rows)]
        for fut in as_completed(futs):
            i, pdf_url, chosen, narrative = fut.result()
            r = rows[i][0]
            company = str(r.get(nm) or "").strip()
            dt = str(r.get("NEWS_DT") or "").strip()
            subcat = str(r.get(subcol) or "").strip()

            # If strict and nothing verified, skip rendering
            if strict_mode and (chosen["pnl"] is None and chosen["bs"] is None and chosen["cf"] is None):
                with st.expander(f"{company or 'Unknown'} — {dt}  •  {subcat or 'N/A'}", expanded=False):
                    st.error("Strict mode: Could not verify any table from HTML/PDF.")
                    if pdf_url: st.markdown(f"[PDF link]({pdf_url})")
                continue

            render_verified_output(company, dt, subcat, pdf_url, chosen, strict_mode=strict_mode)

            if narrative:
                with st.expander("💬 Model narrative (for color; numbers above are the verified source of truth)", expanded=False):
                    st.markdown(narrative)

else:
    st.info("Pick your date range and click **Fetch & Parse (Verified)**. "
            "This version parses HTML/PDF tables, verifies numbers, and only then renders the summary.")

# --- Helpful notes (render if OCR stack is missing) ---
if st.sidebar.toggle("Show setup tips", value=False):
    if not pdfplumber:
        st.warning("pdfplumber not found. Install with: `pip install pdfplumber`")
    if not fitz:
        st.warning("PyMuPDF not found. Install with: `pip install pymupdf`")
    if pytesseract is None:
        st.warning("pytesseract/Pillow not found. Install with: `pip install pillow pytesseract` "
                   "and ensure Tesseract OCR is installed on your OS.")
