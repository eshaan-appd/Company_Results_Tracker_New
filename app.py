# app.py
# Streamlit app: deterministic extraction of Consolidated P&L (Quarter), Balance Sheet, Cash Flow from BSE PDFs
# Replaces LLM parsing to prevent column mis-mapping (e.g., mixing quarter vs. H1).

import os, re, io, time, tempfile, math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
import pandas as pd
import numpy as np
import camelot
from openpyxl import Workbook

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="📈 Listed Company Results Tracker (Deterministic)", layout="wide")
st.title("📈 Listed Company Results Tracker — Deterministic Tables (INR Cr)")

st.caption(
    "Fetch BSE ‘Result’ announcements → parse PDF tables with Camelot → show Consolidated P&L (Quarter), "
    "Balance Sheet, and Cash Flow with unit normalization and tie-out checks."
)

# ===============================
# Small utilities
# ===============================
def _fmt(d: datetime.date) -> str:
    return d.strftime("%Y%m%d")

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

_ILLEGAL_RX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _clean(s: str) -> str:
    return _ILLEGAL_RX.sub('', s) if isinstance(s, str) else s

def _slug(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s or "")).strip("_")
    return (s[:maxlen] if len(s) > maxlen else s) or "file"

# ===============================
# BSE API fetch (Result category)
# ===============================
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

def fetch_bse_announcements_result(start_yyyymmdd: str,
                                   end_yyyymmdd: str,
                                   request_timeout: int = 25) -> pd.DataFrame:
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

    # Variants to bypass occasional BSE API quirks
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
            if "application/json" not in (r.headers.get("content-type","")):
                break
            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)
            if total is None:
                try:
                    total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception:
                    total = None
            if not table:
                break
            params["pageno"] += 1; page += 1; time.sleep(0.25)
            if total and len(rows) >= total:
                break
        if rows:
            all_rows = rows
            break

    if not all_rows:
        return pd.DataFrame()

    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))

    # Filter category == "Result"
    cat_col = _first_col(df, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
    if not cat_col:
        return df
    df2 = df.copy()
    df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
    out = df2.loc[df2["_cat_norm"] == "result"].drop(columns=["_cat_norm"])
    return out

# ===============================
# PDF download
# ===============================
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
    return r.content

# ===============================
# Deterministic PDF parsing (Camelot)
# ===============================
NEEDLE_PNL   = re.compile(r"(statement of (consolidated )?(audited )?financial results|statement of profit|profit\s*&\s*loss|income statement)", re.I)
NEEDLE_BAL   = re.compile(r"(statement of assets|assets\s*&\s*liabilities|balance\s*sheet)", re.I)
NEEDLE_CASH  = re.compile(r"(cash\s*flow\s*statement|statement of cash flows)", re.I)
NEEDLE_CONS  = re.compile(r"\bconsolidated\b", re.I)
UNIT_RX      = re.compile(r"(₹|rs\.?)\s*(in)?\s*(crore|cr|lakhs|lakh|mn|million|bn|billion)", re.I)

def _detect_unit_multiplier(text: str) -> float:
    m = UNIT_RX.search(text or "")
    if not m: return 1.0
    unit = m.group(3).lower()
    return {
        "crore": 1.0, "cr": 1.0,
        "lakhs": 0.01, "lakh": 0.01,
        "mn": 0.1, "million": 0.1,
        "bn": 100.0, "billion": 100.0
    }.get(unit, 1.0)

def _parse_number(x):
    if x is None:
        return np.nan
    s = str(x).strip().replace("\u2212","-").replace(",","")
    if s in ["","-","—","NA","N.A.","Not Applicable","not applicable"]:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("(","").replace(")","")
    try:
        v = float(s)
        return -v if neg else v
    except:
        return np.nan

def _camelot_tables(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = f.name
    try:
        t = camelot.read_pdf(path, pages="all", flavor="lattice")
        if t.n == 0: raise RuntimeError("no lattice tables")
    except Exception:
        t = camelot.read_pdf(path, pages="all", flavor="stream")
    return t

def _unit_multiplier_from_all(tables) -> float:
    blob = " ".join(" ".join(tb.df.astype(str).values.ravel()) for tb in tables)
    return _detect_unit_multiplier(blob)

def _choose_table(tables, rx_header):
    picks = []
    for tb in tables:
        text = " ".join(tb.df.astype(str).values.ravel())
        if rx_header.search(text):
            picks.append(tb.df)
    if not picks:
        return None
    for df in picks:
        text = " ".join(df.astype(str).values.ravel())
        if NEEDLE_CONS.search(text):
            return df
    return picks[0]

def _find_header_row(df):
    # pick a row that contains "Particulars" or looks like "Particulars | As at ..."
    for i in range(min(8, len(df))):
        rowtxt = " ".join(df.iloc[i].astype(str).values).lower()
        if "particular" in rowtxt or "as at" in rowtxt or "as at" in rowtxt.replace("."," "):
            return i
    return 0

def _numeric_columns(df):
    cols = []
    for c in df.columns[1:]:
        nums = sum(pd.notna(df[c].map(_parse_number)))
        if nums >= max(3, int(0.15*len(df))):
            cols.append(c)
    return cols

def _pick_quarter_columns(df_raw):
    """
    For P&L: choose the first three numeric columns after 'Particulars',
    preferring those whose category (previous row above header) says 'Quarter ended'
    (if that row exists). Fallback to first three numeric columns.
    """
    df = df_raw.copy()
    hdr_idx = _find_header_row(df)
    # try to capture category row above
    cat_row = df.iloc[hdr_idx-1] if hdr_idx > 0 else None
    df.columns = df.iloc[hdr_idx].tolist()
    df = df.iloc[hdr_idx+1:].reset_index(drop=True)
    first_col = df.columns[0]
    df.rename(columns={first_col: "Particulars"}, inplace=True)

    # drop empty cols; find numeric ones
    num_cols = _numeric_columns(df)

    # preference for "Quarter ended" if cat_row exists
    def _is_quarter(c):
        if cat_row is None: return False
        s = str(cat_row.get(c, "")) if isinstance(cat_row, pd.Series) else ""
        s = str(s).lower()
        return ("quarter" in s) or ("3 months" in s) or ("three months" in s)

    quarter_cols = [c for c in num_cols if _is_quarter(c)]
    pick = (quarter_cols[:3] if len(quarter_cols) >= 3 else num_cols[:3])

    # build user-friendly date labels
    labels = []
    for c in pick:
        s = str(c)
        m = re.search(r"(\d{1,2}[-/ ]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/ ]?\d{2,4})", s, re.I)
        labels.append(m.group(1) if m else s)
    return df[["Particulars"] + pick].copy(), labels

def _to_markdown(df: pd.DataFrame) -> str:
    # Replace NaN with blank for a clean table
    return df.fillna("").to_markdown(index=False)

def _xlsx_bytes(sheets: dict) -> bytes:
    # sheets: {sheet_name: DataFrame}
    with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as xw:
        for name, frame in sheets.items():
            if isinstance(frame, pd.DataFrame):
                frame.to_excel(xw, name[:31], index=False)
            else:
                # write small meta dicts
                pd.DataFrame([frame]).to_excel(xw, name[:31], index=False)
        xw.book.save(xw._io)
        return xw._io.getvalue()

# ===============================
# High-level parse for 3 statements
# ===============================
def extract_financials_from_pdf(pdf_bytes: bytes) -> dict:
    tables = _camelot_tables(pdf_bytes)
    if tables.n == 0:
        raise RuntimeError("No tables detected in PDF.")
    unit_mult = _unit_multiplier_from_all(tables)

    # ---- P&L (Quarter) ----
    pnl_df_raw = _choose_table(tables, NEEDLE_PNL)
    pnl_md = None
    pnl_issues = []
    pnl_sheet = None
    pnl_labels = []
    if pnl_df_raw is not None:
        pnl_quarters, pnl_labels = _pick_quarter_columns(pnl_df_raw)
        # coerce
        for c in pnl_quarters.columns[1:]:
            pnl_quarters[c] = pnl_quarters[c].map(_parse_number) * unit_mult

        def _find(label_regex):
            ix = pnl_quarters["Particulars"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
            hit = ix.str.contains(label_regex, case=False, regex=True, na=False)
            return pnl_quarters.loc[hit].head(1)

        def _val(label, col):
            df = _find(label)
            if df.empty: return np.nan
            v = df.iloc[0][col]
            return float(v) if pd.notna(v) else np.nan

        latest, prev, yoy = pnl_quarters.columns[1], pnl_quarters.columns[2], pnl_quarters.columns[3]
        # totals & checks
        rev_c = _val(r"^revenue\s*from\s*operations", latest)
        oth_c = _val(r"^other\s*income", latest)
        tin_c = _val(r"^total\s*income", latest)
        if all(pd.notna([rev_c, oth_c, tin_c])) and abs((rev_c+oth_c) - tin_c) > max(0.5, 0.005*max(abs(tin_c),1)):
            pnl_issues.append(f"P&L: Revenue ({rev_c:.2f}) + Other income ({oth_c:.2f}) != Total income ({tin_c:.2f})")

        # computed EBITDA = PBT + Dep + Fin
        pbt_c = _val(r"(profit|loss)\s*before\s*tax|pbt", latest)
        dep_c = _val(r"depreciation|amortis", latest)
        fin_c = _val(r"finance\s*cost", latest)
        ebitda_c = np.nan if any(pd.isna([pbt_c, dep_c, fin_c])) else pbt_c + dep_c + fin_c

        def _pct(curr, base):
            if pd.isna(curr) or pd.isna(base) or base == 0:
                return np.nan
            return (curr - base) / abs(base) * 100.0

        # also compute EBITDA for prev/yoy if possible (for margins/%Δ)
        pbt_p = _val(r"(profit|loss)\s*before\s*tax|pbt", prev)
        dep_p = _val(r"depreciation|amortis", prev)
        fin_p = _val(r"finance\s*cost", prev)
        ebitda_p = np.nan if any(pd.isna([pbt_p, dep_p, fin_p])) else pbt_p + dep_p + fin_p

        pbt_y = _val(r"(profit|loss)\s*before\s*tax|pbt", yoy)
        dep_y = _val(r"depreciation|amortis", yoy)
        fin_y = _val(r"finance\s*cost", yoy)
        ebitda_y = np.nan if any(pd.isna([pbt_y, dep_y, fin_y])) else pbt_y + dep_y + fin_y

        def _row(label_regex, nice_label):
            c = _val(label_regex, latest); p = _val(label_regex, prev); y = _val(label_regex, yoy)
            return [nice_label,
                    None if pd.isna(c) else round(c,2),
                    None if pd.isna(p) else round(p,2),
                    None if pd.isna(y) else round(y,2),
                    None if any(pd.isna([c,y])) else round(_pct(c,y),1),
                    None if any(pd.isna([c,p])) else round(_pct(c,p),1)]

        out = []
        out.append(_row(r"^revenue\s*from\s*operations", "Revenue from operations"))
        out.append(_row(r"^other\s*income", "Other income"))
        out.append(_row(r"^total\s*income", "Total income"))
        out.append(_row(r"materials\s*consumed", "Cost of materials consumed"))
        out.append(_row(r"stock-?in-?trade", "Purchases of stock-in-trade"))
        out.append(_row(r"changes\s*in\s*inventor", "Changes in inventories of FG/WIP/stock-in-trade"))
        out.append(_row(r"employee\s*benefit", "Employee benefits expense"))
        out.append(_row(r"^other\s*expenses$", "Other expenses"))

        # EBITDA + margin
        def _margin(val_num, base_rev):
            if pd.isna(val_num) or pd.isna(base_rev) or base_rev == 0:
                return np.nan
            return (val_num/base_rev)*100.0

        m_curr = _margin(ebitda_c, rev_c)
        m_prev = _margin(ebitda_p, _val(r"^revenue\s*from\s*operations", prev))
        m_yoy  = _margin(ebitda_y, _val(r"^revenue\s*from\s*operations", yoy))

        def _bps(a,b):
            if pd.isna(a) or pd.isna(b): return np.nan
            return round((a-b)*100)

        out.append([
            "EBITDA (computed)",
            None if pd.isna(ebitda_c) else round(ebitda_c,2),
            None if pd.isna(ebitda_p) else round(ebitda_p,2),
            None if pd.isna(ebitda_y) else round(ebitda_y,2),
            None if any(pd.isna([ebitda_c, ebitda_y])) else round(_pct(ebitda_c, ebitda_y),1),
            None if any(pd.isna([ebitda_c, ebitda_p])) else round(_pct(ebitda_c, ebitda_p),1)
        ])

        out.append([
            "EBITDA margin (%)",
            None if pd.isna(m_curr) else round(m_curr,1),
            None if pd.isna(m_prev) else round(m_prev,1),
            None if pd.isna(m_yoy) else round(m_yoy,1),
            None if any(pd.isna([m_curr, m_yoy])) else f"{_bps(m_curr, m_yoy)} bps",
            None if any(pd.isna([m_curr, m_prev])) else f"{_bps(m_curr, m_prev)} bps",
        ])

        out += [
            _row(r"finance\s*cost", "Finance costs"),
            _row(r"depreciation|amortis", "Depreciation and amortisation expense"),
            _row(r"(profit|loss)\s*before\s*tax|pbt", "Profit before tax (PBT)"),
            _row(r"tax\s*expense|current\s*tax", "Tax expense"),
            _row(r"(profit|loss)\s*after\s*tax|pat", "Profit after tax (PAT)"),
        ]

        md_cols = ["Line item",
                   f"{pnl_labels[0]} (INR Cr)",
                   f"{pnl_labels[1]} (INR Cr)",
                   f"{pnl_labels[2]} (INR Cr)",
                   "%YoY","%QoQ"]

        pnl_sheet = pd.DataFrame(out, columns=md_cols)
        pnl_md = _to_markdown(pnl_sheet)

    # ---- Balance Sheet ----
    bal_df_raw = _choose_table(tables, NEEDLE_BAL)
    bal_md = None
    bal_issues = []
    bal_sheet = None
    if bal_df_raw is not None:
        df = bal_df_raw.copy()
        hdr_idx = _find_header_row(df)
        df.columns = df.iloc[hdr_idx].tolist()
        df = df.iloc[hdr_idx+1:].reset_index(drop=True)
        first_col = df.columns[0]
        df.rename(columns={first_col: "Line item"}, inplace=True)
        # pick first 2 numeric columns (As at dates), sometimes 3 exist (latest, prev FY, prior FY)
        num_cols = _numeric_columns(df)
        pick = num_cols[:2] if len(num_cols) >= 2 else num_cols
        # coerce
        for c in pick:
            df[c] = df[c].map(_parse_number) * unit_mult
        # tie-out
        def _find_row(rx):
            ix = df["Line item"].astype(str).str.lower()
            hit = ix.str.contains(rx, regex=True, na=False)
            return df.loc[hit].head(1)
        ta = _find_row(r"total\s*assets$")
        tel = _find_row(r"total\s*(equity.*liabilit|liabilities\s*&\s*equity)")
        if not ta.empty and not tel.empty:
            for c in pick:
                a = ta.iloc[0][c]; b = tel.iloc[0][c]
                if pd.notna(a) and pd.notna(b):
                    if abs(a - b) > max(0.5, 0.005*max(abs(a),abs(b))):
                        bal_issues.append(f"Balance Sheet mismatch in {c}: Total Assets ({a:.2f}) vs Total Equity & Liabilities ({b:.2f})")
        # keep trimmed sheet
        bal_sheet = df[["Line item"] + pick].copy()
        bal_md = _to_markdown(bal_sheet)

    # ---- Cash Flow ----
    cash_df_raw = _choose_table(tables, NEEDLE_CASH)
    cash_md = None
    cash_issues = []
    cash_sheet = None
    if cash_df_raw is not None:
        df = cash_df_raw.copy()
        hdr_idx = _find_header_row(df)
        df.columns = df.iloc[hdr_idx].tolist()
        df = df.iloc[hdr_idx+1:].reset_index(drop=True)
        first_col = df.columns[0]
        df.rename(columns={first_col: "Line item"}, inplace=True)
        num_cols = _numeric_columns(df)
        pick = num_cols[:3] if len(num_cols) >= 3 else num_cols  # keep up to three period columns
        for c in pick:
            df[c] = df[c].map(_parse_number) * unit_mult

        def _rowval(rx, col):
            hit = df["Line item"].astype(str).str.contains(rx, case=False, regex=True, na=False)
            if not df.loc[hit].empty:
                v = df.loc[hit].iloc[0][col]
                return float(v) if pd.notna(v) else np.nan
            return np.nan

        # sanity: CFO+CFI+CFF = Net change; Opening + Net change = Closing
        for c in pick:
            cfo = _rowval(r"(net\s*cash.*operat|cash\s*flow.*operat)", c)
            cfi = _rowval(r"(net\s*cash.*invest|cash\s*flow.*invest)", c)
            cff = _rowval(r"(net\s*cash.*financ|cash\s*flow.*financ)", c)
            net = _rowval(r"(net\s*increase.*decrease.*cash|net\s*change.*cash)", c)
            if not any(pd.isna([cfo, cfi, cff, net])):
                calc = cfo + cfi + cff
                if abs(calc - net) > max(0.5, 0.02*max(abs(calc),abs(net))):
                    cash_issues.append(f"{c}: CFO({cfo:.2f}) + CFI({cfi:.2f}) + CFF({cff:.2f}) = {calc:.2f} ≠ Net change({net:.2f})")
            opn = _rowval(r"(cash.*at\s*the\s*beginning|opening\s*cash)", c)
            cls = _rowval(r"(cash.*at\s*the\s*end|closing\s*cash)", c)
            if not any(pd.isna([opn, net, cls])):
                if abs(opn + net - cls) > max(0.5, 0.02*max(abs(opn+net),abs(cls))):
                    cash_issues.append(f"{c}: Opening({opn:.2f}) + Net({net:.2f}) ≠ Closing({cls:.2f})")

        cash_sheet = df[["Line item"] + pick].copy()
        cash_md = _to_markdown(cash_sheet)

    return {
        "unit_multiplier_to_INR_crore": unit_mult,
        "pnl_sheet": pnl_sheet, "pnl_md": pnl_md, "pnl_issues": pnl_issues, "pnl_labels": pnl_labels,
        "bal_sheet": bal_sheet, "bal_md": bal_md, "bal_issues": bal_issues,
        "cash_sheet": cash_sheet, "cash_md": cash_md, "cash_issues": cash_issues,
    }

# ===============================
# UI controls
# ===============================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    # Your test case: 23-Oct-2025
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)
    max_workers = st.slider("Parallel extracts", 1, 8, 3)
    max_items   = st.slider("Max announcements", 1, 200, 50)
    run = st.button("🚀 Fetch & Extract (Deterministic)", type="primary")

# ===============================
# Run
# ===============================
def _pick_company_cols(df: pd.DataFrame) -> tuple[str, str]:
    nm = _first_col(df, ["SLONGNAME","SNAME","SC_NAME","COMPANYNAME"]) or "SLONGNAME"
    subcol = _first_col(df, ["SUBCATEGORYNAME","SUBCATEGORY","SUB_CATEGORY","NEWS_SUBCATEGORY"]) or "SUBCATEGORYNAME"
    return nm, subcol

def worker(idx, row, urls):
    pdf_bytes, used_url = None, ""
    for u in urls:
        try:
            data = _download_pdf(u, timeout=25)
            if data and len(data) > 500:
                pdf_bytes, used_url = data, u
                break
        except Exception:
