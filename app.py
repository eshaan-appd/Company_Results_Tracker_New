# app.py
# Streamlit app that extracts Consolidated P&L (Quarter), Balance Sheet, and Cash Flow
# from BSE PDFs using ONLY OpenAI (no Camelot/Ghostscript). The model returns strict JSON
# via a function tool; we verify tallies locally and render markdown + Excel.

import os, re, io, time, json, tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openai import OpenAI

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="📈 BSE Results — OpenAI-only Extractor", layout="wide")
st.title("📈 BSE Results — OpenAI-only Extractor (INR Cr)")

st.caption(
    "Fetch BSE ‘Result’ announcements → upload each PDF to OpenAI → model returns strict JSON for "
    "Consolidated P&L (Quarter), Balance Sheet, and Cash Flow. We re-check tallies locally."
)

# ===============================
# Small utilities
# ===============================
def _fmt_date(d: datetime.date) -> str:
    return d.strftime("%Y%m%d")

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _slug(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s or "")).strip("_")
    return (s[:maxlen] if len(s) > maxlen else s) or "file"

def _xlsx_bytes(sheets: dict) -> bytes:
    with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as xw:
        for name, frame in sheets.items():
            if isinstance(frame, pd.DataFrame):
                frame.to_excel(xw, name[:31], index=False)
            else:
                pd.DataFrame([frame]).to_excel(xw, name[:31], index=False)
        xw.book.save(xw._io)
        return xw._io.getvalue()

def _to_markdown(df: pd.DataFrame) -> str:
    return df.fillna("").to_markdown(index=False)

def _pct(curr, base):
    if pd.isna(curr) or pd.isna(base) or base == 0:
        return np.nan
    return (curr - base) / abs(base) * 100.0

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
# OpenAI client and schema/tools
# ===============================
def _client():
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY (env var or Streamlit secrets).")
        st.stop()
    return OpenAI(api_key=key)

def _schema():
    # Pure JSON Schema used as function.parameters
    return {
      "type": "object",
      "properties": {
        "meta": {
          "type": "object",
          "properties": {
            "source_url": {"type": "string"},
            "company": {"type": "string"},
            "basis": {"type": "string", "enum": ["Consolidated","Standalone"]},
            "unit_detected": {"type": "string"},
            "unit_to_inr_crore": {"type": "number"},
            "periods": {
              "type": "object",
              "properties": {
                "pnl": {
                  "type": "object",
                  "properties": {
                    "latest_label": {"type": "string"},
                    "prev_label":   {"type": "string"},
                    "yoy_label":    {"type": "string"}
                  },
                  "required": ["latest_label","prev_label","yoy_label"]
                },
                "balance": {
                  "type": "object",
                  "properties": {
                    "asof_latest": {"type": "string"},
                    "asof_prev":   {"type": "string"},
                    "asof_yoy":    {"type": "string"}
                  },
                  "required": ["asof_latest","asof_prev","asof_yoy"]
                },
                "cashflow": {
                  "type": "object",
                  "properties": {
                    "basis": {"type": "string", "enum": ["quarter","ytd","h1","9m","year"]},
                    "latest_label": {"type": "string"},
                    "prev_label":   {"type": "string"},
                    "yoy_label":    {"type": "string"}
                  },
                  "required": ["basis","latest_label"]
                }
              },
              "required": ["pnl","balance","cashflow"]
            }
          },
          "required": ["source_url","basis","unit_detected","unit_to_inr_crore","periods"]
        },
        "tables": {
          "type": "object",
          "properties": {
            "pnl": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "latest": {"type": "number"},
                  "prev":   {"type": ["number","null"]},
                  "yoy":    {"type": ["number","null"]},
                  "cell_refs": {
                    "type": "object",
                    "properties": {
                      "latest": {"type": "string"},
                      "prev":   {"type": "string"},
                      "yoy":    {"type": "string"}
                    }
                  }
                },
                "required": ["name","latest"]
              }
            },
            "balance": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type":"string"},
                  "asof_latest": {"type":"number"},
                  "asof_prev":   {"type":["number","null"]},
                  "asof_yoy":    {"type":["number","null"]},
                  "cell_refs": {"type":"object"}
                },
                "required": ["name","asof_latest"]
              }
            },
            "cashflow": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type":"string"},
                  "latest": {"type":"number"},
                  "prev":   {"type":["number","null"]},
                  "yoy":    {"type":["number","null"]},
                  "cell_refs": {"type":"object"}
                },
                "required": ["name","latest"]
              }
            }
          },
          "required": ["pnl"]
        },
        "checks": {
          "type": "object",
          "properties": {
            "pnl": {"type":"array","items":{"type":"string"}},
            "balance": {"type":"array","items":{"type":"string"}},
            "cashflow": {"type":"array","items":{"type":"string"}}
          },
          "required": ["pnl","balance","cashflow"]
        },
        "notes": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["meta","tables","checks"]
    }

def _tool_args_from_response(resp, tool_name: str):
    """
    Works across SDK versions: convert the Response to plain JSON and
    pull arguments from the tool call with the given name.
    """
    try:
        data = json.loads(resp.model_dump_json())
    except Exception:
        data = resp

    # Responses API shape: output -> [ message ] -> content -> [ parts... ]
    out = data.get("output") or []
    for msg in out:
        for part in msg.get("content", []) or []:
            if part.get("type") == "tool_call" and part.get("name") == tool_name:
                args = part.get("arguments")
                if isinstance(args, str):
                    try:
                        return json.loads(args)
                    except Exception:
                        pass
                return args

    # Some older variants nest under "tool_calls"
    for msg in out:
        for part in msg.get("content", []) or []:
            tool_calls = part.get("tool_calls") or []
            for tc in tool_calls:
                if tc.get("name") == tool_name:
                    args = tc.get("arguments")
                    if isinstance(args, str):
                        return json.loads(args)
                    return args
    raise RuntimeError("Tool call arguments not found in response.")

def _openai_extract(pdf_bytes: bytes, source_url: str, company_hint: str, model_name: str):
    client = _client()

    # Upload PDF for the model to read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        file_obj = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")

    schema = _schema()

    tool_name = "submit_financial_extraction"
    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                "Return the extracted statements using the provided schema. "
                "HARD RULES:\n"
                "• P&L must use ONLY the three 'Quarter ended' columns: latest, previous quarter, same quarter last year.\n"
                "• Balance Sheet must use 'As at' columns (latest, previous, YoY if present).\n"
                "• Cash Flow must state its basis (quarter/ytd/h1/9m/year) and keep periods consistent.\n"
                "• Detect units (₹ in Crore/Lakh/Million/Billion) and normalize to INR Crore.\n"
                "• Provide cell_refs like 'row label ~ column header' for traceability.\n"
                "• Run your own checks; put messages into checks.pnl/balance/cashflow."
            ),
            "parameters": schema
        }
    }]

    system_msg = (
        "You are a meticulous financial table extractor. Output only via the function tool call. "
        "Never include prose. Never mix quarter columns with H1/YTD/Year. "
        "Prefer CONSOLIDATED; if a consolidated section is missing but standalone exists, set basis='Standalone' for that section."
    )

    user_payload = {"source_url": source_url, "company_hint": company_hint or ""}

    resp = client.responses.create(
        model=model_name,
        temperature=0.1,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": tool_name}},
        input=[{
            "role": "system",
            "content": system_msg
        },{
            "role": "user",
            "content": [
                {"type": "input_text", "text": json.dumps(user_payload)},
                {"type": "input_file", "file_id": file_obj.id}
            ]
        }]
    )

    # Extract validated JSON args
    args = _tool_args_from_response(resp, tool_name)
    return args  # dict matching the schema

# ===============================
# Build display tables from JSON
# ===============================
def _build_pnl_table(json_obj):
    meta = json_obj["meta"]
    periods = meta["periods"]["pnl"]
    cols = [
        "Line item",
        f"{periods['latest_label']} (INR Cr)",
        f"{periods['prev_label']} (INR Cr)",
        f"{periods['yoy_label']} (INR Cr)",
        "%YoY","%QoQ"
    ]
    rows = []
    def _get(name):
        for r in json_obj["tables"]["pnl"]:
            if r["name"].strip().lower() == name.strip().lower():
                return r
        return None

    base_order = [
        "Revenue from operations",
        "Other income",
        "Total income",
        "Cost of materials consumed",
        "Purchases of stock-in-trade",
        "Changes in inventories of FG/WIP/stock-in-trade",
        "Employee benefits expense",
        "Other expenses",
        "Finance costs",
        "Depreciation and amortisation expense",
        "Profit before tax (PBT)",
        "Tax expense",
        "Profit after tax (PAT)"
    ]
    seen = set()
    def add_row(r):
        if not r: return
        seen.add(r["name"].lower())
        cur, prev, yoy = r.get("latest"), r.get("prev"), r.get("yoy")
        rows.append([
            r["name"],
            None if cur is None else round(cur, 2),
            None if prev is None else round(prev, 2),
            None if yoy is None else round(yoy, 2),
            None if (cur is None or yoy is None or yoy == 0) else round(_pct(cur, yoy), 1),
            None if (cur is None or prev is None or prev == 0) else round(_pct(cur, prev), 1)
        ])

    for name in base_order:
        add_row(_get(name))
    for r in json_obj["tables"]["pnl"]:
        if r["name"].lower() not in seen:
            add_row(r)

    return pd.DataFrame(rows, columns=cols)

def _build_balance_table(json_obj):
    meta = json_obj["meta"]
    periods = meta["periods"]["balance"]
    cols = [
        "Line item",
        f"As at {periods['asof_latest']} (INR Cr)",
        f"As at {periods['asof_prev']} (INR Cr)",
        f"As at {periods['asof_yoy']} (INR Cr)"
    ]
    rows = []
    for r in json_obj.get("tables", {}).get("balance", []):
        rows.append([
            r["name"],
            None if r.get("asof_latest") is None else round(r.get("asof_latest"), 2),
            None if r.get("asof_prev") is None else round(r.get("asof_prev"), 2),
            None if r.get("asof_yoy") is None else round(r.get("asof_yoy"), 2),
        ])
    return pd.DataFrame(rows, columns=cols) if rows else None

def _build_cashflow_table(json_obj):
    meta = json_obj["meta"]
    periods = meta["periods"]["cashflow"]
    cols = [
        "Line item",
        f"{periods.get('latest_label','Latest')} (INR Cr)",
        f"{periods.get('prev_label','Prev')} (INR Cr)",
        f"{periods.get('yoy_label','YoY')} (INR Cr)"
    ]
    rows = []
    for r in json_obj.get("tables", {}).get("cashflow", []):
        rows.append([
            r["name"],
            None if r.get("latest") is None else round(r.get("latest"), 2),
            None if r.get("prev") is None else round(r.get("prev"), 2),
            None if r.get("yoy") is None else round(r.get("yoy"), 2),
        ])
    return pd.DataFrame(rows, columns=cols) if rows else None, periods.get("basis","")

def _local_checks(json_obj):
    """Re-run key tallies locally for extra safety; return list of warnings."""
    warns = []

    unit_mult = float(json_obj["meta"].get("unit_to_inr_crore", 1.0))
    if unit_mult <= 0 or unit_mult > 1000:
        warns.append(f"Unusual unit_to_inr_crore: {unit_mult}")

    pnl = {r["name"].lower(): r for r in json_obj["tables"]["pnl"]}
    rev = pnl.get("revenue from operations", {})
    oth = pnl.get("other income", {})
    tin = pnl.get("total income", {})
    def chk(cur_rev, cur_oth, cur_ti):
        if cur_rev is None or cur_oth is None or cur_ti is None:
            return
        tol = max(0.5, 0.005*max(abs(cur_ti),1))
        if abs((cur_rev + cur_oth) - cur_ti) > tol:
            warns.append(f"P&L: Revenue({cur_rev}) + Other({cur_oth}) != Total income({cur_ti})")
    chk(rev.get("latest"), oth.get("latest"), tin.get("latest"))

    bal = {r["name"].lower(): r for r in json_obj.get("tables", {}).get("balance", [])}
    ta = bal.get("total assets")
    tel = bal.get("total equity and liabilities") or bal.get("total liabilities and equity")
    if ta and tel:
        for k in ["asof_latest","asof_prev","asof_yoy"]:
            a = ta.get(k); b = tel.get(k)
            if a is None or b is None: continue
            tol = max(0.5, 0.005*max(abs(a),abs(b)))
            if abs(a - b) > tol:
                warns.append(f"Balance Sheet {k}: Total Assets({a}) vs Equity+Liabilities({b})")

    cf_list = json_obj.get("tables", {}).get("cashflow", [])
    cf = {r["name"].lower(): r for r in cf_list}
    cfo = cf.get("net cash from operating activities (cfo)") or cf.get("net cash from operating activities")
    cfi = cf.get("net cash used in investing activities (cfi)") or cf.get("net cash used in investing activities") or cf.get("net cash from investing activities")
    cff = cf.get("net cash from/(used in) financing activities (cff)") or cf.get("net cash from/(used in) financing activities")
    net = cf.get("net increase/(decrease) in cash and cash equivalents") or cf.get("net change in cash and cash equivalents")
    opn = cf.get("cash and cash equivalents at beginning of period") or cf.get("cash and cash equivalents at the beginning of the period")
    cls = cf.get("cash and cash equivalents at end of period") or cf.get("cash and cash equivalents at the end of the period")

    def getv(item, key):
        return None if item is None else item.get(key)

    if all([cfo, cfi, cff, net]):
        calc = lambda key: None if any(v is None for v in [getv(cfo,key),getv(cfi,key),getv(cff,key)]) else (getv(cfo,key)+getv(cfi,key)+getv(cff,key))
        for k in ["latest","prev","yoy"]:
            if getv(net,k) is None: continue
            val = calc(k)
            if val is None: continue
            tol = max(0.5, 0.02*max(abs(val),abs(getv(net,k))))
            if abs(val - getv(net,k)) > tol:
                warns.append(f"Cash Flow {k}: CFO+CFI+CFF={val} vs Net change={getv(net,k)}")

    if all([opn, net, cls]):
        for k in ["latest","prev","yoy"]:
            if getv(opn,k) is None or getv(net,k) is None or getv(cls,k) is None: continue
            tol = max(0.5, 0.02*max(abs(getv(opn,k)+getv(net,k)),abs(getv(cls,k))))
            if abs(getv(opn,k) + getv(net,k) - getv(cls,k)) > tol:
                warns.append(f"Cash bridge {k}: Opening+Net != Closing")

    return warns

# ===============================
# Sidebar controls
# ===============================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    model_name = st.selectbox(
        "OpenAI model",
        ["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"],
        index=1
    )
    max_workers = st.slider("Parallel extracts", 1, 6, 3)
    max_items   = st.slider("Max announcements", 1, 200, 50)
    show_json   = st.checkbox("Show raw JSON")
    run = st.button("🚀 Fetch & Extract (OpenAI-only)", type="primary")

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
            continue
    if not pdf_bytes:
        return idx, used_url, {"error": "Could not fetch a valid PDF."}

    # best-effort company hint
    company = str(row.get(_first_col(pd.DataFrame([row]), ["SLONGNAME","SNAME","SC_NAME","COMPANYNAME"]) or "SLONGNAME") or "").strip()

    try:
        out = _openai_extract(pdf_bytes, used_url, company, model_name)
        return idx, used_url, out
    except Exception as e:
        return idx, used_url, {"error": f"OpenAI extract error: {e}"}

if run:
    start_str, end_str = _fmt_date(start_date), _fmt_date(end_date)
    with st.status("Fetching announcements…", expanded=True):
        hits = fetch_bse_announcements_result(start_str, end_str)
        st.write(f"Matched rows (category='Result'): **{len(hits)}**")
        if hits.empty:
            st.stop()

    if len(hits) > max_items:
        hits = hits.head(max_items)

    nm, subcol = _pick_company_cols(hits)
    rows = []
    for _, r in hits.iterrows():
        urls = _candidate_urls(r)
        rows.append((r, urls))

    st.subheader("📑 Extracted Statements (per filing)")

    def dl_button_bytes(label, data_bytes, file_name, key):
        st.download_button(label, data=data_bytes, file_name=file_name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=key)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, i, r, urls) for i, (r, urls) in enumerate(rows)]
        for fut in as_completed(futs):
            i, pdf_url, result = fut.result()
            r = rows[i][0]
            company = str(r.get(nm) or "").strip()
            dt = str(r.get("NEWS_DT") or "").strip()
            subcat = str(r.get(subcol) or "").strip()
            headline = str(r.get("HEADLINE") or "").strip()

            title = f"{company or 'Unknown'} — {dt}  •  {subcat or 'N/A'}"
            with st.expander(title, expanded=False):
                if headline:
                    st.markdown(f"**Headline:** {headline}")
                if pdf_url:
                    st.markdown(f"[PDF link]({pdf_url})")

                if "error" in result:
                    st.error(result["error"])
                    continue

                # Local checks
                local_warns = _local_checks(result)

                # Build display tables
                pnl_df     = _build_pnl_table(result)
                bal_df     = _build_balance_table(result)
                cash_df, cf_basis = _build_cashflow_table(result)

                st.markdown("### Consolidated Income Statement (Quarter)")
                st.markdown(_to_markdown(pnl_df))

                if bal_df is not None:
                    st.markdown("### Consolidated Balance Sheet (Statement of Assets & Liabilities)")
                    st.markdown(_to_markdown(bal_df))
                else:
                    st.info("Balance Sheet not detected or not disclosed in this PDF.")

                if cash_df is not None:
                    st.markdown(f"### Consolidated Cash Flow Statement  \n*Basis detected: {cf_basis or 'unspecified'}*")
                    st.markdown(_to_markdown(cash_df))
                else:
                    st.info("Cash Flow not detected or not disclosed in this PDF.")

                # Show model's checks + our local warnings
                m_checks = result.get("checks", {})
                if any([m_checks.get("pnl"), m_checks.get("balance"), m_checks.get("cashflow"), local_warns]):
                    st.warning("Validation checks:")
                    for section in ["pnl","balance","cashflow"]:
                        for msg in (m_checks.get(section) or []):
                            st.write(f"- {section.upper()}: {msg}")
                    for msg in local_warns:
                        st.write(f"- LOCAL: {msg}")

                # Download Excel
                meta = {
                    "source_pdf": pdf_url,
                    "basis": result["meta"].get("basis"),
                    "unit_detected": result["meta"].get("unit_detected"),
                    "unit_to_inr_crore": result["meta"].get("unit_to_inr_crore"),
                    "pnl_checks_model": "; ".join(m_checks.get("pnl", [])),
                    "balance_checks_model": "; ".join(m_checks.get("balance", [])),
                    "cashflow_checks_model": "; ".join(m_checks.get("cashflow", [])),
                }
                sheets = {"README": pd.DataFrame([meta]), "Consolidated PnL (Quarter)": pnl_df}
                if bal_df is not None:
                    sheets["Consolidated Balance Sheet"] = bal_df
                if cash_df is not None:
                    sheets["Consolidated Cash Flow"] = cash_df

                xlsx_bytes = _xlsx_bytes(sheets)
                dl_button_bytes("💾 Download Excel", xlsx_bytes,
                                f"{_slug(company)}_{_slug(dt)}.xlsx",
                                key=f"dl_{i}")

                if show_json:
                    st.divider()
                    st.caption("Raw JSON returned by the model")
                    st.json(result)
else:
    st.info("Pick your date range (e.g., **23 Oct 2025**) and click **Fetch & Extract (OpenAI-only)**.")
