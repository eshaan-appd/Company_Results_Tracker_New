import os, io, re, json, tempfile
from typing import List, Dict, Any
import requests
import pandas as pd
import numpy as np
import streamlit as st

# High-DPI rendering for image-only PDFs
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from openai import OpenAI

# =========================
# UI
# =========================
st.set_page_config(page_title="OpenAI PDF Tables (High-DPI, JSON-Verified)", layout="wide")
st.title("📈 OpenAI PDF Tables — High-DPI Vision + JSON")
st.caption("Uploads your PDF to OpenAI, renders high-DPI page images, and extracts CONSOLIDATED financial tables in strict JSON. No HTML used.")

with st.sidebar:
    st.header("⚙️ Controls")
    pdf_url = st.text_input("PDF URL (direct link to .pdf)", "")
    file = st.file_uploader("…or upload a PDF", type=["pdf"])
    model = st.selectbox("OpenAI model", ["gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini"], index=0)
    attach_hidpi = st.toggle("Attach high-DPI page images", value=True)
    dpi = st.slider("Render DPI for images", 280, 520, 380, step=20)
    tol_cr = st.number_input("Sanity tolerance (INR Cr)", min_value=0.0, value=1.0, step=0.5)
    run = st.button("🚀 Extract Tables with OpenAI", type="primary")

# =========================
# Helpers
# =========================
_NUM_RX = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def _download_pdf_from_url(url: str, timeout=40) -> bytes:
    s = requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0","Accept":"application/pdf,application/octet-stream,*/*"})
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    if not r.content or len(r.content) < 200:
        raise RuntimeError("Downloaded content too small to be a PDF.")
    return r.content

def _render_pdf_to_images(pdf_bytes: bytes, dpi: int) -> List[str]:
    """Return file paths to rendered PNGs (one per page)."""
    if not fitz:
        return []
    scale = dpi / 72.0
    out = []
    tmpdir = tempfile.mkdtemp(prefix="hidpi_")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        fp = os.path.join(tmpdir, f"page_{i+1:03d}.png")
        pix.save(fp)
        out.append(fp)
    return out

def _to_crore_multiplier(unit_hint: str) -> float:
    u = (unit_hint or "").strip().lower()
    if "crore" in u or u.endswith("cr") or " cr" in u: return 1.0
    if "million" in u or "mn" in u: return 0.1            # 1 mn = 0.1 cr
    if "lakh" in u or "lac" in u: return 0.01            # 1 lakh = 0.01 cr
    if "thousand" in u or "000" in u: return 0.0001
    return 1.0  # default to crore

def _apply_multiplier(df: pd.DataFrame, mult: float) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if c == "Line item": continue
        df2[c] = pd.to_numeric(df2[c], errors="coerce") * mult
    return df2

def _df_from_table_obj(t: Dict[str, Any]) -> pd.DataFrame:
    """
    Expect t like: {"columns": ["Line item","Q1 FY26","Q4 FY25","Q1 FY25"],
                    "rows":[{"Line item":"Revenue from operations","values":[123.45, 120.0, 98.7]}, ...]}
    """
    cols = t.get("columns") or []
    rows = t.get("rows") or []
    if not cols or not rows:
        return pd.DataFrame()
    # Build rows
    data = []
    for r in rows:
        label = r.get("Line item") or r.get("line_item") or ""
        vals = r.get("values") or []
        # pad/trim to match len(cols)-1 numeric cols
        need = len(cols) - 1
        if len(vals) < need: vals = vals + [None]*(need - len(vals))
        elif len(vals) > need: vals = vals[:need]
        data.append([label, *vals])
    df = pd.DataFrame(data, columns=cols)
    # Coerce numerics
    for c in df.columns:
        if c == "Line item": continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _choose_current_col(df: pd.DataFrame) -> str|None:
    """Pick the 'current' period column heuristically: max filled numeric count; tie-break to leftmost."""
    if df is None or df.empty: return None
    numeric_cols = [c for c in df.columns if c != "Line item"]
    if not numeric_cols: return None
    counts = [(c, df[c].notna().sum()) for c in numeric_cols]
    counts.sort(key=lambda x: (-x[1], numeric_cols.index(x[0])))
    return counts[0][0]

def _sanity_pnl(df: pd.DataFrame, tol_cr: float):
    if df is None or df.empty: return None
    col = _choose_current_col(df)
    if not col: return None
    def pick(rx):
        s = df.loc[df["Line item"].str.contains(rx, flags=re.I, na=False), col]
        return float(s.mean()) if not s.dropna().empty else None
    rev = pick(r"revenue from operations|income from operations")
    oth = pick(r"other income")
    tot = pick(r"total income")
    if rev is None or oth is None or tot is None: return None
    diff = abs(tot - (rev + oth))
    return (diff <= tol_cr), {"col": col, "revenue": rev, "other": oth, "total": tot, "diff": diff}

def _sanity_bs(df: pd.DataFrame, tol_cr: float):
    if df is None or df.empty: return None
    col = _choose_current_col(df)
    if not col: return None
    def pick(rx):
        s = df.loc[df["Line item"].str.contains(rx, flags=re.I, na=False), col]
        return float(s.mean()) if not s.dropna().empty else None
    ta  = pick(r"total assets")
    tel = pick(r"(equity and liabilities|total equity.*liab)")
    if ta is None or tel is None: return None
    diff = abs(ta - tel)
    return (diff <= tol_cr), {"col": col, "total_assets": ta, "equity_liab": tel, "diff": diff}

# =========================
# OpenAI call
# =========================
def _extract_with_openai(pdf_bytes: bytes, image_files: List[str], model: str) -> Dict[str, Any]:
    """
    Sends PDF + (optional) high-DPI page images to OpenAI and requests strict JSON tables.
    Returns parsed dict.
    """
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in env or Streamlit secrets.")
    client = OpenAI(api_key=api_key)

    # Upload PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes); tmp.flush()
        f_pdf = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")

    content = [
        {
            "type": "input_text",
            "text": (
                "You are a meticulous financial data extractor. Read the attached financial results PDF. "
                "Return CONSOLIDATED tables ONLY if present; otherwise return STANDALONE (mark basis accordingly). "
                "STRICTLY follow the JSON schema. If a figure is missing in the PDF, use null (do NOT invent values). "
                "Units: extract the unit note exactly as printed (e.g., '₹ in Crore', '₹ in million')."
            )
        },
        {"type": "input_file", "file_id": f_pdf.id},
    ]

    # Upload high-DPI page images to improve vision on image-only PDFs
    if image_files:
        for p in image_files[:12]:  # cap to keep request light
            try:
                f_img = client.files.create(file=open(p, "rb"), purpose="vision")
                content.append({"type":"input_image","image":{"file_id": f_img.id}})
            except Exception:
                continue

    # JSON schema (tight but flexible)
    json_schema = {
        "name": "financial_tables",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "basis": {"type":"string", "enum":["Consolidated","Standalone"]},
                "unit_note": {"type":"string"},
                "period_labels": {
                    "type":"object",
                    "additionalProperties": True,
                    "properties": {
                        "current": {"type":"string"},
                        "previous": {"type":"string"},
                        "yoy": {"type":"string"}
                    }
                },
                "tables": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pnl": {
                            "type": "object",
                            "required": ["columns","rows"],
                            "properties": {
                                "title": {"type":"string"},
                                "columns": {
                                    "type":"array",
                                    "minItems": 2,
                                    "items": {"type":"string"}
                                },
                                "rows": {
                                    "type":"array",
                                    "items": {
                                        "type":"object",
                                        "required":["Line item","values"],
                                        "properties": {
                                            "Line item":{"type":"string"},
                                            "values":{
                                                "type":"array",
                                                "items":{"type":["number","null"]}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "bs": {
                            "type": "object",
                            "required": ["columns","rows"],
                            "properties": {
                                "title": {"type":"string"},
                                "columns": {"type":"array","minItems": 2,"items":{"type":"string"}},
                                "rows": {
                                    "type":"array",
                                    "items": {
                                        "type":"object",
                                        "required":["Line item","values"],
                                        "properties": {
                                            "Line item":{"type":"string"},
                                            "values":{"type":"array","items":{"type":["number","null"]}}
                                        }
                                    }
                                }
                            }
                        },
                        "cf": {
                            "type": "object",
                            "required": ["columns","rows"],
                            "properties": {
                                "title": {"type":"string"},
                                "columns": {"type":"array","minItems": 2,"items":{"type":"string"}},
                                "rows": {
                                    "type":"array",
                                    "items": {
                                        "type":"object",
                                        "required":["Line item","values"],
                                        "properties": {
                                            "Line item":{"type":"string"},
                                            "values":{"type":"array","items":{"type":["number","null"]}}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["basis","unit_note","tables"]
        }
    }

    resp = client.responses.create(
        model=model,
        input=[{"role":"user","content":content}],
        response_format={"type":"json_schema","json_schema":json_schema},
        temperature=0.0,
        max_output_tokens=4000
    )

    # Parse JSON
    # The SDK exposes .output_text or .output[...]; we stick to .output_text for portability.
    raw = (getattr(resp, "output_text", None) or "").strip()
    if not raw:
        raise RuntimeError("No JSON returned by model.")
    try:
        data = json.loads(raw)
    except Exception as e:
        # If JSON schema was ignored, try to salvage
        raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw:\n{raw[:800]}")

    return data

# =========================
# Main
# =========================
if run:
    # Get PDF bytes
    try:
        if file is not None:
            pdf_bytes = file.read()
        elif pdf_url:
            pdf_bytes = _download_pdf_from_url(pdf_url)
        else:
            st.error("Provide a PDF URL or upload a PDF.")
            st.stop()
    except Exception as e:
        st.error(f"PDF load failed: {e}")
        st.stop()

    # Optional high-DPI attachments
    image_files = []
    if attach_hidpi:
        if not fitz:
            st.warning("PyMuPDF not installed; cannot attach high-DPI page images.")
        else:
            try:
                image_files = _render_pdf_to_images(pdf_bytes, dpi=dpi)
            except Exception as e:
                st.warning(f"High-DPI rendering failed: {e}")

    with st.status("Asking OpenAI to extract strict JSON tables…", expanded=False):
        try:
            data = _extract_with_openai(pdf_bytes, image_files, model=model)
        except Exception as e:
            st.error(f"OpenAI extraction failed: {e}")
            st.stop()

    basis = data.get("basis") or "Unknown"
    unit_note = data.get("unit_note") or ""
    mult = _to_crore_multiplier(unit_note)

    st.write(f"**Reporting basis:** {basis}  |  **Unit note from PDF:** _{unit_note}_  → Converted to **INR Crore × {mult:.4g}**")

    tables = data.get("tables") or {}
    pnl_df = _apply_multiplier(_df_from_table_obj(tables.get("pnl", {})), mult) if tables.get("pnl") else pd.DataFrame()
    bs_df  = _apply_multiplier(_df_from_table_obj(tables.get("bs",  {})), mult) if tables.get("bs")  else pd.DataFrame()
    cf_df  = _apply_multiplier(_df_from_table_obj(tables.get("cf",  {})), mult) if tables.get("cf")  else pd.DataFrame()

    # Sanity checks
    col1, col2 = st.columns(2)
    with col1:
        s_pnl = _sanity_pnl(pnl_df, tol_cr=tol_cr)
        if s_pnl is None:
            st.info("P&L sanity: Not enough info to validate (need Revenue, Other income, Total income).")
        else:
            ok, ctx = s_pnl
            st.write("**P&L sanity (Total income ≈ Revenue + Other):**", "🟢 OK" if ok else "🔴 Mismatch")
            st.json({k:(round(v,2) if isinstance(v,(int,float)) else v) for k,v in ctx.items()})
    with col2:
        s_bs = _sanity_bs(bs_df, tol_cr=tol_cr)
        if s_bs is None:
            st.info("BS sanity: Not enough info to validate (need Total assets, Equity & liabilities).")
        else:
            ok, ctx = s_bs
            st.write("**BS sanity (Total assets ≈ Equity & Liabilities):**", "🟢 OK" if ok else "🔴 Mismatch")
            st.json({k:(round(v,2) if isinstance(v,(int,float)) else v) for k,v in ctx.items()})

    # Show tables + downloads
    def _show(title, df, key):
        st.subheader(title)
        if df is None or df.empty:
            st.warning("Not detected.")
            return
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇️ Download {key}.csv", data=csv, file_name=f"{key}.csv", mime="text/csv")

    _show("Income Statement (extracted via OpenAI)", pnl_df, "pnl")
    _show("Balance Sheet (extracted via OpenAI)", bs_df, "bs")
    _show("Cash Flow Statement (extracted via OpenAI)", cf_df, "cf")

else:
    st.info("Paste/upload a PDF and run. The app sends the PDF (and optionally high-DPI page images) to OpenAI and gets strict JSON tables back. No HTML is used.")
