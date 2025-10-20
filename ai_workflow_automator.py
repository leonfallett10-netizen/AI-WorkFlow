#!/usr/bin/env python3
"""
AI Workflow Automator — Single-File App (FastAPI + Google Cloud Storage + OpenAI)
[... trimmed in this string on purpose ...]
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os, io, time, json, re, uuid
from datetime import datetime
from pdfminer.high_level import extract_text
from google.cloud import storage

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI SDK not installed. Run: pip install openai==1.*") from e

app = FastAPI(title="AI Workflow Automator", version="1.0.0")

class GCSProcessRequest(BaseModel):
    bucket: str
    blob_name: str
    store_result: bool = False
    result_prefix: Optional[str] = None

class AnalysisResult(BaseModel):
    doc_id: str
    filename: Optional[str] = None
    char_count: int
    workflow: Dict[str, Any]
    timing_seconds: float
    efficiency_improvement_pct: float

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    return OpenAI()

def get_openai_model():
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_efficiency_baseline_seconds():
    try:
        return float(os.getenv("MANUAL_SECONDS_PER_DOC", "300"))
    except:
        return 300.0

def parse_text_from_upload(content: bytes, content_type: str) -> str:
    if content_type == "application/pdf":
        with io.BytesIO(content) as fh:
            text = extract_text(fh) or ""
    elif content_type.startswith("text/") or content_type in {"text/plain", "application/octet-stream"}:
        text = content.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")
    return text.strip()

def download_blob_to_bytes(bucket: str, blob_name: str) -> bytes:
    try:
        client = storage.Client()
        b = client.bucket(bucket).blob(blob_name)
        if not b.exists(client):
            raise HTTPException(status_code=404, detail=f"GCS blob not found: gs://{bucket}/{blob_name}")
        return b.download_as_bytes()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS download error: {e}")

def upload_json_to_gcs(bucket: str, blob_name: str, data: dict) -> str:
    try:
        client = storage.Client()
        b = client.bucket(bucket).blob(blob_name)
        import json as _json
        b.upload_from_string(_json.dumps(data, ensure_ascii=False, indent=2), content_type="application/json")
        return f"gs://{bucket}/{blob_name}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS upload error: {e}")

def sanitize_stem(name: str) -> str:
    import re, uuid
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip())
    return stem or f"doc_{uuid.uuid4().hex[:8]}"

def generate_workflow_from_text(text: str, filename: Optional[str] = None) -> Dict[str, Any]:
    client = get_openai_client()
    model = get_openai_model()
    system = (
        "You are a precise workflow analyst. Extract key information from documents "
        "and output a concise, actionable workflow plan as strict JSON matching the schema."
    )
    user = f"Document filename: {filename or 'uploaded'}\n\nDocument text (truncated to 6000 chars below):\n{text[:6000]}"
    schema_hint = {
        "type": "object",
        "properties": {
            "document_type": {"type": "string"},
            "summary": {"type": "string"},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "owner": {"type": "string"},
                        "priority": {"type": "string"},
                        "due_date_guess": {"type": "string"}
                    },
                    "required": ["title", "description"]
                }
            },
            "approvals": {"type": "array","items": {"type": "string"}},
            "routing": {"type": "array","items": {"type": "string"}},
            "risks": {"type": "array","items": {"type": "string"}},
            "missing_information": {"type": "array","items": {"type": "string"}}
        },
        "required": ["document_type", "summary", "tasks"]
    }
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "user", "content": f"Return ONLY valid JSON matching this schema (no extra text): {__import__('json').dumps(schema_hint)}"}
        ]
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.strip("` \n").replace("```json", "").replace("```", "").strip()
    try:
        return __import__('json').loads(raw)
    except Exception:
        return {
            "document_type": "unknown",
            "summary": raw[:500],
            "tasks": [{"title": "Review document", "description": "Model returned non-JSON; manual follow-up required."}],
            "approvals": [], "routing": [], "risks": ["LLM output not JSON"], "missing_information": []
        }

def build_result_payload(doc_id: str, filename: Optional[str], text: str, workflow: Dict[str, Any], elapsed: float):
    baseline = get_efficiency_baseline_seconds()
    improvement = max(0.0, (baseline - elapsed) / baseline * 100.0) if baseline > 0 else 0.0
    return {
        "doc_id": doc_id,
        "filename": filename,
        "char_count": len(text),
        "workflow": workflow,
        "timing_seconds": elapsed,
        "efficiency_improvement_pct": round(improvement, 2),
    }

from fastapi import Body

@app.post("/process")
async def process_document(
    file: UploadFile = File(..., description="PDF or TXT"),
    bucket: Optional[str] = Form(default=None),
    store_result: bool = Form(default=False),
    result_prefix: Optional[str] = Form(default=None),
):
    import time, uuid
    start = time.time()
    content = await file.read()
    text = parse_text_from_upload(content, file.content_type or "application/octet-stream")
    workflow = generate_workflow_from_text(text, filename=file.filename)
    elapsed = time.time() - start
    doc_id = __import__('uuid').uuid4().hex[:12]
    result = build_result_payload(doc_id, file.filename, text, workflow, elapsed)

    if store_result and bucket:
        stem = sanitize_stem(file.filename or f"doc_{doc_id}")
        out_name = f"{(result_prefix.strip('/') + '/' if result_prefix else '')}{stem}_workflow.json"
        gcs_uri = upload_json_to_gcs(bucket, out_name, result)
        result["stored_at"] = gcs_uri
    return JSONResponse(result)

class _Req(BaseModel):
    bucket: str
    blob_name: str
    store_result: bool = False
    result_prefix: Optional[str] = None

@app.post("/process_gcs")
def process_gcs(req: _Req = Body(...)):
    import time, uuid
    start = time.time()
    raw = download_blob_to_bytes(req.bucket, req.blob_name)
    content_type = "application/pdf" if req.blob_name.lower().endswith(".pdf") else "text/plain"
    text = parse_text_from_upload(raw, content_type)
    filename = req.blob_name.split("/")[-1]
    workflow = generate_workflow_from_text(text, filename=filename)
    elapsed = time.time() - start
    doc_id = __import__('uuid').uuid4().hex[:12]
    result = build_result_payload(doc_id, filename, text, workflow, elapsed)
    if req.store_result:
        stem = sanitize_stem(filename)
        out_name = f"{(req.result_prefix.strip('/') + '/' if req.result_prefix else '')}{stem}_workflow.json"
        gcs_uri = upload_json_to_gcs(req.bucket, out_name, result)
        result["stored_at"] = gcs_uri
    return JSONResponse(result)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": get_openai_model()}

def _cli():
    import argparse, pathlib, time, uuid
    ap = argparse.ArgumentParser(description="AI Workflow Automator (single-file)")
    ap.add_argument("--file", help="Local path to PDF/TXT")
    ap.add_argument("--bucket", help="GCS bucket to store result (optional)")
    ap.add_argument("--store", action="store_true", help="Store result JSON into GCS")
    ap.add_argument("--result-prefix", default=None, help="Optional GCS prefix for results")
    args = ap.parse_args()

    if not args.file:
        raise SystemExit("Please provide --file")

    path = pathlib.Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    content = path.read_bytes()
    content_type = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain"

    start = time.time()
    text = parse_text_from_upload(content, content_type)
    workflow = generate_workflow_from_text(text, filename=path.name)
    elapsed = time.time() - start

    doc_id = uuid.uuid4().hex[:12]
    result = build_result_payload(doc_id, path.name, text, workflow, elapsed)

    if args.store and args.bucket:
        stem = sanitize_stem(path.stem)
        out_name = f"{(args.result_prefix.strip('/') + '/' if args.result_prefix else '')}{stem}_workflow.json"
        gcs_uri = upload_json_to_gcs(args.bucket, out_name, result)
        result["stored_at"] = gcs_uri

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _cli()
