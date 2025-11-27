import uuid
import traceback
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from analyzer.ifc_loader import load_ifc_from_bytes
from analyzer.instrument_finder import find_instruments
from analyzer.pipe_graph import build_pipe_graph
from analyzer.geometry_rules import analyze_instrument
from analyzer.report_builder import build_report
from gemini_summary import router as gemini_router


app = FastAPI(title="IFC Flow Instrument Inspector API")
app.include_router(gemini_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict] = {}

@app.post("/analyze-model")
async def analyze_model(file: UploadFile = File(...)):
    """
    Synchronous analyze endpoint:
    - Accepts multipart file upload (IFC)
    - Runs analysis pipeline (may be CPU-heavy)
    - Returns final JSON report
    """
    if not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Only .ifc files are supported in this endpoint")

    try:
        content = await file.read()
        ifc = load_ifc_from_bytes(content)

        instruments = find_instruments(ifc)

        pipe_graph = build_pipe_graph(ifc)

        results = []
        for inst in instruments:
            r = analyze_instrument(inst, ifc, pipe_graph)
            results.append(r)

        report = build_report(results)
        return report

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-model-async")
async def analyze_model_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Async endpoint: schedules a background task and returns a job_id.
    Frontend should poll GET /jobs/{job_id}
    """
    if not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Only .ifc files are supported in this endpoint")

    content = await file.read()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}

    background_tasks.add_task(_background_analyze, job_id, content)
    return {"job_id": job_id, "status": "queued"}

async def _background_analyze(job_id: str, file_bytes: bytes):
    """
    Background worker wrapper. Stores result into jobs[job_id].
    """
    try:
        jobs[job_id]["status"] = "running"

        ifc = load_ifc_from_bytes(file_bytes)
        instruments = find_instruments(ifc)
        pipe_graph = build_pipe_graph(ifc)

        results = []
        for inst in instruments:
            r = analyze_instrument(inst, ifc, pipe_graph)
            results.append(r)

        report = build_report(results)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = report

    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job id not found")
    return job

@app.get("/")
async def root():
    return {"msg": "IFC Flow Instrument Inspector API - healthy"}


import logging
logger = logging.getLogger("uvicorn.error")

async def _background_analyze(job_id: str, file_bytes: bytes):
    """
    Background worker wrapper. Stores result into jobs[job_id].
    This version logs progress and captures exceptions.
    """
    try:
        logger.info(f"[JOB {job_id}] background task starting")
        jobs[job_id]["status"] = "running"

        # Load IFC
        logger.info(f"[JOB {job_id}] loading IFC...")
        ifc = load_ifc_from_bytes(file_bytes)
        logger.info(f"[JOB {job_id}] IFC loaded")

        try:
            els = []
            if hasattr(ifc, "by_type"):
                for typ in ("IfcPipeSegment","IfcFlowInstrument","IfcDistributionFlowElement","IfcElement","IfcFlowSegment","IfcFitting"):
                    try:
                        c = len(list(ifc.by_type(typ)))
                    except Exception:
                        c = 0
                    logger.info(f"[JOB {job_id}] count of {typ}: {c}")
        except Exception:
            logger.exception(f"[JOB {job_id}] failed counting IFC types (non-fatal)")

        logger.info(f"[JOB {job_id}] finding instruments...")
        instruments = find_instruments(ifc)
        logger.info(f"[JOB {job_id}] instrument candidates found: {len(instruments)}")

        logger.info(f"[JOB {job_id}] building pipe graph...")
        pipe_graph = build_pipe_graph(ifc)
        try:
            gn = len(pipe_graph.nodes)
            ge = len(pipe_graph.edges)
        except Exception:
            gn = ge = 0
        logger.info(f"[JOB {job_id}] pipe_graph nodes={gn} edges={ge}")

        results = []
        for i, inst in enumerate(instruments):
            logger.info(f"[JOB {job_id}] analyzing instrument {i+1}/{len(instruments)}")
            r = analyze_instrument(inst, ifc, pipe_graph)
            results.append(r)

        report = build_report(results)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = report
        logger.info(f"[JOB {job_id}] done, instruments reported: {len(results)}")

    except Exception as e:
        logger.exception(f"[JOB {job_id}] failed during analysis: {e}")
        jobs[job_id]["status"] = "failed"
        import traceback as _tb
        jobs[job_id]["error"] = _tb.format_exc()