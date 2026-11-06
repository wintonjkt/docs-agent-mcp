"""REST API router, request/response models, and authentication dependency."""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from .config import Settings
from .controller import JobController
from .cos_client import discover_cos_batch
from .jobs import (
    DuplicateActiveJobError,
    InvalidTransition,
    JobConflict,
    JobNotFound,
    JobStore,
    render_job,
)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Request / Response Schemas
# --------------------------------------------------------------------------- #
class CreateJobRequest(BaseModel):
    bucket: str = Field(..., description="Source COS bucket name")
    prefix: str = Field("", description="Object key prefix filter")
    collection: Optional[str] = Field(None, description="Astra DB collection name")
    archive_bucket: Optional[str] = Field(None, description="Destination archive bucket")
    concurrency: Optional[int] = Field(None, ge=1, description="Parallel worker count")


class ActionRequest(BaseModel):
    reason: Optional[str] = Field("", description="Optional reason for pause/stop")


class HealthResponse(BaseModel):
    status: str
    details: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# State Dependencies
# --------------------------------------------------------------------------- #
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def get_controller(request: Request) -> JobController:
    return request.app.state.controller


def get_cos_client(request: Request) -> Any:
    return request.app.state.cos_client


def verify_auth(
    request: Request,
    x_access_token: Optional[str] = Header(None, alias="X-Access-Token"),
    authorization: Optional[str] = Header(None),
) -> None:
    expected_token = request.app.state.settings.ui_access_token
    token = x_access_token
    if not token and authorization:
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()

    if not token or not secrets.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing access token",
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.post("/jobs", dependencies=[Depends(verify_auth)])
def create_job(
    req: CreateJobRequest,
    controller: JobController = Depends(get_controller),
) -> Dict[str, Any]:
    try:
        job = controller.start_job(
            bucket=req.bucket,
            prefix=req.prefix,
            collection=req.collection,
            archive_bucket=req.archive_bucket,
            concurrency=req.concurrency,
        )
        return render_job(job)
    except DuplicateActiveJobError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("/jobs", dependencies=[Depends(verify_auth)])
def list_jobs(
    limit: int = Query(50, ge=1, le=100),
    status_filter: Optional[List[str]] = Query(None, alias="status"),
    job_store: JobStore = Depends(get_job_store),
) -> List[Dict[str, Any]]:
    jobs = job_store.list_jobs(limit=limit, statuses=status_filter)
    return [render_job(j) for j in jobs]


@router.get("/jobs/{job_id}", dependencies=[Depends(verify_auth)])
def get_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> Dict[str, Any]:
    job = job_store.load_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )
    return render_job(job)


@router.post("/jobs/{job_id}/pause", dependencies=[Depends(verify_auth)])
def pause_job(
    job_id: str,
    req: ActionRequest = ActionRequest(),
    controller: JobController = Depends(get_controller),
) -> Dict[str, Any]:
    try:
        job = controller.pause_job(job_id, reason=req.reason or "")
        return render_job(job)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidTransition as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/resume", dependencies=[Depends(verify_auth)])
def resume_job(
    job_id: str,
    controller: JobController = Depends(get_controller),
) -> Dict[str, Any]:
    try:
        job = controller.resume_job(job_id)
        return render_job(job)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidTransition as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/stop", dependencies=[Depends(verify_auth)])
def stop_job(
    job_id: str,
    req: ActionRequest = ActionRequest(),
    controller: JobController = Depends(get_controller),
) -> Dict[str, Any]:
    try:
        job = controller.stop_job(job_id, reason=req.reason or "")
        return render_job(job)
    except JobNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidTransition as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except JobConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/preview", dependencies=[Depends(verify_auth)])
def preview_prefix(
    bucket: str = Query(..., description="Bucket name"),
    prefix: str = Query("", description="Key prefix"),
    limit: int = Query(25, ge=1, le=100),
    settings: Settings = Depends(get_settings),
    cos_client: Any = Depends(get_cos_client),
) -> Dict[str, Any]:
    try:
        items, next_tok, has_more = discover_cos_batch(
            cos_client, settings, bucket_name=bucket, prefix=prefix, target_count=limit,
        )
        return {
            "bucket": bucket,
            "prefix": prefix,
            "count": len(items),
            "has_more": has_more,
            "items": items,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
