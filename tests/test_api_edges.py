import pytest

from app.models import Detection, Job, Star


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [{"tic_id": -1, "sector": 1}, {"tic_id": 0, "sector": 1}, {"tic_id": "abc", "sector": 1}])
async def test_invalid_tic_returns_422(client, payload):
    response = await client.post("/api/pipeline/run", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_missing_job_returns_404(client):
    response = await client.get("/api/pipeline/status/not-real")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_results_before_completion_returns_404(client):
    response = await client.get("/api/results/123")
    assert response.status_code == 404
    assert response.json()["detail"] == "No results yet"


@pytest.mark.anyio
async def test_missing_phase_plot_returns_404(client, db_session):
    db_session.add(Star(tic_id=123, sector=1))
    db_session.add(Detection(tic_id=123, sector=1, label="PLANET", confidence=0.9, period_days=2.0, duration_hours=2.0, depth_ppt=1.0, parameter_errors={}, plot_paths={"phase": "missing.png"}))
    await db_session.commit()
    response = await client.get("/api/results/123/plot/phase")
    assert response.status_code == 404
    assert response.json()["detail"] == "Plot file not found"


@pytest.mark.anyio
async def test_train_requires_api_key(client):
    response = await client.post("/api/train", json={"dataset_path": "missing.csv"})
    assert response.status_code == 403


@pytest.mark.anyio
async def test_concurrent_same_tic_returns_existing_job(client, db_session, monkeypatch):
    class DummyTask:
        def apply_async(self, *args, **kwargs):
            return None

    monkeypatch.setattr("app.routers.pipeline.run_pipeline_task", DummyTask())
    first = await client.post("/api/pipeline/run", json={"tic_id": 123, "sector": 1})
    second = await client.post("/api/pipeline/run", json={"tic_id": 123, "sector": 1})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]


@pytest.mark.anyio
async def test_tic_not_found_job_can_fail_gracefully(client, db_session):
    db_session.add(Job(job_id="job1", tic_id=999, sector=1, status="FAILED", error_msg="preprocess: FITS not found"))
    await db_session.commit()
    response = await client.get("/api/pipeline/status/job1")
    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert "FITS not found" in response.json()["error_msg"]
