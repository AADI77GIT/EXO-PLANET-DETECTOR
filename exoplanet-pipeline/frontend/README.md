# Frontend Contract

The dashboard is intentionally secondary for the first backend milestone.
Build it as a separate React + Tailwind + Recharts app that consumes only:

- `POST /api/pipeline/run`
- `GET /api/pipeline/status/{job_id}`
- `GET /api/results/{tic_id}`
- `GET /api/results/{tic_id}/plot`
- `GET /api/stars/`
- `GET /api/health`

Use the pasted mission-control design prompt as the UI source of truth.

