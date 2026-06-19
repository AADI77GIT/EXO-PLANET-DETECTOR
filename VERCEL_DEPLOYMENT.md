# ExoDetector Vercel and Multi-Service Deployment Guide

This guide provides command-by-command steps to deploy the **ExoDetector** frontend dashboard to Vercel and hook it up to a running backend.

---

## 1. Deployment Architecture Overview

Because the **ExoDetector** backend is a complex machine learning pipeline, it cannot run solely on Vercel. 
* **React Frontend**: Deployed on **Vercel** (excellent for fast, global, serverless frontend delivery).
* **FastAPI Backend, Redis, Celery Worker, PostgreSQL**: Deployed to a platform supporting persistent Dockerized workloads or long-running processes (e.g., **Render**, **Railway**, **AWS**, or **DigitalOcean**).

```
┌────────────────────────┐             ┌────────────────────────┐
│     Vercel Hosting     │             │    Render / Railway    │
│                        │             │                        │
│ ┌────────────────────┐ │ API requests│ ┌────────────────────┐ │
│ │  React Dashboard   ├─┼────────────►│ │  FastAPI Web App   │ │
│ └────────────────────┘ │             └──────────┬───────────┘ │
└────────────────────────┘                        │ (Jobs queue)│
                                                  ▼             │
                               ┌──────────────────────────────┐ │
                               │ Celery Background Worker     │ │
                               │ (Runs PyTorch models & BLS)  │ │
                               └─────────┬──────────────┬─────┘ │
                                         │              │       │
                                         ▼              ▼       │
                              ┌─────────────┐    ┌────────────┐ │
                              │ PostgreSQL  │    │ Redis Cache│ │
                              │  Database   │    │  (Broker)  │ │
                              └─────────────┘    └────────────┘ 
```

---

## 2. Step 1: Deploying the Backend Infrastructure

You can deploy the backend to **Render.com** or **Railway.app** using the existing `Dockerfile` in the `/backend` folder.

### Using Render (Free/Hobby Tier):
1. **PostgreSQL Database**:
   * Go to **Render Dashboard** -> **New** -> **PostgreSQL**.
   * Name: `exodetector-db`.
   * Click **Create Database**. Copy the **Internal Database URL** or **External Database URL**.
2. **Redis Instance**:
   * Go to **New** -> **Redis**.
   * Name: `exodetector-redis`.
   * Click **Create Redis**. Copy the **Internal Redis URL** or **External Redis URL**.
3. **FastAPI Web Service**:
   * Go to **New** -> **Web Service**.
   * Connect your GitHub Repository.
   * **Root Directory**: `backend` (or leave empty and select Docker builder).
   * **Runtime**: `Docker`.
   * Add the following **Environment Variables**:
     * `DATABASE_URL`: `postgresql+asyncpg://<user>:<password>@<host>/<database>` (Your Render PostgreSQL URL)
     * `REDIS_URL`: `redis://<host>:<port>` (Your Render Redis URL)
     * `API_KEY`: `your-custom-secure-key`
   * Click **Deploy Web Service**. Render will build the Docker container and expose a public URL (e.g., `https://exodetector-api.onrender.com`).
4. **Celery Worker**:
   * Go to **New** -> **Background Worker**.
   * Connect your GitHub Repository.
   * **Runtime**: `Docker`.
   * Under Docker Command or start command override: `celery -A celery_worker.celery_app worker --loglevel=info`
   * Add the same environment variables (`DATABASE_URL`, `REDIS_URL`, `API_KEY`).
   * Deploy.

---

## 3. Step 2: Deploying the Frontend to Vercel

The frontend is located under `exoplanet-pipeline/frontend`. Vercel will install the Node.js packages and build Vite on its build servers.

### Command-Line Deployment (Vercel CLI)
You can deploy directly from your local terminal using the Vercel CLI.

1. **Install Vercel CLI globally**:
   ```bash
   npm install -g vercel
   ```
   *(If Node/NPM isn't installed locally, you can skip this and deploy using GitHub integration instead).*

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Navigate to the frontend folder**:
   *Note: On Windows, use backslashes or forward slashes depending on your shell.*
   ```bash
   cd exoplanet-pipeline/frontend
   ```

4. **Initialize Vercel project**:
   ```bash
   vercel
   ```
   * Respond to the configuration prompts:
     * `Set up and deploy ...?` **Yes**
     * `Which scope ...?` **[Your Account]**
     * `Link to existing project?` **No**
     * `What's your project's name?` `exodetector-dashboard`
     * `In which directory is your code located?` `./`
     * `Want to modify settings?` **No** (Vercel will auto-detect Vite framework presets).

5. **Configure Production Environment Variables**:
   Vercel needs to know where your deployed FastAPI server resides. Set this on Vercel so the frontend queries the cloud server instead of localhost.
   ```bash
   vercel env add VITE_API_URL production
   ```
   * Enter the value: `https://your-backend-api-url.onrender.com` (Your Render FastAPI URL).

6. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

---

### Alternative: GitHub Deployment (Highly Recommended)
Vercel integrates seamlessly with GitHub and automates builds on every git push.

1. Create a new repository on GitHub and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initialize project structure with React dashboard"
   git branch -M main
   git remote add origin https://github.com/your-username/exodetector.git
   git push -u origin main
   ```
2. Log into the [Vercel Dashboard](https://vercel.com).
3. Click **Add New** -> **Project**.
4. Import your GitHub repository.
5. In the configuration settings:
   * **Root Directory**: Click edit and select `exoplanet-pipeline/frontend`.
   * **Framework Preset**: Auto-detected as **Vite**.
   * Under **Environment Variables**, add:
     * Name: `VITE_API_URL`
     * Value: `https://your-backend-api-url.onrender.com` (Your Render backend URL).
6. Click **Deploy**. Vercel will pull your frontend folder, run `npm run build`, and deploy it live!

---

## 4. Addressing Local CSS Warnings

You may notice warnings in your IDE stylesheet editor regarding:
* `Unknown at rule @tailwind`
* `Unknown at rule @apply`

These are standard editor-specific linting warnings and **will not affect the Vercel build**. The Vite compiler uses PostCSS to process these rules. If you wish to disable these warnings in your VS Code workspace, add the following to your project's `.vscode/settings.json` file:
```json
{
  "css.lint.unknownAtRules": "ignore"
}
```
