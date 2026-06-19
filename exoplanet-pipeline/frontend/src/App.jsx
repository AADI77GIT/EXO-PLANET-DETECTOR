import React, { useState, useEffect } from 'react';
import { 
  Rocket, 
  Activity, 
  Database, 
  TrendingUp, 
  RefreshCw, 
  CheckCircle2, 
  XCircle, 
  AlertTriangle,
  Cpu,
  Search,
  ChevronLeft,
  ChevronRight,
  Sliders,
  Image as ImageIcon,
  Layers,
  Zap,
  Info
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

// Determine backend base URL. During development, Vite proxies /api to http://localhost:8000.
// For production, you can set VITE_API_URL or deploy as relative.
const API_BASE = import.meta.env.VITE_API_URL || '';

export default function App() {
  // System states
  const [health, setHealth] = useState({ status: 'unknown', db: 'unknown', redis: 'unknown', models_loaded: false });
  const [loadingHealth, setLoadingHealth] = useState(true);
  
  // Stars listing
  const [stars, setStars] = useState([]);
  const [totalStars, setTotalStars] = useState(0);
  const [page, setPage] = useState(1);
  const [limit] = useState(8);
  const [searchTic, setSearchTic] = useState('');
  
  // Selection
  const [selectedTic, setSelectedTic] = useState(null);
  const [selectedResult, setSelectedResult] = useState(null);
  const [loadingResult, setLoadingResult] = useState(false);
  const [activePlotTab, setActivePlotTab] = useState('raw'); // raw, denoised, prob, phase
  
  // Launch Pipeline run
  const [runTic, setRunTic] = useState('');
  const [runSector, setRunSector] = useState('1');
  const [activeJob, setActiveJob] = useState(null); // { id, status, progress, error }
  const [jobInterval, setJobInterval] = useState(null);

  // Model Training
  const [trainDataset, setTrainDataset] = useState('data/processed/training_set.csv');
  const [trainEpochs, setTrainEpochs] = useState(100);
  const [trainBatch, setTrainBatch] = useState(64);
  const [apiKey, setApiKey] = useState('');
  const [trainingJob, setTrainingJob] = useState(null);

  // Notifications
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  // Recharts interactive mockup data for an exoplanet transit
  const generateMockTransitData = (period = 4.5, depthPpt = 12.4) => {
    const data = [];
    for (let i = 0; i <= 100; i++) {
      const phase = (i / 100) - 0.5; // -0.5 to 0.5
      let flux = 1.0;
      // create a transit dip centered at 0
      if (Math.abs(phase) < 0.08) {
        // smooth trapezoidal dip
        const x = Math.abs(phase) / 0.08;
        const dip = (1 - Math.pow(x, 2)) * (depthPpt / 1000);
        flux -= dip;
      }
      // add minor noise
      flux += (Math.random() - 0.5) * 0.0008;
      data.push({ phase: parseFloat(phase.toFixed(3)), flux: parseFloat(flux.toFixed(6)) });
    }
    return data;
  };

  const [interactiveData, setInteractiveData] = useState([]);

  // Fetch system health on load
  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth({ status: 'down', db: 'down', redis: 'down', models_loaded: false });
      }
    } catch (err) {
      setHealth({ status: 'down', db: 'down', redis: 'down', models_loaded: false });
    } finally {
      setLoadingHealth(false);
    }
  };

  // Fetch processed stars
  const fetchStars = async (p = 1) => {
    try {
      const res = await fetch(`${API_BASE}/api/stars/?page=${p}&limit=${limit}`);
      if (res.ok) {
        const data = await res.json();
        setStars(data.items);
        setTotalStars(data.total);
        setPage(data.page);
      }
    } catch (err) {
      console.error("Error fetching stars list", err);
    }
  };

  // Fetch selected star details
  const selectStar = async (ticId) => {
    setSelectedTic(ticId);
    setLoadingResult(true);
    setErrorMsg('');
    setSelectedResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/results/${ticId}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedResult(data);
        // Setup interactive chart parameters
        setInteractiveData(generateMockTransitData(data.period_days, data.depth_ppt));
      } else {
        const errData = await res.json().catch(() => ({}));
        setErrorMsg(errData.detail || `Failed to fetch results for TIC ${ticId}`);
      }
    } catch (err) {
      setErrorMsg("Network error fetching transit analysis.");
    } finally {
      setLoadingResult(false);
    }
  };

  // Launch pipeline job
  const handleLaunchPipeline = async (e) => {
    e.preventDefault();
    if (!runTic) return;
    setErrorMsg('');
    setSuccessMsg('');
    
    try {
      const res = await fetch(`${API_BASE}/api/pipeline/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tic_id: parseInt(runTic),
          sector: parseInt(runSector)
        })
      });
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to launch pipeline task");
      }
      
      const job = await res.json();
      setActiveJob({
        id: job.job_id,
        status: job.status,
        tic_id: runTic,
        sector: runSector,
        message: job.message
      });
      setSuccessMsg(`Job ${job.job_id.substring(0, 8)} queued successfully!`);
      
      // Start polling
      pollJobStatus(job.job_id);
    } catch (err) {
      setErrorMsg(err.message);
    }
  };

  // Poll Job Status helper
  const pollJobStatus = (jobId) => {
    if (jobInterval) clearInterval(jobInterval);
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/status/${jobId}`);
        if (!res.ok) return;
        const statusData = await res.json();
        
        setActiveJob(prev => {
          if (!prev) return null;
          return {
            ...prev,
            status: statusData.status,
            error_msg: statusData.error_msg,
            result_id: statusData.result_id
          };
        });

        if (statusData.status === 'SUCCESS') {
          clearInterval(interval);
          setSuccessMsg(`Pipeline completed successfully for TIC ${statusData.tic_id}!`);
          fetchStars(1); // refresh list
          selectStar(statusData.tic_id); // auto-select newly processed star
          setTimeout(() => setActiveJob(null), 5000);
        } else if (statusData.status === 'FAILURE') {
          clearInterval(interval);
          setErrorMsg(`Job failed: ${statusData.error_msg || 'Unknown pipeline exception'}`);
          setTimeout(() => setActiveJob(null), 10000);
        }
      } catch (err) {
        console.error("Error polling job", err);
      }
    }, 2000);
    
    setJobInterval(interval);
  };

  // Launch Model Training job
  const handleLaunchTraining = async (e) => {
    e.preventDefault();
    if (!apiKey) {
      setErrorMsg("X-API-Key is required to launch training runs.");
      return;
    }
    setErrorMsg('');
    setSuccessMsg('');
    
    try {
      const res = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({
          dataset_path: trainDataset,
          epochs: parseInt(trainEpochs),
          batch_size: parseInt(trainBatch)
        })
      });
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Forbidden or invalid inputs");
      }
      
      const job = await res.json();
      setTrainingJob({
        id: job.job_id,
        status: job.status,
        message: job.message
      });
      setSuccessMsg(`Model training task ${job.job_id.substring(0, 8)} started in Celery background.`);
      
      // Start polling training job
      pollTrainingJobStatus(job.job_id);
    } catch (err) {
      setErrorMsg(err.message);
    }
  };

  const pollTrainingJobStatus = (jobId) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/status/${jobId}`);
        if (!res.ok) return;
        const statusData = await res.json();
        
        setTrainingJob(prev => {
          if (!prev) return null;
          return {
            ...prev,
            status: statusData.status,
            error_msg: statusData.error_msg
          };
        });

        if (statusData.status === 'SUCCESS') {
          clearInterval(interval);
          setSuccessMsg(`Model training successfully finished and saved!`);
          fetchHealth(); // refresh models_loaded status
          setTimeout(() => setTrainingJob(null), 8000);
        } else if (statusData.status === 'FAILURE') {
          clearInterval(interval);
          setErrorMsg(`Training job failed: ${statusData.error_msg || 'Dataset not found or GPU/Memory error'}`);
          setTimeout(() => setTrainingJob(null), 10000);
        }
      } catch (err) {
        console.error("Error polling training job", err);
      }
    }, 3000);
  };

  useEffect(() => {
    fetchHealth();
    fetchStars(1);
    
    // Auto refresh health state
    const healthTimer = setInterval(fetchHealth, 10000);
    return () => {
      clearInterval(healthTimer);
      if (jobInterval) clearInterval(jobInterval);
    };
  }, []);

  // Filter stars client-side if searched
  const filteredStars = stars.filter(s => s.tic_id.toString().includes(searchTic));

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 stars-bg flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* HEADER HUD */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-600/20 text-indigo-400 rounded-lg border border-indigo-500/30">
              <Rocket className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold font-display tracking-tight text-white flex items-center gap-2">
                EXODETECTOR <span className="text-xs bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 px-2 py-0.5 rounded font-mono">MISSION HUB</span>
              </h1>
              <p className="text-xs text-slate-400">Deep Learning Pipeline for TESS Light Curve Planet Hunting</p>
            </div>
          </div>

          {/* Core System Indicators */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-2 flex items-center space-x-4">
              <div className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-slate-400">Database:</span>
                {loadingHealth ? (
                  <span className="w-2 h-2 rounded-full bg-yellow-500 animate-ping"></span>
                ) : health.db === 'connected' || health.db === 'up' ? (
                  <span className="text-emerald-400 font-medium">ONLINE</span>
                ) : (
                  <span className="text-rose-400 font-medium">OFFLINE</span>
                )}
              </div>

              <div className="h-4 w-px bg-slate-800"></div>

              <div className="flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-slate-400">Broker (Redis):</span>
                {loadingHealth ? (
                  <span className="w-2 h-2 rounded-full bg-yellow-500 animate-ping"></span>
                ) : health.redis === 'connected' || health.redis === 'up' ? (
                  <span className="text-emerald-400 font-medium">ONLINE</span>
                ) : (
                  <span className="text-rose-400 font-medium">OFFLINE</span>
                )}
              </div>

              <div className="h-4 w-px bg-slate-800"></div>

              <div className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-slate-400">Classifier Model:</span>
                {loadingHealth ? (
                  <span className="w-2 h-2 rounded-full bg-yellow-500 animate-ping"></span>
                ) : health.models_loaded ? (
                  <span className="text-emerald-400 font-medium px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">LOADED</span>
                ) : (
                  <span className="text-yellow-500 font-medium px-1.5 py-0.5 rounded bg-yellow-500/10 border border-yellow-500/20">UNINITIALIZED</span>
                )}
              </div>
            </div>

            <button 
              onClick={() => { fetchHealth(); fetchStars(1); }} 
              className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition border border-slate-700 active:scale-95"
              title="Force Refresh Metrics"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* SYSTEM MESSAGES BANNER */}
      {errorMsg && (
        <div className="bg-rose-900/40 border-y border-rose-500/30 text-rose-200 px-6 py-3 flex items-center justify-between text-sm max-w-7xl mx-auto w-full my-2 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button onClick={() => setErrorMsg('')} className="hover:text-white font-bold">×</button>
        </div>
      )}

      {successMsg && (
        <div className="bg-emerald-900/40 border-y border-emerald-500/30 text-emerald-200 px-6 py-3 flex items-center justify-between text-sm max-w-7xl mx-auto w-full my-2 rounded-lg">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <span>{successMsg}</span>
          </div>
          <button onClick={() => setSuccessMsg('')} className="hover:text-white font-bold">×</button>
        </div>
      )}

      {/* CORE WORKSPACE GRID */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT PANEL: LAUNCH & STAR CATALOG (COL SPAN 4) */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          
          {/* PIPELINE CONTROL COMMAND DECK */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 backdrop-blur-md">
            <h2 className="text-sm font-semibold tracking-wider text-slate-300 uppercase mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4 text-indigo-400" /> Run Light Curve Pipeline
            </h2>
            <form onSubmit={handleLaunchPipeline} className="space-y-4">
              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1">TIC ID (TESS Input Catalog)</label>
                <input 
                  type="number" 
                  placeholder="e.g. 251554286"
                  required
                  value={runTic} 
                  onChange={(e) => setRunTic(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 font-mono transition"
                />
              </div>

              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1">Observation Sector</label>
                <select 
                  value={runSector} 
                  onChange={(e) => setRunSector(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 font-mono"
                >
                  {[...Array(30).keys()].map(i => (
                    <option key={i + 1} value={i + 1}>Sector {i + 1}</option>
                  ))}
                </select>
              </div>

              <button 
                type="submit" 
                disabled={activeJob && activeJob.status !== 'SUCCESS' && activeJob.status !== 'FAILURE'}
                className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:opacity-50 text-white font-medium rounded-lg px-4 py-2.5 text-sm transition flex items-center justify-center gap-2 active:scale-95 cursor-pointer shadow-lg shadow-indigo-600/20"
              >
                <Rocket className="w-4 h-4" />
                {activeJob && activeJob.status !== 'SUCCESS' && activeJob.status !== 'FAILURE' ? 'RUNNING PIPELINE...' : 'LAUNCH PIPELINE'}
              </button>
            </form>

            {/* Active pipeline run progress indicator */}
            {activeJob && (
              <div className="mt-4 p-3 bg-slate-950/80 rounded-lg border border-slate-800 text-xs font-mono space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Job: {activeJob.id.substring(0, 10)}...</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    activeJob.status === 'RUNNING' ? 'bg-indigo-500/20 text-indigo-400 animate-pulse' :
                    activeJob.status === 'PENDING' ? 'bg-amber-500/20 text-amber-400' :
                    activeJob.status === 'SUCCESS' ? 'bg-emerald-500/20 text-emerald-400' :
                    'bg-rose-500/20 text-rose-400'
                  }`}>{activeJob.status}</span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                  <div className={`h-full bg-indigo-500 transition-all duration-500 ${
                    activeJob.status === 'RUNNING' ? 'w-2/3 glowing-status-active' :
                    activeJob.status === 'SUCCESS' ? 'w-full' :
                    activeJob.status === 'FAILURE' ? 'w-full bg-rose-500' : 'w-1/12'
                  }`}></div>
                </div>
                <p className="text-[10px] text-slate-400 leading-tight">
                  TIC ID: {activeJob.tic_id} | Sector: {activeJob.sector}
                </p>
              </div>
            )}
          </div>

          {/* STAR HISTORY CATALOG */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 backdrop-blur-md flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold tracking-wider text-slate-300 uppercase flex items-center gap-2">
                <Database className="w-4 h-4 text-indigo-400" /> Star Catalog
              </h2>
              <span className="text-xs bg-slate-800 border border-slate-700 px-2 py-0.5 rounded font-mono text-slate-400">
                Total: {totalStars}
              </span>
            </div>

            {/* Filter Search Input */}
            <div className="relative mb-3">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <input 
                type="text" 
                placeholder="Search TIC ID..."
                value={searchTic} 
                onChange={(e) => setSearchTic(e.target.value)}
                className="w-full bg-slate-950 border border-slate-850 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono transition"
              />
            </div>

            {/* Star Items list */}
            <div className="space-y-1 overflow-y-auto max-h-[300px] flex-1 pr-1">
              {filteredStars.length === 0 ? (
                <div className="text-center py-8 text-xs text-slate-500 italic">
                  No processed stars in catalog.
                </div>
              ) : (
                filteredStars.map((star) => (
                  <button
                    key={star.tic_id}
                    onClick={() => selectStar(star.tic_id)}
                    className={`w-full text-left font-mono text-xs px-3 py-2.5 rounded-lg border transition flex items-center justify-between ${
                      selectedTic === star.tic_id 
                        ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200' 
                        : 'bg-slate-950/40 border-slate-850 hover:bg-slate-800/50 hover:border-slate-700 text-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                      <span>TIC {star.tic_id}</span>
                    </div>
                    <div className="flex items-center gap-2 text-[10px] text-slate-400">
                      <span>Sec {star.sector || 'N/A'}</span>
                      <span className="opacity-60 text-[9px]">
                        {star.processed_at ? new Date(star.processed_at).toLocaleDateString() : ''}
                      </span>
                    </div>
                  </button>
                ))
              )}
            </div>

            {/* Pagination HUD */}
            <div className="flex items-center justify-between mt-4 pt-3 border-t border-slate-850 text-xs">
              <button 
                disabled={page <= 1}
                onClick={() => fetchStars(page - 1)}
                className="p-1.5 bg-slate-950 border border-slate-800 hover:bg-slate-800 disabled:opacity-30 rounded transition flex items-center"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-slate-400 font-mono">Page {page} of {Math.ceil(totalStars / limit) || 1}</span>
              <button 
                disabled={page >= Math.ceil(totalStars / limit)}
                onClick={() => fetchStars(page + 1)}
                className="p-1.5 bg-slate-950 border border-slate-800 hover:bg-slate-800 disabled:opacity-30 rounded transition flex items-center"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </section>

        {/* MIDDLE & RIGHT PANEL: TRANSIT RESULT INSPECTOR (COL SPAN 8) */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          {loadingResult ? (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-20 flex flex-col items-center justify-center gap-4 min-h-[500px]">
              <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin" />
              <p className="text-sm font-mono text-slate-400">Retrieving orbital parameters and neural network predictions...</p>
            </div>
          ) : selectedResult ? (
            <div className="space-y-6">
              
              {/* STAGE 1: METADATA & NEURAL NET CLASSIFICATION RES */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 backdrop-blur-md">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-4 mb-4 gap-2">
                  <div>
                    <h2 className="text-lg font-bold text-white font-mono">STAR SYSTEM: TIC {selectedTic}</h2>
                    <p className="text-xs text-slate-400">Processed: {new Date(selectedResult.created_at).toLocaleString()}</p>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400 uppercase font-mono">Primary Classification:</span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold font-mono border ${
                      selectedResult.label === 'PLANET' ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300' :
                      selectedResult.label === 'ECLIPSING_BINARY' ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300' :
                      selectedResult.label === 'STARSPOT' ? 'bg-amber-500/20 border-amber-500/40 text-amber-300' :
                      'bg-rose-500/20 border-rose-500/40 text-rose-300'
                    }`}>
                      🪐 {selectedResult.label}
                    </span>
                    <span className="text-xs font-bold text-white font-mono bg-slate-800 px-2 py-1 rounded">
                      {(selectedResult.confidence * 100).toFixed(1)}% Conf
                    </span>
                  </div>
                </div>

                {/* Score bar chart details */}
                <div>
                  <h3 className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">Neural Net Probabilities</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Planet Score */}
                    <div>
                      <div className="flex justify-between text-xs font-mono mb-1">
                        <span className="text-slate-300">🪐 Exoplanet Planet</span>
                        <span className="text-emerald-400">
                          {selectedResult.label === 'PLANET' ? (selectedResult.confidence * 100).toFixed(1) : '—'}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                        <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${selectedResult.label === 'PLANET' ? selectedResult.confidence * 100 : 15}%` }}></div>
                      </div>
                    </div>

                    {/* EB Score */}
                    <div>
                      <div className="flex justify-between text-xs font-mono mb-1">
                        <span className="text-slate-300">🌟 Eclipsing Binary Star</span>
                        <span className="text-indigo-400">
                          {selectedResult.label === 'ECLIPSING_BINARY' ? (selectedResult.confidence * 100).toFixed(1) : '—'}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                        <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${selectedResult.label === 'ECLIPSING_BINARY' ? selectedResult.confidence * 100 : 10}%` }}></div>
                      </div>
                    </div>

                    {/* False Positive Score */}
                    <div>
                      <div className="flex justify-between text-xs font-mono mb-1">
                        <span className="text-slate-300">⚠️ Instrument False Positive</span>
                        <span className="text-rose-400">
                          {selectedResult.label === 'FALSE_POSITIVE' ? (selectedResult.confidence * 100).toFixed(1) : '—'}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                        <div className="bg-rose-500 h-full rounded-full" style={{ width: `${selectedResult.label === 'FALSE_POSITIVE' ? selectedResult.confidence * 100 : 5}%` }}></div>
                      </div>
                    </div>

                    {/* Starspot Score */}
                    <div>
                      <div className="flex justify-between text-xs font-mono mb-1">
                        <span className="text-slate-300">☀️ Stellar Activity / Spot</span>
                        <span className="text-amber-400">
                          {selectedResult.label === 'STARSPOT' ? (selectedResult.confidence * 100).toFixed(1) : '—'}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                        <div className="bg-amber-500 h-full rounded-full" style={{ width: `${selectedResult.label === 'STARSPOT' ? selectedResult.confidence * 100 : 5}%` }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* STAGE 2: DETECTED ORBITAL PARAMETERS GRID */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Period CARD */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 backdrop-blur-md flex flex-col justify-between">
                  <span className="text-xs text-slate-400 uppercase font-mono tracking-wider">Orbital Period</span>
                  <div className="my-2">
                    <span className="text-2xl font-bold text-white font-mono">{selectedResult.period_days.toFixed(4)}</span>
                    <span className="text-xs text-slate-400 font-mono ml-1">days</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    Err: ±{selectedResult.parameter_errors?.period_err ? selectedResult.parameter_errors.period_err.toFixed(5) : '0.00010'} days
                  </span>
                </div>

                {/* Duration CARD */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 backdrop-blur-md flex flex-col justify-between">
                  <span className="text-xs text-slate-400 uppercase font-mono tracking-wider">Transit Duration</span>
                  <div className="my-2">
                    <span className="text-2xl font-bold text-white font-mono">{selectedResult.duration_hours.toFixed(2)}</span>
                    <span className="text-xs text-slate-400 font-mono ml-1">hours</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    Err: ±{selectedResult.parameter_errors?.duration_err ? selectedResult.parameter_errors.duration_err.toFixed(3) : '0.05'} hours
                  </span>
                </div>

                {/* Depth CARD */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 backdrop-blur-md flex flex-col justify-between">
                  <span className="text-xs text-slate-400 uppercase font-mono tracking-wider">Transit Depth</span>
                  <div className="my-2">
                    <span className="text-2xl font-bold text-white font-mono">{selectedResult.depth_ppt.toFixed(3)}</span>
                    <span className="text-xs text-slate-400 font-mono ml-1">ppt</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    Err: ±{selectedResult.parameter_errors?.depth_err ? selectedResult.parameter_errors.depth_err.toFixed(3) : '0.100'} ppt (parts-per-thousand)
                  </span>
                </div>
              </div>

              {/* STAGE 3: PLOT DECK GALLERY & INTERACTIVE CHART */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 backdrop-blur-md">
                <div className="flex flex-col sm:flex-row items-center justify-between border-b border-slate-800 pb-3 mb-4 gap-2">
                  <h3 className="text-sm font-semibold tracking-wider text-slate-300 uppercase flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" /> Pipeline Diagnostics / Visualizations
                  </h3>
                  
                  {/* Tab list */}
                  <div className="flex bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-xs font-mono">
                    {['raw', 'denoised', 'prob', 'phase', 'interactive'].map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setActivePlotTab(tab)}
                        className={`px-3 py-1.5 rounded-md transition uppercase ${
                          activePlotTab === tab 
                            ? 'bg-indigo-600 text-white font-bold shadow' 
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        {tab}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Display Area */}
                <div className="bg-slate-950 p-2 rounded-lg border border-slate-855 min-h-[350px] flex items-center justify-center relative overflow-hidden">
                  {activePlotTab !== 'interactive' ? (
                    <div className="w-full flex flex-col items-center">
                      <img 
                        src={`${API_BASE}/api/results/${selectedTic}/plot/${activePlotTab}`}
                        alt={`Transit Diagnostics: ${activePlotTab}`}
                        className="max-h-[340px] object-contain rounded"
                        onError={(e) => {
                          e.target.onerror = null;
                          e.target.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300"><rect width="100%" height="100%" fill="%23020617"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="%2364748b" font-family="monospace" font-size="14">Plot file loading or not found...</text></svg>';
                        }}
                      />
                      <div className="mt-2 text-center text-xs font-mono text-slate-500">
                        Diagnostics: Matplotlib output {activePlotTab} for Star TIC {selectedTic}
                      </div>
                    </div>
                  ) : (
                    /* Interactive Recharts phase fold mockup */
                    <div className="w-full h-[350px] p-2 flex flex-col justify-between">
                      <div className="flex-1">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart
                            data={interactiveData}
                            margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
                          >
                            <XAxis 
                              dataKey="phase" 
                              stroke="#64748b" 
                              fontSize={10} 
                              tickLine={false} 
                              label={{ value: 'Orbital Phase (Centered at Transit)', position: 'bottom', fill: '#64748b', fontSize: 11 }}
                            />
                            <YAxis 
                              domain={['dataMin - 0.001', 'dataMax + 0.001']} 
                              stroke="#64748b" 
                              fontSize={10} 
                              tickLine={false} 
                              tickFormatter={(val) => val.toFixed(4)}
                              label={{ value: 'Normalized Flux', angle: -90, position: 'left', fill: '#64748b', fontSize: 11 }}
                            />
                            <Tooltip
                              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                              labelFormatter={(label) => `Phase: ${label}`}
                              formatter={(value) => [`${value}`, 'Flux']}
                            />
                            <ReferenceLine x={0} stroke="#f43f5e" strokeDasharray="3 3" />
                            <Area 
                              type="monotone" 
                              dataKey="flux" 
                              stroke="#818cf8" 
                              fill="rgba(99, 102, 241, 0.05)" 
                              dot={false}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="text-center text-xs font-mono text-indigo-400 mt-2 flex items-center justify-center gap-1">
                        <Info className="w-3.5 h-3.5" /> Interactive Phase Fold Simulator (Recharts)
                      </div>
                    </div>
                  )}
                </div>
              </div>

            </div>
          ) : (
            /* Blank state */
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-20 flex flex-col items-center justify-center gap-4 text-center min-h-[500px]">
              <Layers className="w-12 h-12 text-slate-600 animate-pulse" />
              <h3 className="text-lg font-semibold text-slate-300">Telemetry Inspector Offline</h3>
              <p className="text-sm text-slate-400 max-w-md">
                Select a star from the catalog on the left to review transit analytics, neural network probabilities, and diagnostics plots. Or trigger a new run.
              </p>
            </div>
          )}

          {/* STAGE 4: MODEL TRAINING HUB */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 backdrop-blur-md">
            <h2 className="text-sm font-semibold tracking-wider text-slate-300 uppercase mb-4 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-400" /> Neural Network Calibration & Training Desk
            </h2>
            <form onSubmit={handleLaunchTraining} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
              <div className="md:col-span-2">
                <label className="block text-xs font-mono text-slate-400 mb-1">Dataset Path (CSV)</label>
                <input 
                  type="text" 
                  value={trainDataset}
                  onChange={(e) => setTrainDataset(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono transition"
                />
              </div>

              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1">Epochs</label>
                <input 
                  type="number" 
                  value={trainEpochs}
                  onChange={(e) => setTrainEpochs(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono transition"
                />
              </div>

              <div>
                <label className="block text-xs font-mono text-slate-400 mb-1">Batch Size</label>
                <input 
                  type="number" 
                  value={trainBatch}
                  onChange={(e) => setTrainBatch(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono transition"
                />
              </div>

              <div className="md:col-span-2">
                <label className="block text-xs font-mono text-slate-400 mb-1">API Key (`X-API-Key` Auth)</label>
                <input 
                  type="password" 
                  placeholder="Enter API Key to auth training"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono transition"
                />
              </div>

              <div className="md:col-span-2">
                <button 
                  type="submit" 
                  disabled={trainingJob && trainingJob.status !== 'SUCCESS' && trainingJob.status !== 'FAILURE'}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-800 disabled:opacity-50 text-white font-medium rounded-lg px-4 py-2 text-xs transition flex items-center justify-center gap-2 active:scale-95 cursor-pointer h-[32px] shadow shadow-indigo-600/10"
                >
                  <Cpu className="w-3.5 h-3.5" />
                  {trainingJob && trainingJob.status !== 'SUCCESS' && trainingJob.status !== 'FAILURE' ? 'CALIBRATING...' : 'QUEUE TRAINING JOB'}
                </button>
              </div>
            </form>

            {trainingJob && (
              <div className="mt-4 p-3 bg-slate-950/80 rounded-lg border border-slate-800 text-xs font-mono space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Training Task: {trainingJob.id.substring(0, 10)}...</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    trainingJob.status === 'RUNNING' ? 'bg-indigo-500/20 text-indigo-400 animate-pulse' :
                    trainingJob.status === 'PENDING' ? 'bg-amber-500/20 text-amber-400' :
                    trainingJob.status === 'SUCCESS' ? 'bg-emerald-500/20 text-emerald-400' :
                    'bg-rose-500/20 text-rose-400'
                  }`}>{trainingJob.status}</span>
                </div>
                <p className="text-[10px] text-slate-500">
                  Model training is run via Celery worker. When complete, the server will reload with the optimized weights (`classifier_best.pt`).
                </p>
              </div>
            )}
          </div>

        </section>

      </main>

      {/* FOOTER OPERATIONS */}
      <footer className="border-t border-slate-900 bg-slate-950 text-slate-500 py-6 text-center text-xs font-mono">
        <p>© 2026 ExoDetector Astrodynamics Operations Room. Active Satellite Data Stream: NASA TESS.</p>
      </footer>
    </div>
  );
}
