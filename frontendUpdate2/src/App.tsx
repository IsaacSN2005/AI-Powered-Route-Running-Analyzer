import React, { useState, useEffect, Component } from 'react';
import { 
  Upload, 
  ChevronLeft, 
  Share2, 
  Play, 
  RotateCcw, 
  RotateCw, 
  Volume2, 
  Maximize, 
  User as UserIcon, 
  Settings, 
  Info,
  Activity,
  Download,
  LogOut,
  AlertCircle,
  Tag,
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Home from './components/Home';
import type { User, Metric, Analysis, AnalysisMode, AnalyzeRequest, AnalyzeResponse, AnalyzeStartResponse, AnalysisJobResponse, Player, RouteSummary } from './types';

const MOCK_ROSTER: Player[] = [
  { id: 'p1', name: 'James Wilson', position: 'WR', jerseyNumber: '88', photoURL: 'https://picsum.photos/seed/p1/100/100' },
  { id: 'p2', name: 'Marcus Reed', position: 'WR', jerseyNumber: '11', photoURL: 'https://picsum.photos/seed/p2/100/100' },
  { id: 'p3', name: 'Tyler Vance', position: 'TE', jerseyNumber: '85', photoURL: 'https://picsum.photos/seed/p3/100/100' },
  { id: 'p4', name: 'Devin Cole', position: 'WR', jerseyNumber: '14', photoURL: 'https://picsum.photos/seed/p4/100/100' },
];

const UNKNOWN_PLAYER: Player = {
  id: 'unknown',
  name: 'Unknown Athlete',
  position: 'N/A',
  jerseyNumber: '00',
};

const DEFAULT_ROUTE_SUMMARY: RouteSummary = {
  routeGuess: 'Out Route',
  breakStyle: 'Rounded',
  actualPathCutAngleDeg: 54.8,
  fullTurnAngleDeg: 90.0,
  idealizedCutAngleDeg: 90.0,
  peakSpeedMph: 22.2,
  offLine3YdTimeS: 1.27,
  offLine3YdAccelMphPerSec: 10.1,
  cutTimeS: 0.4,
  cutDecelMphPerSec: -22.7,
  hipDropPctBodyHeight: 2.5,
  cutConfidence: 'medium',
  calibrationConfidence: 0.82,
};

function labelForMode(mode: AnalysisMode) {
  return 'Side View';
}

function parseOptionalFrameInput(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || parsed < 0) return null;
  return Math.round(parsed);
}

function formatNumber(value: number | null, digits = 1) {
  if (value === null || Number.isNaN(value)) return 'N/A';
  return Number(value.toFixed(digits));
}

function formatPercent(value: number | null, digits = 0) {
  if (value === null || Number.isNaN(value)) return 'N/A';
  return `${(value * 100).toFixed(digits)}%`;
}

function statusForPeakSpeed(speed: number | null): Metric['status'] | undefined {
  if (speed === null) return undefined;
  if (speed >= 20) return 'Elite';
  if (speed >= 17) return 'Optimal';
  if (speed >= 14) return 'Average';
  return 'Poor';
}

function statusFor3YardTime(time: number | null): Metric['status'] | undefined {
  if (time === null) return undefined;
  if (time <= 0.8) return 'Elite';
  if (time <= 1.0) return 'Optimal';
  if (time <= 1.3) return 'Average';
  return 'Poor';
}

function statusForCutTime(time: number | null): Metric['status'] | undefined {
  if (time === null) return undefined;
  if (time <= 0.28) return 'Elite';
  if (time <= 0.38) return 'Optimal';
  if (time <= 0.5) return 'Average';
  return 'Poor';
}

function statusForHipDrop(drop: number | null): Metric['status'] | undefined {
  if (drop === null) return undefined;
  if (drop >= 3.0) return 'Elite';
  if (drop >= 2.0) return 'Optimal';
  if (drop >= 1.0) return 'Average';
  return 'Poor';
}

function buildMetricsFromSummary(summary: RouteSummary): Metric[] {
  return [
    {
      label: 'Route Guess',
      value: summary.routeGuess || 'Unknown',
      description: 'Best-fit route family based on fitted stem and break geometry.'
    },
    {
      label: 'Break Style',
      value: summary.breakStyle || 'Unknown',
      description: 'Classifies whether the cut geometry looks sharp or rounded.'
    },
    {
      label: 'Actual Cut Angle',
      value: formatNumber(summary.actualPathCutAngleDeg, 1),
      unit: 'deg',
      description: 'Measured change of direction from the tracked path the receiver actually ran.'
    },
    {
      label: 'Idealized Cut Angle',
      value: formatNumber(summary.idealizedCutAngleDeg, 1),
      unit: 'deg',
      description: 'Cleaner stem-to-break angle that approximates the intended route geometry.'
    },
    {
      label: 'Peak Speed',
      value: formatNumber(summary.peakSpeedMph, 1),
      unit: 'mph',
      status: statusForPeakSpeed(summary.peakSpeedMph),
      description: 'Sustained top-end route speed after smoothing and conservative clipping.'
    },
    {
      label: '3-Yard Burst Time',
      value: formatNumber(summary.offLine3YdTimeS, 2),
      unit: 's',
      status: statusFor3YardTime(summary.offLine3YdTimeS),
      description: 'Time it takes the receiver to cover the first 3 yards off the line.'
    },
    {
      label: 'Off-Line Acceleration',
      value: formatNumber(summary.offLine3YdAccelMphPerSec, 1),
      unit: 'mph/s',
      description: 'Average acceleration during the first 3 yards of the release.'
    },
    {
      label: 'Cut Time',
      value: formatNumber(summary.cutTimeS, 2),
      unit: 's',
      status: statusForCutTime(summary.cutTimeS),
      description: 'Duration of the detected change-of-direction window around the break.'
    },
    {
      label: 'Cut Deceleration',
      value: formatNumber(summary.cutDecelMphPerSec, 1),
      unit: 'mph/s',
      description: 'Average speed loss into the cut window. More negative means harder braking.'
    },
    {
      label: 'Hip Drop',
      value: formatNumber(summary.hipDropPctBodyHeight, 1),
      unit: '% body ht',
      status: statusForHipDrop(summary.hipDropPctBodyHeight),
      description: 'Hip sink near the cut, normalized by body height when pose confidence supports it.'
    }
  ];
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

type SetupTool = 'target' | 'calibration';
type SetupCalibrationPoint = {
  imageX: number;
  imageY: number;
  fieldX: number;
  fieldY: number;
};

type DraftCalibrationPoint = {
  imageX: number;
  imageY: number;
};

const DEFAULT_METRICS_TEMPLATE: Metric[] = buildMetricsFromSummary(DEFAULT_ROUTE_SUMMARY);

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: any;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState;
  public props: ErrorBoundaryProps;
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      let message = "Something went wrong.";
      try {
        const parsed = JSON.parse(this.state.error.message);
        if (parsed.error) message = parsed.error;
      } catch (e) {
        message = this.state.error.message || message;
      }

      return (
        <div className="h-screen w-full bg-black flex flex-col items-center justify-center p-6 text-center">
          <AlertCircle className="text-red-500 w-16 h-16 mb-4" />
          <h1 className="text-2xl font-bold text-white mb-2">Application Error</h1>
          <p className="text-slate-400 max-w-md mb-6">{message}</p>
          <button 
            onClick={() => window.location.reload()}
            className="bg-primary text-white px-6 py-2 rounded-lg font-bold"
          >
            Reload Application
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function AppContent() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('Preparing analysis...');
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null);
  const [showGlossary, setShowGlossary] = useState(false);
  const [isTagging, setIsTagging] = useState(false);
  const [tagInput, setTagInput] = useState({ name: '', jersey: '' });
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('side-view');
  const [analysisControls, setAnalysisControls] = useState({
    startFrame: '',
    endFrame: '',
    cutFrame: '',
  });
  const [clipConfirmed, setClipConfirmed] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingVideoUrl, setPendingVideoUrl] = useState<string | null>(null);
  const [setupTool, setSetupTool] = useState<SetupTool>('target');
  const [setupAutoCalibrate, setSetupAutoCalibrate] = useState(true);
  const [setupCalibrationPoints, setSetupCalibrationPoints] = useState<SetupCalibrationPoint[]>([]);
  const [draftCalibrationPoint, setDraftCalibrationPoint] = useState<DraftCalibrationPoint | null>(null);
  const [draftFieldX, setDraftFieldX] = useState('');
  const [draftFieldY, setDraftFieldY] = useState('');
  const [setupTargetPoint, setSetupTargetPoint] = useState<{ x: number; y: number } | null>(null);
  const [jobProgressPreviewUrl, setJobProgressPreviewUrl] = useState<string | null>(null);
  const [jobProgressFrame, setJobProgressFrame] = useState<number | null>(null);
  const [jobTotalFrames, setJobTotalFrames] = useState<number | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const setupVideoRef = React.useRef<HTMLVideoElement>(null);

  const resetSetupState = (revokeUrl = true) => {
    if (revokeUrl && pendingVideoUrl) {
      URL.revokeObjectURL(pendingVideoUrl);
    }
    setPendingFile(null);
    setPendingVideoUrl(null);
    setClipConfirmed(false);
    setSetupTool('target');
    setSetupAutoCalibrate(true);
    setSetupCalibrationPoints([]);
    setDraftCalibrationPoint(null);
    setDraftFieldX('');
    setDraftFieldY('');
    setSetupTargetPoint(null);
    setJobProgressPreviewUrl(null);
    setJobProgressFrame(null);
    setJobTotalFrames(null);
  };

  const handleLoginSuccess = (userData: User) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
  };

  if (loading) {
    return (
      <div className="h-screen w-full bg-black flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 px-6 text-center w-full max-w-5xl">
          <motion.div 
            animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="bg-primary rounded-xl p-4"
          >
            <Activity className="text-white w-12 h-12" />
          </motion.div>
          {pendingVideoUrl && (
            <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <p className="text-xs text-slate-400 mb-2 uppercase tracking-widest font-bold">Confirmed Clip</p>
                <video src={pendingVideoUrl} className="w-full rounded-lg" controls muted playsInline />
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <p className="text-xs text-slate-400 mb-2 uppercase tracking-widest font-bold">Live Tracking Preview</p>
                {jobProgressPreviewUrl ? (
                  <img src={jobProgressPreviewUrl} className="w-full rounded-lg" alt="Live tracking preview" />
                ) : (
                  <div className="h-64 rounded-lg bg-slate-900 flex items-center justify-center text-slate-500 text-sm">
                    Waiting for the first tracked frame...
                  </div>
                )}
              </div>
            </div>
          )}
          <div className="space-y-1">
            <p className="text-white font-bold">Analyzing Side-View Clip</p>
            <p className="text-sm text-slate-400 max-w-md">{loadingMessage}</p>
            {jobProgressFrame !== null && jobTotalFrames !== null && (
              <p className="text-xs text-slate-500">Frame {jobProgressFrame} / {jobTotalFrames}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Home onLoginSuccess={handleLoginSuccess} />;
  }

  const selectedAnalysis = analyses.find(a => a.id === selectedAnalysisId) || analyses[0];
  const selectedSummary = selectedAnalysis?.summary || DEFAULT_ROUTE_SUMMARY;

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setUploadError(null);
    
    if (file && user) {
      // Validate file type
      if (!file.type.startsWith('video/')) {
        setUploadError('Please upload a valid video file (MP4, MOV, WebM).');
        if (fileInputRef.current) fileInputRef.current.value = '';
        return;
      }

      const videoUrl = URL.createObjectURL(file);
      setPendingFile(file);
      setPendingVideoUrl(videoUrl);
      setClipConfirmed(false);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const startAnalysisFromSetup = async () => {
    if (!pendingFile || !user || !pendingVideoUrl) return;
    if (!setupTargetPoint) {
      setUploadError('Click the receiver in the video before starting analysis.');
      return;
    }
    if (!setupAutoCalibrate && setupCalibrationPoints.length < 4) {
      setUploadError('Add at least 4 calibration points or re-enable auto-calibration.');
      return;
    }

    setLoading(true);
    setLoadingMessage('Uploading clip and starting analysis...');

    try {
        const payload: AnalyzeRequest = {
          mode: analysisMode,
          filename: pendingFile.name,
          mimeType: pendingFile.type,
          sizeBytes: pendingFile.size,
          startFrame: parseOptionalFrameInput(analysisControls.startFrame),
          endFrame: parseOptionalFrameInput(analysisControls.endFrame),
          cutFrame: parseOptionalFrameInput(analysisControls.cutFrame),
        };
        const setupPayload = {
          autoCalibrate: setupAutoCalibrate,
          calibrationPoints: setupCalibrationPoints.map((point) => ({
            image_x: point.imageX,
            image_y: point.imageY,
            field_x: point.fieldX,
            field_y: point.fieldY,
          })),
          targetPoint: setupTargetPoint,
        };

        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': pendingFile.type || 'application/octet-stream',
            'X-Mode': analysisMode,
            'X-Filename': encodeURIComponent(pendingFile.name),
            'X-File-Size': String(pendingFile.size),
            ...(payload.startFrame !== null ? { 'X-Start-Frame': String(payload.startFrame) } : {}),
            ...(payload.endFrame !== null ? { 'X-End-Frame': String(payload.endFrame) } : {}),
            ...(payload.cutFrame !== null ? { 'X-Cut-Frame': String(payload.cutFrame) } : {}),
            'X-Analysis-Setup': encodeURIComponent(JSON.stringify(setupPayload)),
          },
          body: pendingFile,
        });

        if (!response.ok) {
          throw new Error(`Analysis request failed (${response.status})`);
        }

        const start: AnalyzeStartResponse = await response.json();
        setLoadingMessage(start.message);

        let result: AnalyzeResponse | undefined;
        while (!result) {
          await sleep(250);
          const jobResponse = await fetch(`/api/analyze/${start.jobId}`);
          if (!jobResponse.ok) {
            throw new Error(`Job status request failed (${jobResponse.status})`);
          }
          const job: AnalysisJobResponse = await jobResponse.json();
          setLoadingMessage(job.message);
          setJobProgressPreviewUrl(job.progressPreviewUrl || null);
          setJobProgressFrame(job.progressFrame ?? null);
          setJobTotalFrames(job.totalFrames ?? null);
          if (job.status === 'failed') {
            throw new Error(job.error || 'Analysis failed.');
          }
          if (job.status === 'completed' && job.result) {
            result = job.result;
          }
        }

        const newId = Date.now().toString();
        const newAnalysis: Analysis = {
          id: newId,
          uid: user.uid,
          playerId: UNKNOWN_PLAYER.id,
          mode: result.mode,
          playerSnapshot: UNKNOWN_PLAYER,
          title: pendingFile.name.split('.')[0] || 'New Analysis',
          time: 'Just now',
          score: result.score,
          image: result.image,
          videoUrl: pendingVideoUrl,
          metrics: result.metrics.length > 0 ? result.metrics : buildMetricsFromSummary(result.summary),
          summary: result.summary,
          routeDebugPlotUrl: result.routeDebugPlotUrl,
          cleanCsvUrl: result.cleanCsvUrl,
          posePointsCsvUrl: result.posePointsCsvUrl,
          repCleanCsvUrl: result.repCleanCsvUrl,
          summaryCsvUrl: result.summaryCsvUrl,
          createdAt: new Date(result.analyzedAt),
        };

        setAnalyses(prev => [newAnalysis, ...prev]);
        setSelectedAnalysisId(newId);
        resetSetupState(false);
      } catch (error) {
        resetSetupState();
        setUploadError(error instanceof Error ? error.message : 'Unable to analyze this clip right now.');
      } finally {
        setLoading(false);
      }
  };

  const handleTagAthlete = () => {
    if (!selectedAnalysisId || !tagInput.name) return;

    setAnalyses(prev => prev.map(analysis => {
      if (analysis.id === selectedAnalysisId) {
        return {
          ...analysis,
          playerSnapshot: {
            ...analysis.playerSnapshot,
            name: tagInput.name,
            jerseyNumber: tagInput.jersey || '00',
            id: `p-${Date.now()}`
          }
        };
      }
      return analysis;
    }));
    setIsTagging(false);
    setTagInput({ name: '', jersey: '' });
  };

  const handleSetupVideoClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const video = setupVideoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;
    if (!clipConfirmed) return;

    const rect = video.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * video.videoWidth;
    const y = ((event.clientY - rect.top) / rect.height) * video.videoHeight;

    if (setupTool === 'target') {
      setSetupTargetPoint({ x, y });
      return;
    }
    setDraftCalibrationPoint({ imageX: x, imageY: y });
    setDraftFieldX('');
    setDraftFieldY('');
  };

  const addDraftCalibrationPoint = () => {
    if (!draftCalibrationPoint) return;
    const fieldX = Number(draftFieldX);
    const fieldY = Number(draftFieldY);
    if (!Number.isFinite(fieldX) || !Number.isFinite(fieldY)) {
      setUploadError('Calibration point skipped because the field coordinates were invalid.');
      return;
    }
    setSetupCalibrationPoints((prev) => [
      ...prev,
      {
        imageX: draftCalibrationPoint.imageX,
        imageY: draftCalibrationPoint.imageY,
        fieldX,
        fieldY,
      },
    ]);
    setDraftCalibrationPoint(null);
    setDraftFieldX('');
    setDraftFieldY('');
  };

  const handleExport = () => {
    if (!selectedAnalysis || !user) return;
    
    const timestamp = new Date().toLocaleString();
    const reportHeader = `
==================================================
ROUTEIQ PERFORMANCE ANALYSIS REPORT
==================================================
Athlete: ${selectedAnalysis.playerSnapshot.name} (#${selectedAnalysis.playerSnapshot.jerseyNumber})
Analysis Title: ${selectedAnalysis.title}
Date Generated: ${timestamp}
Overall Performance Score: ${selectedAnalysis.score}/100
--------------------------------------------------
`.trim();

    const metricsContent = selectedAnalysis.metrics.map(m => `
[${m.label}]
Value: ${m.value}${m.unit || ''}
Status: ${m.status || 'N/A'}
Description: ${m.description}
`).join('\n');

    const reportFooter = `
--------------------------------------------------
Generated by RouteIQ Analytics
Professional-grade route running analysis.
==================================================
`.trim();

    const fullReport = `${reportHeader}\n${metricsContent}\n${reportFooter}`;

    const blob = new Blob([fullReport], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `RouteIQ-Report-${selectedAnalysis.title.replace(/\s+/g, '-')}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background-light dark:bg-background-dark">
      {/* Hidden File Input */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept="video/*"
        className="hidden" 
      />

      {/* Error Notification */}
      <AnimatePresence>
        {uploadError && (
          <motion.div 
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 20 }}
            exit={{ opacity: 0, y: -50 }}
            className="fixed top-0 left-1/2 -translate-x-1/2 z-[100] bg-destructive text-white px-6 py-3 rounded-xl shadow-2xl flex items-center gap-3 font-bold text-sm"
          >
            <AlertCircle className="w-5 h-5" />
            {uploadError}
            <button onClick={() => setUploadError(null)} className="ml-2 hover:opacity-80">
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-slate-50 dark:bg-slate-900/50 border-r border-slate-200 dark:border-slate-800 flex flex-col">
        <div className="p-6">
          <div 
            onClick={handleLogout}
            className="flex items-center gap-3 mb-8 cursor-pointer hover:opacity-80 transition-opacity"
          >
            <div className="bg-primary rounded-lg p-1.5 flex items-center justify-center">
              <Activity className="text-white w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">RouteIQ</h1>
          </div>

          <button 
            onClick={handleUploadClick}
            className="w-full flex items-center justify-center gap-2 bg-primary hover:bg-primary/90 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors mb-2 shadow-lg shadow-primary/20"
          >
            <Upload className="w-5 h-5" />
            <span>Upload Video</span>
          </button>
          <p className="text-[10px] text-slate-500 text-center mb-8 font-medium italic">
            * Side-view mode is best for break mechanics, hip drop, and release detail.
          </p>

          <div className="mb-8 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3 space-y-3">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Analysis Controls</p>
              <p className="text-[11px] text-slate-400 mt-1">Optional frame overrides if the auto rep window or cut is messy.</p>
            </div>
            <div className="space-y-2">
              <input
                type="number"
                min="0"
                placeholder="Start frame"
                value={analysisControls.startFrame}
                onChange={(e) => setAnalysisControls((prev) => ({ ...prev, startFrame: e.target.value }))}
                className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 px-3 py-2 text-xs"
              />
              <input
                type="number"
                min="0"
                placeholder="End frame"
                value={analysisControls.endFrame}
                onChange={(e) => setAnalysisControls((prev) => ({ ...prev, endFrame: e.target.value }))}
                className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 px-3 py-2 text-xs"
              />
              <input
                type="number"
                min="0"
                placeholder="Cut frame"
                value={analysisControls.cutFrame}
                onChange={(e) => setAnalysisControls((prev) => ({ ...prev, cutFrame: e.target.value }))}
                className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 px-3 py-2 text-xs"
              />
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 px-2">Recent Analyses</h3>
            <div className="space-y-1 overflow-y-auto no-scrollbar max-h-[calc(100vh-320px)]">
              {analyses.length === 0 && (
                <div className="text-center py-8 px-4">
                  <p className="text-xs text-slate-400 italic">No analyses yet. Upload a side-view clip to start analysis.</p>
                </div>
              )}
              {analyses.map((analysis) => (
                <motion.div 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  key={analysis.id}
                  onClick={() => setSelectedAnalysisId(analysis.id)}
                  className={`group flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all ${
                    analysis.id === selectedAnalysisId 
                      ? 'bg-slate-200 dark:bg-slate-800 border border-primary/20 shadow-sm' 
                      : 'hover:bg-slate-200 dark:hover:bg-slate-800'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-bold truncate text-slate-900 dark:text-white">{analysis.title}</p>
                    <p className="text-[10px] text-slate-500 font-bold uppercase mt-0.5">
                      {analysis.playerSnapshot.name} • #{analysis.playerSnapshot.jerseyNumber}
                    </p>
                    <p className="text-[10px] text-slate-400 mt-1">
                      {analysis.time} • {analysis.score}% Score • {labelForMode(analysis.mode)}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-16 flex items-center justify-between px-8 border-b border-slate-200 dark:border-slate-800 shrink-0 bg-white/50 dark:bg-slate-900/50 backdrop-blur-md">
          <div className="flex items-center gap-4">
            <ChevronLeft className="text-slate-400 w-6 h-6 cursor-pointer hover:text-slate-600" />
            <div className="flex flex-col">
              <h2 className="text-lg font-bold leading-tight">Analysis: {selectedAnalysis?.title || 'No Analysis Selected'}</h2>
              {selectedAnalysis && (
                <div className="flex items-center gap-2">
                  <p className="text-[10px] font-bold text-primary uppercase tracking-widest">
                    Athlete: {selectedAnalysis.playerSnapshot.name} (#{selectedAnalysis.playerSnapshot.jerseyNumber})
                  </p>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">
                    {labelForMode(selectedAnalysis.mode)}
                  </span>
                  {selectedAnalysis.summary?.routeConfidence && (
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">
                      Route Confidence: {selectedAnalysis.summary.routeConfidence}
                    </span>
                  )}
                  {selectedAnalysis.playerSnapshot.id === 'unknown' && !isTagging && (
                    <motion.button 
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setIsTagging(true)}
                      className="flex items-center gap-1.5 text-[10px] font-black bg-primary text-white px-3 py-1 rounded-full shadow-lg shadow-primary/30 hover:bg-primary/90 transition-all ml-2"
                    >
                      <Tag className="w-3 h-3" />
                      TAG ATHLETE
                    </motion.button>
                  )}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button 
              onClick={handleExport}
              disabled={!selectedAnalysis}
              className="flex items-center gap-2 bg-slate-900 dark:bg-slate-100 text-slate-100 dark:text-slate-900 px-4 py-2 rounded-lg text-sm font-bold hover:opacity-90 transition-opacity disabled:opacity-50"
            >
              <Download className="w-4 h-4" />
              Export Report
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 no-scrollbar">
          {pendingVideoUrl && pendingFile && (
            <section className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
              <div className="flex flex-col gap-6 lg:grid lg:grid-cols-[1fr_320px]">
                <div>
                  <div className="space-y-1 mb-4">
                    <h3 className="text-lg font-bold">Set Up Analysis</h3>
                    <p className="text-sm text-slate-500">
                      {!clipConfirmed ? 'Preview and confirm the clip first.' : 'Use the paused first frame to click the receiver and add calibration points.'}
                    </p>
                  </div>
                  <div className="relative inline-block max-w-full cursor-crosshair" onClick={handleSetupVideoClick}>
                    <video
                      ref={setupVideoRef}
                      src={pendingVideoUrl}
                      className="max-h-[70vh] max-w-full rounded-xl bg-black"
                      controls={!clipConfirmed}
                      muted
                      playsInline
                      onLoadedMetadata={(event) => {
                        event.currentTarget.currentTime = 0;
                        event.currentTarget.pause();
                      }}
                      onPlay={(event) => {
                        if (clipConfirmed) {
                          event.currentTarget.pause();
                        }
                      }}
                    />
                    <div className="pointer-events-none absolute inset-0">
                      {setupCalibrationPoints.map((point, index) => (
                        <div
                          key={`${point.imageX}-${point.imageY}-${index}`}
                          className="absolute -translate-x-1/2 -translate-y-1/2"
                          style={{
                            left: `${(point.imageX / (setupVideoRef.current?.videoWidth || 1)) * 100}%`,
                            top: `${(point.imageY / (setupVideoRef.current?.videoHeight || 1)) * 100}%`,
                          }}
                        >
                          <div className="w-4 h-4 rounded-full bg-cyan-400 border-2 border-white shadow-lg" />
                          <div className="mt-1 text-[10px] font-bold text-white whitespace-nowrap">
                            {index + 1}: ({point.fieldX}, {point.fieldY})
                          </div>
                        </div>
                      ))}
                      {draftCalibrationPoint && (
                        <div
                          className="absolute -translate-x-1/2 -translate-y-1/2"
                          style={{
                            left: `${(draftCalibrationPoint.imageX / (setupVideoRef.current?.videoWidth || 1)) * 100}%`,
                            top: `${(draftCalibrationPoint.imageY / (setupVideoRef.current?.videoHeight || 1)) * 100}%`,
                          }}
                        >
                          <div className="w-4 h-4 rounded-full bg-amber-400 border-2 border-white shadow-lg" />
                        </div>
                      )}
                      {setupTargetPoint && (
                        <div
                          className="absolute -translate-x-1/2 -translate-y-1/2"
                          style={{
                            left: `${(setupTargetPoint.x / (setupVideoRef.current?.videoWidth || 1)) * 100}%`,
                            top: `${(setupTargetPoint.y / (setupVideoRef.current?.videoHeight || 1)) * 100}%`,
                          }}
                        >
                          <div className="w-5 h-5 rounded-full bg-primary border-2 border-white shadow-xl" />
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 p-4 space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Click Mode</p>
                    {!clipConfirmed && (
                      <button
                        onClick={() => {
                          if (setupVideoRef.current) {
                            setupVideoRef.current.pause();
                            setupVideoRef.current.currentTime = 0;
                          }
                          setClipConfirmed(true);
                        }}
                        className="w-full rounded-xl bg-primary px-3 py-2 text-sm font-bold text-white"
                      >
                        Confirm This Clip
                      </button>
                    )}
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSetupTool('target')}
                        disabled={!clipConfirmed}
                        className={`flex-1 rounded-xl px-3 py-2 text-sm font-bold ${setupTool === 'target' ? 'bg-primary text-white' : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-800'} ${!clipConfirmed ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        Select Receiver
                      </button>
                      <button
                        onClick={() => setSetupTool('calibration')}
                        disabled={!clipConfirmed}
                        className={`flex-1 rounded-xl px-3 py-2 text-sm font-bold ${setupTool === 'calibration' ? 'bg-primary text-white' : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-800'} ${!clipConfirmed ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        Add Calibration
                      </button>
                    </div>
                    <label className="flex items-center gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-4 py-3">
                      <input
                        type="checkbox"
                        checked={setupAutoCalibrate}
                        onChange={(e) => setSetupAutoCalibrate(e.target.checked)}
                      />
                      <div>
                        <p className="text-sm font-bold">Use auto-calibration fallback</p>
                        <p className="text-xs text-slate-500">Leave this on unless you want to provide all field points manually.</p>
                      </div>
                    </label>
                  </div>

                  {draftCalibrationPoint && (
                    <div className="rounded-xl border border-amber-300/40 bg-amber-500/10 p-4 space-y-3">
                      <p className="text-sm font-bold text-amber-200">Add Calibration Point</p>
                      <p className="text-xs text-amber-100/80">
                        Clicked image point: ({draftCalibrationPoint.imageX.toFixed(1)}, {draftCalibrationPoint.imageY.toFixed(1)})
                      </p>
                      <div className="grid grid-cols-2 gap-2">
                        <input
                          type="number"
                          placeholder="Field X (yd)"
                          value={draftFieldX}
                          onChange={(e) => setDraftFieldX(e.target.value)}
                          className="rounded-lg border border-amber-300/30 bg-slate-950/40 px-3 py-2 text-sm text-white"
                        />
                        <input
                          type="number"
                          placeholder="Field Y (yd)"
                          value={draftFieldY}
                          onChange={(e) => setDraftFieldY(e.target.value)}
                          className="rounded-lg border border-amber-300/30 bg-slate-950/40 px-3 py-2 text-sm text-white"
                        />
                      </div>
                      <div className="flex gap-2">
                        <button onClick={addDraftCalibrationPoint} className="flex-1 rounded-xl bg-primary px-3 py-2 text-sm font-bold text-white">Save Point</button>
                        <button onClick={() => setDraftCalibrationPoint(null)} className="flex-1 rounded-xl border border-amber-300/30 px-3 py-2 text-sm font-bold text-amber-100">Cancel</button>
                      </div>
                    </div>
                  )}

                  <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 p-4 space-y-2">
                    <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Setup Status</p>
                    <p className="text-sm font-medium">Clip: {pendingFile.name}</p>
                    <p className="text-sm">Receiver selected: {setupTargetPoint ? 'Yes' : 'No'}</p>
                    <p className="text-sm">Calibration points: {setupCalibrationPoints.length}</p>
                  </div>

                  <div className="flex gap-2">
                    <button onClick={() => setSetupCalibrationPoints([])} className="flex-1 rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold">
                      Clear Points
                    </button>
                    <button onClick={() => setSetupTargetPoint(null)} className="flex-1 rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold">
                      Clear Receiver
                    </button>
                  </div>

                  <div className="flex gap-2">
                    <button onClick={() => resetSetupState()} className="flex-1 rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold">
                      Cancel
                    </button>
                    <button onClick={startAnalysisFromSetup} className="flex-1 rounded-xl bg-primary px-3 py-2 text-sm font-bold text-white">
                      Start Analysis
                    </button>
                  </div>
                </div>
              </div>
            </section>
          )}

          {!selectedAnalysis ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400">
              <Activity className="w-16 h-16 mb-4 opacity-20" />
              <p>Select an analysis to view performance data</p>
            </div>
          ) : (
            <>
              {/* Video Player Section */}
              <section className="relative aspect-video w-full max-h-[500px] bg-black rounded-xl overflow-hidden shadow-2xl border border-slate-800 group">
                {selectedAnalysis.videoUrl ? (
                  <video 
                    src={selectedAnalysis.videoUrl}
                    className="w-full h-full object-contain"
                    controls
                    autoPlay
                    muted
                  />
                ) : (
                  <>
                    <img 
                      className="w-full h-full object-cover opacity-60" 
                      src="https://lh3.googleusercontent.com/aida-public/AB6AXuDV6fN6kXyWzcmfShRvE9AveOpNPfWuGX9DxYCGy9un7AkQe1yKdSqZookn7JIvcz_qL3CJsMTgDw5rZ9fSU5gpyqeSU5iEBtsoXLvGYwq-3od2jap39eI26MDxxPDuO78QE2eh68gBG0LIAwNPvK300zv6yA20LEfnVVmNA1MQkNEP5zEMQoBGh87uCJI5pa8VAyvQ-H9qZaCDo2uMDnz1od3nEAKt3pUFB2TomiNEmFxpS_Yrz24rz_Qk_y5YxcKXUEG2F5FGenM" 
                      alt="Sideline view of an American football game"
                      referrerPolicy="no-referrer"
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <motion.button 
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="w-20 h-20 bg-primary/90 text-white rounded-full flex items-center justify-center shadow-xl backdrop-blur-sm"
                      >
                        <Play className="w-10 h-10 fill-current ml-1" />
                      </motion.button>
                    </div>
                  </>
                )}

                {/* Overlay Metrics */}
                {/* Removed AI TRACKING ACTIVE label as requested */}

                {!selectedAnalysis.videoUrl && (
                  /* Playback Controls (Only for placeholder) */
                  <div className="absolute bottom-0 inset-x-0 p-4 bg-gradient-to-t from-black/90 to-transparent">
                    <div className="flex items-center gap-4 mb-3">
                      <span className="text-xs text-white font-mono">00:32 / 02:45</span>
                      <div className="flex-1 h-1 bg-white/20 rounded-full relative cursor-pointer group">
                        <div className="absolute top-0 left-0 w-1/3 h-full bg-primary rounded-full"></div>
                        <div className="absolute top-1/2 left-1/3 -translate-y-1/2 w-3 h-3 bg-white border-2 border-primary rounded-full shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"></div>
                      </div>
                    </div>
                    <div className="flex justify-between items-center text-white">
                      <div className="flex items-center gap-6">
                        <RotateCcw className="w-5 h-5 cursor-pointer hover:text-primary transition-colors" />
                        <Play className="w-6 h-6 cursor-pointer hover:text-primary fill-current" />
                        <RotateCw className="w-5 h-5 cursor-pointer hover:text-primary transition-colors" />
                      </div>
                      <div className="flex items-center gap-4">
                        <Volume2 className="w-5 h-5 cursor-pointer hover:text-primary transition-colors" />
                        <Maximize className="w-5 h-5 cursor-pointer hover:text-primary transition-colors" />
                      </div>
                    </div>
                  </div>
                )}
              </section>

              {/* 2D Route Visualization */}
              <section className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm overflow-hidden">
                <div className="flex justify-between items-center mb-6">
                  <div className="space-y-1">
                    <h3 className="text-lg font-bold">2D Route Visualization</h3>
                    <p className="text-xs text-slate-500 font-medium">Structural diagram of route stem and break points</p>
                  </div>
                  <div className="flex gap-4 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-primary"></div>Route Path</div>
                    <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-orange-400"></div>Break Point</div>
                  </div>
                </div>
                
                <div className="relative w-full h-80 bg-slate-50 dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden">
                  {selectedAnalysis.routeDebugPlotUrl ? (
                    <img
                      src={selectedAnalysis.routeDebugPlotUrl}
                      alt="Route debug visualization"
                      className="w-full h-full object-contain bg-slate-950"
                    />
                  ) : (
                    <>
                  {/* Subtle Field Markings */}
                  <div className="absolute inset-0 opacity-20 pointer-events-none">
                    <div className="absolute inset-x-0 h-px bg-slate-400 top-1/4"></div>
                    <div className="absolute inset-x-0 h-px bg-slate-400 top-2/4"></div>
                    <div className="absolute inset-x-0 h-px bg-slate-400 top-3/4"></div>
                    <div className="absolute inset-y-0 w-px bg-slate-400 left-10"></div>
                  </div>
                  
                  {/* Route Diagram SVG */}
                  <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none" viewBox="0 0 800 320">
                    <defs>
                      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#137fec" />
                      </marker>
                    </defs>

                    {/* Vertical Stem */}
                    <motion.line 
                      initial={{ y2: 280 }}
                      animate={{ y2: 140 }}
                      transition={{ duration: 1, ease: "linear" }}
                      x1="50" y1="280" x2="50" y2="140" 
                      stroke="#137fec" 
                      strokeWidth="3" 
                      strokeDasharray="8 4"
                      className="opacity-40"
                    />

                    {/* Main Route Path */}
                    <motion.path 
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 1.5, ease: "easeInOut" }}
                      d="M50,280 L50,140 L340,140" 
                      fill="none" 
                      stroke="#137fec" 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth="4" 
                    />

                    {/* Break Point Emphasis */}
                    <motion.circle 
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.8, type: "spring" }}
                      cx="50" cy="140" fill="#fb923c" r="8" 
                      className="shadow-lg"
                    />
                    
                    {/* Labels */}
                    <g className="font-bold text-[10px] uppercase tracking-widest">
                      <text x="65" y="275" fill="#64748b">Release</text>
                      <text x="65" y="135" fill="#fb923c">
                        {`The Break (${formatNumber(selectedSummary.actualPathCutAngleDeg, 1)}°)`}
                      </text>
                      <text x="300" y="135" fill="#137fec">{selectedSummary.routeGuess}</text>
                    </g>
                  </svg>

                  {/* Contextual Labels */}
                  <div className="absolute bottom-4 left-4 text-[9px] font-black text-slate-400 dark:text-slate-700 uppercase tracking-[0.2em]">Sideline Boundary</div>
                  <div className="absolute top-4 left-4 text-[9px] font-black text-slate-400 dark:text-slate-700 uppercase tracking-[0.2em]">Line of Scrimmage</div>
                    </>
                  )}
                </div>
                {selectedAnalysis.summary?.routeReason && (
                  <p className="mt-4 text-sm text-slate-500 dark:text-slate-400">{selectedAnalysis.summary.routeReason}</p>
                )}
              </section>
            </>
          )}
        </div>
      </main>

      {/* Right Analytics Panel */}
      <aside className="w-80 flex-shrink-0 bg-slate-50 dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 overflow-y-auto no-scrollbar">
        <div className="p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-slate-900 dark:text-white uppercase text-xs tracking-widest">Performance Metrics</h3>
            <button
              onClick={() => setShowGlossary(true)}
              className="text-[10px] font-bold uppercase tracking-widest text-slate-500 hover:text-slate-900 dark:hover:text-white"
            >
              Glossary
            </button>
          </div>

          {selectedAnalysis?.summary?.calibrationWarning && (
            <div className="rounded-xl border border-amber-300/30 bg-amber-500/10 p-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5" />
                <div>
                  <p className="text-xs font-bold uppercase tracking-wider text-amber-300">
                    {selectedAnalysis.summary.calibrationWarning.replaceAll('_', ' ')}
                  </p>
                  <p className="text-xs text-amber-100/80 mt-1">
                    {selectedAnalysis.summary.calibrationWarningDetail}
                  </p>
                </div>
              </div>
            </div>
          )}

          {selectedAnalysis?.summary && (
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Calibration Confidence</p>
                <p className="text-xl font-bold mt-1">{formatPercent(selectedAnalysis.summary.calibrationConfidence ?? null)}</p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Trim Mode</p>
                <p className="text-xl font-bold mt-1">{selectedAnalysis.summary.repTrimMode || 'auto'}</p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Rep Frames</p>
                <p className="text-sm font-bold mt-1">
                  {selectedAnalysis.summary.repStartFrame ?? 'N/A'} - {selectedAnalysis.summary.repEndFrame ?? 'N/A'}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Break Side</p>
                <p className="text-xl font-bold mt-1">{selectedAnalysis.summary.breakSide || 'N/A'}</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4">
            {selectedAnalysis?.metrics.map((metric, idx) => (
              <motion.div 
                key={metric.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="bg-white dark:bg-slate-800 p-4 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{metric.label}</span>
                  {metric.status && (
                    <span className={`text-[9px] font-black px-1.5 py-0.5 rounded uppercase tracking-tighter ${
                      metric.status === 'Elite' ? 'text-green-500 bg-green-500/10' :
                      metric.status === 'Optimal' ? 'text-blue-500 bg-blue-500/10' :
                      'text-yellow-500 bg-yellow-500/10'
                    }`}>
                      {metric.status}
                    </span>
                  )}
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-slate-900 dark:text-white">{metric.value}</span>
                  <span className="text-slate-400 text-xs font-medium">{metric.unit}</span>
                </div>
                <p className="text-[11px] text-slate-500 mt-2 leading-relaxed font-medium">
                  {metric.description}
                </p>
              </motion.div>
            ))}
          </div>

          {selectedAnalysis && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/40 p-4 space-y-3">
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Artifacts</p>
              <div className="grid grid-cols-1 gap-2 text-sm">
                {selectedAnalysis.routeDebugPlotUrl && <a className="text-primary hover:underline" href={selectedAnalysis.routeDebugPlotUrl} target="_blank" rel="noreferrer">Open route debug plot</a>}
                {selectedAnalysis.summaryCsvUrl && <a className="text-primary hover:underline" href={selectedAnalysis.summaryCsvUrl} target="_blank" rel="noreferrer">Open summary CSV</a>}
                {selectedAnalysis.cleanCsvUrl && <a className="text-primary hover:underline" href={selectedAnalysis.cleanCsvUrl} target="_blank" rel="noreferrer">Open clean metrics CSV</a>}
                {selectedAnalysis.posePointsCsvUrl && <a className="text-primary hover:underline" href={selectedAnalysis.posePointsCsvUrl} target="_blank" rel="noreferrer">Open pose points CSV</a>}
                {selectedAnalysis.repCleanCsvUrl && <a className="text-primary hover:underline" href={selectedAnalysis.repCleanCsvUrl} target="_blank" rel="noreferrer">Open rep-only metrics CSV</a>}
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Tag Athlete Modal */}
      <AnimatePresence>
        {isTagging && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsTagging(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="relative bg-white dark:bg-slate-900 w-full max-w-md rounded-3xl shadow-2xl overflow-hidden border border-slate-200 dark:border-slate-800"
            >
              <div className="p-6 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <UserIcon className="w-5 h-5 text-primary" />
                  </div>
                  <h3 className="text-lg font-bold">Tag Athlete</h3>
                </div>
                <button 
                  onClick={() => setIsTagging(false)}
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                    Athlete Name
                  </label>
                  <input 
                    type="text"
                    placeholder="e.g. Justin Jefferson"
                    value={tagInput.name}
                    onChange={(e) => setTagInput(prev => ({ ...prev, name: e.target.value }))}
                    className="w-full bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                    Jersey Number
                  </label>
                  <input 
                    type="text"
                    placeholder="e.g. 18"
                    value={tagInput.jersey}
                    onChange={(e) => setTagInput(prev => ({ ...prev, jersey: e.target.value }))}
                    className="w-full bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                  />
                </div>
              </div>

              <div className="p-6 bg-slate-50 dark:bg-slate-800/50 border-t border-slate-100 dark:border-slate-800 flex gap-3">
                <button 
                  onClick={() => setIsTagging(false)}
                  className="flex-1 px-4 py-2.5 rounded-xl text-sm font-bold text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                >
                  Cancel
                </button>
                <button 
                  onClick={handleTagAthlete}
                  disabled={!tagInput.name}
                  className="flex-1 px-4 py-2.5 rounded-xl text-sm font-bold bg-primary text-white hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-primary/20"
                >
                  Save Tag
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Metric Glossary Modal */}
      <AnimatePresence>
        {showGlossary && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowGlossary(false)}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-2xl bg-white dark:bg-slate-900 rounded-2xl shadow-2xl overflow-hidden border border-slate-200 dark:border-slate-800"
            >
              <div className="p-6 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div className="bg-primary/10 p-2 rounded-lg">
                    <Info className="text-primary w-5 h-5" />
                  </div>
                  <h2 className="text-xl font-bold">Metric Glossary</h2>
                </div>
                <button 
                  onClick={() => setShowGlossary(false)}
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              <div className="p-6 max-h-[60vh] overflow-y-auto no-scrollbar space-y-6">
                {DEFAULT_METRICS_TEMPLATE.map((metric) => (
                  <div key={metric.label} className="space-y-1">
                    <h4 className="font-bold text-primary text-sm uppercase tracking-wider">{metric.label}</h4>
                    <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">
                      {metric.description}
                    </p>
                    <div className="pt-2 flex gap-2">
                      <span className="text-[10px] font-bold bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded uppercase text-slate-500">
                        Unit: {metric.unit || 'N/A'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="p-6 bg-slate-50 dark:bg-slate-950 border-t border-slate-200 dark:border-slate-800">
                <button 
                  onClick={() => setShowGlossary(false)}
                  className="w-full bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 font-bold py-3 rounded-xl hover:opacity-90 transition-opacity"
                >
                  Got it
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
}
