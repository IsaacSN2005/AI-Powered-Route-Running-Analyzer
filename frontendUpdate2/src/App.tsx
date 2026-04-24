import React, { useState, useEffect, Component } from 'react';
import { 
  Upload, 
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
  cutSpeedDropPct: 18.5,
  brakeDecelMphPerSec: -14.2,
  brakeSpeedDropPct: 21.4,
  hipDropPctBodyHeight: 2.5,
  cutConfidence: 'medium',
  calibrationConfidence: 0.82,
  speedConfidence: 'medium',
  speedConfidenceDetail: 'Speed is usable for comparison, but exact mph can drift depending on clip distance and field visibility.',
  distanceProfile: 'medium',
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

function formatRunTimestamp(value: Date | string | undefined) {
  if (!value) return 'Unknown time';
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return 'Unknown time';
  return date.toLocaleString([], {
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function formatVideoTime(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return 'Not set';
  const totalSeconds = Math.max(0, seconds);
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds - mins * 60;
  return `${mins}:${secs.toFixed(2).padStart(5, '0')}`;
}

function buildVideoSegmentSrc(videoUrl: string | null, startTimeS: number | null, endTimeS: number | null) {
  if (!videoUrl) return '';
  if (startTimeS == null && endTimeS == null) return videoUrl;
  const start = Math.max(0, startTimeS ?? 0);
  if (endTimeS != null && Number.isFinite(endTimeS) && endTimeS > start) {
    return `${videoUrl}#t=${start.toFixed(3)},${endTimeS.toFixed(3)}`;
  }
  return `${videoUrl}#t=${start.toFixed(3)}`;
}

function describeUploadError(error: string) {
  const normalized = error.toLowerCase();
  if (normalized.includes('no detected player was under the selected point')) {
    return {
      title: 'Receiver Not Detected',
      message: 'The app did not find a detection box on the player you clicked in that frame.',
      hint: 'Try a nearby frame where the receiver is more visible, then click again.',
    };
  }
  if (normalized.includes('click the receiver in the video before starting analysis')) {
    return {
      title: 'Receiver Needed',
      message: 'Pick the receiver on the frozen setup frame before starting analysis.',
      hint: 'Click directly on the player you want tracked.',
    };
  }
  if (normalized.includes('trim start has to come before trim end')) {
    return {
      title: 'Trim Range Not Valid',
      message: 'The selected start comes after the selected end.',
      hint: 'Move the start earlier or the end later.',
    };
  }
  if (normalized.includes('please upload a valid video file')) {
    return {
      title: 'Video File Needed',
      message: 'RouteIQ only accepts video files for analysis.',
      hint: 'Upload an MP4, MOV, or WebM clip.',
    };
  }
  if (normalized.includes('analysis request failed') || normalized.includes('job status request failed')) {
    return {
      title: 'Analysis Didn’t Start',
      message: 'The app could not start or check this analysis run.',
      hint: 'Try the upload again in a moment.',
    };
  }
  return {
    title: 'Analysis Issue',
    message: error,
    hint: 'Try another frame or rerun the clip.',
  };
}

function statusForPeakSpeed(speed: number | null): Metric['status'] | undefined {
  if (speed === null) return undefined;
  if (speed >= 20) return 'High';
  if (speed >= 17) return 'Solid';
  if (speed >= 14) return 'Moderate';
  return 'Low';
}

function statusFor3YardTime(time: number | null): Metric['status'] | undefined {
  if (time === null) return undefined;
  if (time <= 0.8) return 'High';
  if (time <= 1.0) return 'Solid';
  if (time <= 1.3) return 'Moderate';
  return 'Low';
}

function statusForCutTime(time: number | null): Metric['status'] | undefined {
  if (time === null) return undefined;
  if (time <= 0.28) return 'High';
  if (time <= 0.38) return 'Solid';
  if (time <= 0.5) return 'Moderate';
  return 'Low';
}

function statusForHipDrop(drop: number | null): Metric['status'] | undefined {
  if (drop === null) return undefined;
  if (drop >= 3.0) return 'High';
  if (drop >= 2.0) return 'Solid';
  if (drop >= 1.0) return 'Moderate';
  return 'Low';
}

function softenStatus(status: Metric['status'] | undefined): Metric['status'] | undefined {
  return status;
}

function displayedCutAngle(summary: RouteSummary) {
  const route = (summary.routeGuess || '').toLowerCase();
  const signed = summary.signedTurnAngleDeg;
  const full = summary.fullTurnAngleDeg;
  const ideal = summary.idealizedCutAngleDeg;
  const fieldDirection = (summary.breakFieldDirection || '').toLowerCase();
  let baseAngle: number | null = null;
  let isFallback = false;

  if (signed != null) {
    baseAngle = signed;
    if ((route.includes('comeback') || route.includes('curl')) && Math.abs(signed) < 90) {
      baseAngle = signed < 0 ? -(180 - Math.abs(signed)) : (180 - Math.abs(signed));
    }
  } else if (full != null) {
    baseAngle = full;
  }

  if (baseAngle == null) return { value: null, isFallback };

  const magnitude = Math.abs(baseAngle);
  if (route.includes('out') || route.includes('dig')) {
    const sign = route.includes('out') ? -1 : 1;
    const fieldMatchedPerpendicular =
      (route.includes('out') && fieldDirection === 'out') ||
      (route.includes('dig') && fieldDirection === 'in');
    if ((ideal != null && ideal >= 75) || fieldMatchedPerpendicular) {
      if (magnitude < 45) {
        isFallback = true;
        return { value: sign * 90, isFallback };
      }
      return { value: sign * Math.min(95, Math.max(75, magnitude)), isFallback };
    }
    return { value: sign * magnitude, isFallback };
  }
  if (route.includes('out') || route.includes('comeback')) return { value: -magnitude, isFallback };
  if (route.includes('dig') || route.includes('slant') || route.includes('in')) return { value: magnitude, isFallback };
  return { value: baseAngle, isFallback };
}

function buildResultSummary(summary: RouteSummary) {
  const route = (summary.routeGuess || 'route').toLowerCase();
  const breakStyle = (summary.breakStyle || 'unknown').toLowerCase();
  const slowdown = summary.brakeSpeedDropPct ?? summary.cutSpeedDropPct;
  const speedConfidence = (summary.speedConfidence || 'unknown').toLowerCase();
  const angleInfo = displayedCutAngle(summary);
  const angleText = angleInfo.value == null
    ? 'an estimated break angle'
    : angleInfo.isFallback
      ? `a fallback route angle near ${formatNumber(angleInfo.value, 1)}deg`
      : `${formatNumber(angleInfo.value, 1)}deg`;
  const slowdownText =
    slowdown == null
      ? 'with limited slowdown detail'
      : `with about ${formatNumber(slowdown, 1)}% speed loss into the break`;
  return `Estimated ${route} with a ${breakStyle} break, ${angleText}, and ${slowdownText}. Speed confidence is ${speedConfidence}.`;
}

function buildMetricsFromSummary(summary: RouteSummary): Metric[] {
  const peakSpeedStatus = summary.speedConfidence === 'high'
    ? softenStatus(statusForPeakSpeed(summary.peakSpeedMph))
    : undefined;
  const angleInfo = displayedCutAngle(summary);
  return [
    {
      label: 'Route Guess',
      value: summary.routeGuess || 'Unknown',
      description: 'The route type the app thinks the player ran.'
    },
    {
      label: 'Break Style',
      value: summary.breakStyle || 'Unknown',
      description: 'Whether the break looked sharp and sudden or more rounded and gradual.'
    },
    {
      label: 'Cut Angle',
      value: formatNumber(angleInfo.value, 1),
      unit: 'deg',
      description: angleInfo.isFallback
        ? 'The raw cut angle did not look trustworthy on this clip, so this is a fallback football route angle for the detected route family. Treat it as an estimate, not the exact measured turn.'
        : 'How much the player turned compared to the direction he was running before the break. Negative means he broke out. Positive means he broke in.'
    },
    {
      label: 'Peak Speed',
      value: formatNumber(summary.peakSpeedMph, 1),
      unit: 'mph',
      status: peakSpeedStatus,
      description: summary.speedConfidence === 'high'
        ? 'The fastest speed the player reached during the route.'
        : 'Estimated top speed from video tracking. Use it more for comparison than as an exact timed number.'
    },
    {
      label: '3-Yard Burst Time',
      value: formatNumber(summary.offLine3YdTimeS, 2),
      unit: 's',
      status: softenStatus(statusFor3YardTime(summary.offLine3YdTimeS)),
      description: 'How long it took the player to get through the first 3 yards after starting the route.'
    },
    {
      label: 'Off-Line Acceleration',
      value: formatNumber(summary.offLine3YdAccelMphPerSec, 1),
      unit: 'mph/s',
      description: 'How quickly the player built speed in the first 3 yards.'
    },
    {
      label: 'Cut Time',
      value: formatNumber(summary.cutTimeS, 2),
      unit: 's',
      status: softenStatus(statusForCutTime(summary.cutTimeS)),
      description: 'How long the player spent changing direction at the break.'
    },
    {
      label: 'Break Slowdown',
      value: formatNumber(summary.brakeSpeedDropPct ?? summary.cutSpeedDropPct ?? null, 1),
      unit: '%',
      description: 'How much of the player’s speed was lost as he slowed into the break compared to his own speed on that rep.'
    },
    {
      label: 'Cut Deceleration',
      value: formatNumber(summary.brakeDecelMphPerSec ?? summary.cutDecelMphPerSec ?? null, 1),
      unit: 'mph/s',
      description: 'How quickly the player slowed down into the break. A bigger negative number means a harder slow-down.'
    },
    {
      label: 'Hip Drop',
      value: formatNumber(summary.hipDropPctBodyHeight, 1),
      unit: '% body ht',
      status: softenStatus(statusForHipDrop(summary.hipDropPctBodyHeight)),
      description: 'How much the player lowered his hips near the break.'
    }
  ];
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

type SetupTool = 'target';

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
  const [clipConfirmed, setClipConfirmed] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingVideoUrl, setPendingVideoUrl] = useState<string | null>(null);
  const [setupTool, setSetupTool] = useState<SetupTool>('target');
  const [setupTargetPoint, setSetupTargetPoint] = useState<{ x: number; y: number } | null>(null);
  const [trimStartTimeS, setTrimStartTimeS] = useState<number | null>(null);
  const [trimEndTimeS, setTrimEndTimeS] = useState<number | null>(null);
  const [setupCurrentTimeS, setSetupCurrentTimeS] = useState(0);
  const [jobProgressPreviewUrl, setJobProgressPreviewUrl] = useState<string | null>(null);
  const [jobProgressFrame, setJobProgressFrame] = useState<number | null>(null);
  const [jobTotalFrames, setJobTotalFrames] = useState<number | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const setupVideoRef = React.useRef<HTMLVideoElement>(null);
  const pendingSegmentUrl = buildVideoSegmentSrc(pendingVideoUrl, trimStartTimeS, trimEndTimeS);

  const resetSetupState = (revokeUrl = true) => {
    if (revokeUrl && pendingVideoUrl) {
      URL.revokeObjectURL(pendingVideoUrl);
    }
    setPendingFile(null);
    setPendingVideoUrl(null);
    setClipConfirmed(false);
    setSetupTool('target');
    setSetupTargetPoint(null);
    setTrimStartTimeS(null);
    setTrimEndTimeS(null);
    setSetupCurrentTimeS(0);
    setJobProgressPreviewUrl(null);
    setJobProgressFrame(null);
    setJobTotalFrames(null);
  };

  useEffect(() => {
    const video = setupVideoRef.current;
    if (!video || !clipConfirmed) return;
    const targetTime = trimStartTimeS ?? 0;

    const syncSelectionFrame = () => {
      video.currentTime = targetTime;
      video.pause();
      setSetupCurrentTimeS(targetTime);
    };

    if (video.readyState >= 1) {
      syncSelectionFrame();
      return;
    }

    video.addEventListener('loadedmetadata', syncSelectionFrame, { once: true });
    return () => {
      video.removeEventListener('loadedmetadata', syncSelectionFrame);
    };
  }, [clipConfirmed, trimStartTimeS]);

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
                <video src={pendingSegmentUrl || pendingVideoUrl} className="w-full rounded-lg" controls muted playsInline />
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
  const selectedAngleInfo = displayedCutAngle(selectedSummary);
  const setupReady = clipConfirmed && !!setupTargetPoint;
  const inSetupMode = !!pendingVideoUrl && !!pendingFile;
  const trimRangeIsValid =
    trimStartTimeS == null || trimEndTimeS == null || trimStartTimeS <= trimEndTimeS;
  const uploadErrorInfo = uploadError ? describeUploadError(uploadError) : null;

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
    if (!trimRangeIsValid) {
      setUploadError('Trim start has to come before trim end.');
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
          startFrame: null,
          endFrame: null,
          cutFrame: null,
        };
        const setupPayload = {
          autoCalibrate: true,
          calibrationPoints: [],
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
            ...(trimStartTimeS !== null ? { 'X-Start-Time': String(trimStartTimeS) } : {}),
            ...(trimEndTimeS !== null ? { 'X-End-Time': String(trimEndTimeS) } : {}),
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
          time: formatRunTimestamp(result.analyzedAt),
          score: result.score,
          image: result.image,
          videoUrl: pendingVideoUrl,
          metrics: buildMetricsFromSummary(result.summary),
          summary: result.summary,
          routeDebugPlotUrl: result.routeDebugPlotUrl,
          breakSnapshotUrl: result.breakSnapshotUrl,
          speedGraphUrl: result.speedGraphUrl,
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

    video.pause();
    setSetupCurrentTimeS(video.currentTime);

    const rect = video.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * video.videoWidth;
    const y = ((event.clientY - rect.top) / rect.height) * video.videoHeight;

    if (setupTool === 'target') {
      setSetupTargetPoint({ x, y });
    }
  };

  const captureTrimTime = () => {
    const video = setupVideoRef.current;
    if (!video) return null;
    return Number.isFinite(video.currentTime) ? Math.max(0, video.currentTime) : null;
  };

  const setTrimStartFromVideo = () => {
    const current = captureTrimTime();
    if (current == null) return;
    setTrimStartTimeS(current);
    setSetupTargetPoint(null);
    if (trimEndTimeS != null && current > trimEndTimeS) {
      setTrimEndTimeS(current);
    }
    if (setupVideoRef.current && clipConfirmed) {
      setupVideoRef.current.currentTime = current;
      setupVideoRef.current.pause();
      setSetupCurrentTimeS(current);
    }
  };

  const setTrimEndFromVideo = () => {
    const current = captureTrimTime();
    if (current == null) return;
    setTrimEndTimeS(current);
    if (trimStartTimeS != null && current < trimStartTimeS) {
      setTrimStartTimeS(current);
    }
  };

  const clampSetupVideoToTrimRange = (video: HTMLVideoElement) => {
    const minTime = trimStartTimeS ?? 0;
    const maxTime = trimEndTimeS;
    if (video.currentTime < minTime) {
      video.currentTime = minTime;
    }
    if (maxTime != null && video.currentTime > maxTime) {
      video.currentTime = maxTime;
      video.pause();
    }
    setSetupCurrentTimeS(video.currentTime);
  };

  const handleExport = () => {
    if (!selectedAnalysis || !user) return;
    const exportMetrics = selectedAnalysis.summary
      ? buildMetricsFromSummary(selectedAnalysis.summary)
      : selectedAnalysis.metrics;
    
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

    const metricsContent = exportMetrics.map(m => `
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
        {uploadError && uploadErrorInfo && (
          <motion.div 
            initial={{ opacity: 0, y: -24 }}
            animate={{ opacity: 1, y: 20 }}
            exit={{ opacity: 0, y: -24 }}
            className="fixed top-0 left-1/2 z-[100] w-[min(92vw,560px)] -translate-x-1/2"
          >
            <div className="rounded-2xl border border-amber-300/30 bg-slate-950/95 px-5 py-4 shadow-2xl backdrop-blur">
              <div className="flex items-start gap-3">
                <div className="mt-0.5 rounded-full bg-amber-500/15 p-2">
                  <AlertCircle className="h-5 w-5 text-amber-400" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-black uppercase tracking-widest text-amber-300">{uploadErrorInfo.title}</p>
                  <p className="mt-1 text-sm font-medium text-white">{uploadErrorInfo.message}</p>
                  <p className="mt-2 text-xs font-medium text-slate-300">{uploadErrorInfo.hint}</p>
                </div>
                <button
                  onClick={() => setUploadError(null)}
                  className="rounded-full p-1 text-slate-400 transition hover:bg-white/5 hover:text-white"
                  aria-label="Dismiss message"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
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
          <p className="text-[10px] text-slate-500 text-center mb-8 font-medium italic -mt-6">
            * Best results when yard markers or clear field lines are visible in the clip.
          </p>

          <div className="mb-8 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3 space-y-3">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Supported Routes</p>
              <p className="text-[11px] text-slate-400 mt-1">RouteIQ currently works with these five route families.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {['Comeback', 'Curl', 'Out', 'Slant', 'Dig'].map((route) => (
                <span
                  key={route}
                  className="rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-[11px] font-bold text-primary"
                >
                  {route}
                </span>
              ))}
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
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-bold truncate text-slate-900 dark:text-white">{analysis.title}</p>
                      {analyses[0]?.id === analysis.id && (
                        <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[9px] font-black uppercase tracking-widest text-primary">
                          Latest Run
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-500 font-bold uppercase mt-0.5">
                      {analysis.playerSnapshot.name} • #{analysis.playerSnapshot.jerseyNumber}
                    </p>
                    <p className="text-[10px] text-slate-400 mt-1">
                      {analysis.time} • {analysis.score}% Score • {labelForMode(analysis.mode)}
                    </p>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Run ID: {analysis.id.slice(-6)}
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
                  {selectedAnalysis.summary?.cutConfidence && (
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">
                      Break Confidence: {selectedAnalysis.summary.cutConfidence}
                    </span>
                  )}
                  <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 dark:text-slate-400">
                    Run: {formatRunTimestamp(selectedAnalysis.createdAt)}
                  </span>
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
                  <button 
                    onClick={handleExport}
                    disabled={!selectedAnalysis}
                    className="flex items-center gap-1.5 text-[10px] font-black bg-primary text-white px-3 py-1 rounded-full shadow-lg shadow-primary/30 hover:bg-primary/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Download className="w-3 h-3" />
                    Export Report
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 no-scrollbar">
          {inSetupMode && (
            <section className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
              <div className="flex flex-col gap-6 lg:grid lg:grid-cols-[1fr_320px]">
                <div>
                  <div className="space-y-1 mb-4">
                    <h3 className="text-lg font-bold">Set Up Analysis</h3>
                    <p className="text-sm text-slate-500">
                      {!clipConfirmed ? 'Preview and confirm the clip first. Manual trim is optional.' : 'Use the paused selection frame to click the receiver you want to analyze.'}
                    </p>
                  </div>
                  <div className="relative inline-block max-w-full cursor-crosshair" onClick={handleSetupVideoClick}>
                    <video
                      ref={setupVideoRef}
                      src={pendingSegmentUrl || pendingVideoUrl}
                      className={`max-h-[70vh] max-w-full rounded-xl bg-black ${clipConfirmed ? 'pointer-events-none' : ''}`}
                      controls={!clipConfirmed}
                      muted
                      playsInline
                      onLoadedMetadata={(event) => {
                        event.currentTarget.currentTime = trimStartTimeS ?? 0;
                        event.currentTarget.pause();
                        setSetupCurrentTimeS(trimStartTimeS ?? 0);
                      }}
                      onPlay={(event) => {
                        clampSetupVideoToTrimRange(event.currentTarget);
                        if (clipConfirmed) {
                          event.currentTarget.pause();
                          setSetupCurrentTimeS(event.currentTarget.currentTime);
                        }
                      }}
                      onSeeking={(event) => clampSetupVideoToTrimRange(event.currentTarget)}
                      onTimeUpdate={(event) => clampSetupVideoToTrimRange(event.currentTarget)}
                    />
                    <div className="pointer-events-none absolute inset-0">
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
                            setupVideoRef.current.currentTime = trimStartTimeS ?? 0;
                          }
                          setSetupCurrentTimeS(trimStartTimeS ?? 0);
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
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 p-4 space-y-3">
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Manual Trim (Optional)</p>
                      <p className="text-xs text-slate-500 mt-1">Scrub the video visually, then set where the clip should start and end. If you set a start, that frame becomes the frozen receiver-selection frame.</p>
                    </div>
                    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-3 py-2 text-sm">
                      Current video time: <span className="font-bold">{formatVideoTime(setupCurrentTimeS)}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={setTrimStartFromVideo}
                        className="rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold"
                      >
                        Set Start Here
                      </button>
                      <button
                        onClick={setTrimEndFromVideo}
                        className="rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold"
                      >
                        Set End Here
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-1">
                        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Start</p>
                        <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-3 py-2 text-sm font-bold">
                          {formatVideoTime(trimStartTimeS)}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">End</p>
                        <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-3 py-2 text-sm font-bold">
                          {formatVideoTime(trimEndTimeS)}
                        </div>
                      </div>
                    </div>
                    {!trimRangeIsValid && (
                      <p className="text-xs font-bold text-amber-500">Start needs to come before end.</p>
                    )}
                    <button
                      onClick={() => {
                        setTrimStartTimeS(null);
                        setTrimEndTimeS(null);
                      }}
                      className="w-full rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold"
                    >
                      Clear Manual Trim
                    </button>
                  </div>

                  <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 p-4 space-y-2">
                    <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Setup Status</p>
                    <p className="text-sm font-medium">Clip: {pendingFile.name}</p>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span>Clip confirmed</span>
                        <span className={`font-bold ${clipConfirmed ? 'text-emerald-500' : 'text-slate-400'}`}>{clipConfirmed ? 'Done' : 'Needed'}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Receiver selected</span>
                        <span className={`font-bold ${setupTargetPoint ? 'text-emerald-500' : 'text-slate-400'}`}>{setupTargetPoint ? 'Done' : 'Needed'}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Selection frame</span>
                        <span className="font-bold text-slate-500">{trimStartTimeS != null ? formatVideoTime(trimStartTimeS) : 'Start of clip'}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Ready to analyze</span>
                        <span className={`font-bold ${setupReady ? 'text-emerald-500' : 'text-amber-500'}`}>{setupReady ? 'Yes' : 'Not yet'}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Manual trim</span>
                        <span className="font-bold text-slate-500">
                          {trimStartTimeS != null || trimEndTimeS != null ? `${formatVideoTime(trimStartTimeS)} to ${formatVideoTime(trimEndTimeS)}` : 'Auto'}
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-slate-500">Field scaling runs automatically in the background. Best results come when yard markers or clear field lines are visible, and speed is still best treated as an estimate when the camera pans a lot.</p>
                  </div>

                  <div className="flex gap-2">
                    <button onClick={() => setSetupTargetPoint(null)} className="flex-1 rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold">
                      Clear Receiver
                    </button>
                  </div>

                  <div className="flex gap-2">
                    <button onClick={() => resetSetupState()} className="flex-1 rounded-xl border border-slate-200 dark:border-slate-800 px-3 py-2 text-sm font-bold">
                      Cancel
                    </button>
                    <button onClick={startAnalysisFromSetup} disabled={!setupReady} className={`flex-1 rounded-xl px-3 py-2 text-sm font-bold text-white ${setupReady ? 'bg-primary' : 'bg-slate-400 cursor-not-allowed'}`}>
                      Start Analysis
                    </button>
                  </div>
                </div>
              </div>
            </section>
          )}

          {!inSetupMode && !selectedAnalysis ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400">
              <Activity className="w-16 h-16 mb-4 opacity-20" />
              <p>Select an analysis to view performance data</p>
            </div>
          ) : !inSetupMode && selectedAnalysis ? (
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

              {/* Route Visuals */}
              <section className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm overflow-hidden">
                <div className="flex justify-between items-center mb-6">
                  <div className="space-y-1">
                    <h3 className="text-lg font-bold">Route Visuals</h3>
                    <p className="text-xs text-slate-500 font-medium">A smooth look at how the player’s speed changed during the route.</p>
                    <p className="text-[11px] text-slate-400">All route visuals and numbers in this report are estimates based on video tracking.</p>
                  </div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Best for review
                  </div>
                </div>
                <div className="mb-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Quick Read</p>
                  <p className="text-sm text-slate-700 dark:text-slate-300 mt-2 leading-relaxed">
                    {buildResultSummary(selectedSummary)}
                  </p>
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden bg-slate-50 dark:bg-slate-950">
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Speed Graph</p>
                    <p className="text-xs text-slate-500 mt-1">A smooth view of how the player sped up and slowed down through the rep.</p>
                  </div>
                  <div className="h-80 bg-slate-950 flex items-center justify-center">
                    {selectedAnalysis.speedGraphUrl ? (
                      <img
                        src={selectedAnalysis.speedGraphUrl}
                        alt="Speed graph"
                        className="w-full h-full object-contain bg-slate-950"
                      />
                    ) : (
                      <div className="px-6 text-center text-sm text-slate-500">
                        The speed graph will show up here after analysis finishes.
                      </div>
                    )}
                  </div>
                </div>
                <div className="mt-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Cut Angle</p>
                  <p className="text-3xl font-bold mt-2">{formatNumber(selectedAngleInfo.value, 1)}<span className="text-sm text-slate-400 ml-1">deg</span></p>
                  <p className="text-xs text-slate-500 mt-2">
                    {selectedAngleInfo.isFallback
                      ? 'The raw cut angle did not look trustworthy on this clip, so this is a fallback football route angle. It is an estimate and not the exact measured turn.'
                      : 'This is the turn angle from the way the player was running before the break.'}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <span className="rounded-full border border-slate-200 dark:border-slate-700 px-2.5 py-1 text-[11px] font-bold text-slate-600 dark:text-slate-300">
                      Negative = Out-breaking
                    </span>
                    <span className="rounded-full border border-slate-200 dark:border-slate-700 px-2.5 py-1 text-[11px] font-bold text-slate-600 dark:text-slate-300">
                      Positive = In-breaking
                    </span>
                  </div>
                </div>
              </section>
            </>
          ) : null}
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

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/40 p-4">
            <p className="text-xs text-slate-500 leading-relaxed">
              All metrics in RouteIQ are estimates based on video tracking, player detection, and field scaling. They are best used for coaching review and comparison, not as exact timed measurements.
            </p>
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
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Route Confidence</p>
                <p className="text-xl font-bold mt-1">{selectedAnalysis.summary.routeConfidence || 'N/A'}</p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Break Confidence</p>
                <p className="text-xl font-bold mt-1">{selectedAnalysis.summary.cutConfidence || 'N/A'}</p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Speed Confidence</p>
                <p className="text-xl font-bold mt-1">{selectedAnalysis.summary.speedConfidence || 'N/A'}</p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Calibration Confidence</p>
                <p className="text-xl font-bold mt-1">{formatPercent(selectedAnalysis.summary.calibrationConfidence ?? null)}</p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Run Time</p>
                <p className="text-sm font-bold mt-1">{formatRunTimestamp(selectedAnalysis.createdAt)}</p>
              </div>
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/50 p-3">
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Run ID</p>
                <p className="text-sm font-bold mt-1">{selectedAnalysis.id.slice(-6)}</p>
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

          {selectedAnalysis?.summary?.speedConfidenceDetail && (
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/40 p-4">
              <p className="text-xs text-slate-500 leading-relaxed">
                {selectedAnalysis.summary.speedConfidenceDetail}
              </p>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4">
            {(selectedAnalysis?.summary ? buildMetricsFromSummary(selectedAnalysis.summary) : selectedAnalysis?.metrics || []).map((metric, idx) => (
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
                      metric.status === 'High' ? 'text-green-500 bg-green-500/10' :
                      metric.status === 'Solid' ? 'text-blue-500 bg-blue-500/10' :
                      metric.status === 'Moderate' ? 'text-yellow-500 bg-yellow-500/10' :
                      'text-slate-500 bg-slate-500/10'
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
                {selectedAnalysis.breakSnapshotUrl && <a className="text-primary hover:underline" href={selectedAnalysis.breakSnapshotUrl} target="_blank" rel="noreferrer">Open break snapshot</a>}
                {selectedAnalysis.speedGraphUrl && <a className="text-primary hover:underline" href={selectedAnalysis.speedGraphUrl} target="_blank" rel="noreferrer">Open speed graph</a>}
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
