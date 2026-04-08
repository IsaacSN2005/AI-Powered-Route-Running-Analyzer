export interface User {
  uid: string;
  displayName: string;
  email: string;
  photoURL?: string;
}

export interface Metric {
  label: string;
  value: string | number;
  unit?: string;
  status?: 'Elite' | 'Optimal' | 'Average' | 'Poor';
  description: string;
}

export type AnalysisMode = 'side-view';

export interface RouteSummary {
  routeGuess: string;
  routeConfidence?: string | null;
  routeReason?: string | null;
  breakStyle: string;
  breakSide?: string | null;
  actualPathCutAngleDeg: number | null;
  fullTurnAngleDeg?: number | null;
  signedTurnAngleDeg?: number | null;
  idealizedCutAngleDeg: number | null;
  peakSpeedMph: number | null;
  avgSpeedMph?: number | null;
  offLine3YdTimeS: number | null;
  offLine3YdAccelMphPerSec: number | null;
  cutTimeS: number | null;
  cutDecelMphPerSec: number | null;
  hipDropPctBodyHeight: number | null;
  cutConfidence?: string | null;
  hipDropConfidence?: string | null;
  calibrationConfidence?: number | null;
  calibrationWarning?: string | null;
  calibrationWarningDetail?: string | null;
  repTrimMode?: string | null;
  repStartFrame?: number | null;
  repEndFrame?: number | null;
  cutFrame?: number | null;
}

export interface Player {
  id: string;
  name: string;
  position: string;
  jerseyNumber: string;
  photoURL?: string;
}

export interface Analysis {
  id: string;
  uid: string;
  playerId: string;
  mode: AnalysisMode;
  playerSnapshot: Player; // Store a snapshot of player info at time of analysis
  title: string;
  time: string;
  score: number;
  image: string;
  videoUrl?: string;
  metrics: Metric[];
  summary?: RouteSummary;
  routeDebugPlotUrl?: string;
  cleanCsvUrl?: string;
  posePointsCsvUrl?: string;
  repCleanCsvUrl?: string;
  summaryCsvUrl?: string;
  createdAt: Date;
}

export interface AnalyzeRequest {
  mode: AnalysisMode;
  filename: string;
  mimeType: string;
  sizeBytes: number;
  startFrame?: number | null;
  endFrame?: number | null;
  cutFrame?: number | null;
}

export interface AnalyzeResponse {
  mode: AnalysisMode;
  score: number;
  image: string;
  summary: RouteSummary;
  metrics: Metric[];
  analyzedAt: string;
  routeDebugPlotUrl?: string;
  cleanCsvUrl?: string;
  posePointsCsvUrl?: string;
  repCleanCsvUrl?: string;
  summaryCsvUrl?: string;
}

export interface AnalyzeStartResponse {
  jobId: string;
  status: 'queued' | 'running';
  message: string;
}

export interface AnalysisJobResponse {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  message: string;
  progressPreviewUrl?: string;
  progressFrame?: number | null;
  totalFrames?: number | null;
  result?: AnalyzeResponse;
  error?: string;
}
