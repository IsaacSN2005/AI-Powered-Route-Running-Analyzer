import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import fs from "fs";
import fsp from "fs/promises";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { spawn } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const jobsRoot = path.join(__dirname, ".analysis_jobs");
const pythonExec = path.join(repoRoot, ".venv", "bin", "python");
const analyzerScript = path.join(repoRoot, "wr_tracker_vision.py");

type RouteSummary = {
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
};

type AnalyzeResult = {
  mode: "side-view";
  score: number;
  image: string;
  summary: RouteSummary;
  metrics: Array<{
    label: string;
    value: string | number;
    unit?: string;
    status?: "Elite" | "Optimal" | "Average" | "Poor";
    description: string;
  }>;
  analyzedAt: string;
  routeDebugPlotUrl?: string;
  cleanCsvUrl?: string;
  posePointsCsvUrl?: string;
  repCleanCsvUrl?: string;
  summaryCsvUrl?: string;
};

type JobStatus = "queued" | "running" | "completed" | "failed";

type AnalysisJob = {
  id: string;
  status: JobStatus;
  message: string;
  workdir: string;
  inputPath: string;
  startFrame?: number | null;
  endFrame?: number | null;
  cutFrame?: number | null;
  setup?: {
    autoCalibrate?: boolean;
    calibrationPoints?: Array<{
      image_x: number;
      image_y: number;
      field_x: number;
      field_y: number;
    }>;
    targetPoint?: {
      x: number;
      y: number;
    } | null;
  } | null;
  stdout: string[];
  stderr: string[];
  progressImagePath?: string;
  progressFrame?: number | null;
  totalFrames?: number | null;
  result?: AnalyzeResult;
  error?: string;
};

const jobs = new Map<string, AnalysisJob>();

function buildMetricsFromSummary(summary: RouteSummary) {
  return [
    {
      label: "Route Guess",
      value: summary.routeGuess,
      description: "Best-fit route family based on fitted stem and break geometry.",
    },
    {
      label: "Break Style",
      value: summary.breakStyle,
      description: "Classifies whether the cut geometry looks sharp or rounded.",
    },
    {
      label: "Cut Confidence",
      value: summary.cutConfidence ?? "N/A",
      description: "How trustworthy the detected break geometry is for route metrics.",
    },
    {
      label: "Actual Cut Angle",
      value: summary.actualPathCutAngleDeg ?? "N/A",
      unit: "deg",
      description: "Measured change of direction from the tracked path the receiver actually ran.",
    },
    {
      label: "Idealized Cut Angle",
      value: summary.idealizedCutAngleDeg ?? "N/A",
      unit: "deg",
      description: "Cleaner stem-to-break angle that approximates the intended route geometry.",
    },
    {
      label: "Full Turn Angle",
      value: summary.fullTurnAngleDeg ?? "N/A",
      unit: "deg",
      description: "Stem-to-break turn on a full 0-180 scale, useful for comeback and curl semantics.",
    },
    {
      label: "Peak Speed",
      value: summary.peakSpeedMph ?? "N/A",
      unit: "mph",
      description: "Sustained top-end route speed after smoothing and conservative clipping.",
    },
    {
      label: "3-Yard Burst Time",
      value: summary.offLine3YdTimeS ?? "N/A",
      unit: "s",
      description: "Time it takes the receiver to cover the first 3 yards off the line.",
    },
    {
      label: "Off-Line Acceleration",
      value: summary.offLine3YdAccelMphPerSec ?? "N/A",
      unit: "mph/s",
      description: "Average acceleration during the first 3 yards of the release.",
    },
    {
      label: "Cut Time",
      value: summary.cutTimeS ?? "N/A",
      unit: "s",
      description: "Duration of the detected change-of-direction window around the break.",
    },
    {
      label: "Cut Deceleration",
      value: summary.cutDecelMphPerSec ?? "N/A",
      unit: "mph/s",
      description: "Average speed loss into the cut window. More negative means harder braking.",
    },
    {
      label: "Hip Drop",
      value: summary.hipDropPctBodyHeight ?? "N/A",
      unit: "% body ht",
      description: "Hip sink near the cut, normalized by body height when pose confidence supports it.",
    },
    {
      label: "Calibration Confidence",
      value: summary.calibrationConfidence ?? "N/A",
      description: "Confidence in the field-plane mapping used to convert pixels into yards and mph.",
    },
  ];
}

function parseSummaryValue(value: string): string | number | boolean | null {
  if (value === "None" || value === "") return null;
  if (value === "True") return true;
  if (value === "False") return false;
  const num = Number(value);
  if (!Number.isNaN(num) && value.trim() !== "") return num;
  return value;
}

async function parseSummaryFile(summaryPath: string): Promise<Record<string, string | number | boolean | null>> {
  const text = await fsp.readFile(summaryPath, "utf-8");
  const out: Record<string, string | number | boolean | null> = {};
  for (const line of text.split(/\r?\n/)) {
    if (!line.includes("=")) continue;
    const [key, ...rest] = line.split("=");
    out[key] = parseSummaryValue(rest.join("="));
  }
  return out;
}

function jobFileUrl(absPath: string) {
  const relative = path.relative(jobsRoot, absPath).split(path.sep).join("/");
  return `/analysis-files/${relative}`;
}

function toRouteSummary(summary: Record<string, string | number | boolean | null>): RouteSummary {
  return {
    routeGuess: String(summary.route_guess ?? "Unknown"),
    routeConfidence: typeof summary.route_confidence === "string" ? summary.route_confidence : null,
    routeReason: typeof summary.route_reason === "string" ? summary.route_reason : null,
    breakStyle: String(summary.break_style ?? "Unknown"),
    breakSide: typeof summary.break_side === "string" ? summary.break_side : null,
    actualPathCutAngleDeg: typeof summary.actual_path_cut_angle_deg === "number" ? summary.actual_path_cut_angle_deg : null,
    fullTurnAngleDeg: typeof summary.full_turn_angle_deg === "number" ? summary.full_turn_angle_deg : null,
    signedTurnAngleDeg: typeof summary.signed_turn_angle_deg === "number" ? summary.signed_turn_angle_deg : null,
    idealizedCutAngleDeg: typeof summary.idealized_cut_angle_deg === "number" ? summary.idealized_cut_angle_deg : null,
    peakSpeedMph: typeof summary.peak_speed_mph === "number" ? summary.peak_speed_mph : null,
    avgSpeedMph: typeof summary.avg_speed_mph === "number" ? summary.avg_speed_mph : null,
    offLine3YdTimeS: typeof summary.off_line_3yd_time_s === "number" ? summary.off_line_3yd_time_s : null,
    offLine3YdAccelMphPerSec: typeof summary.off_line_3yd_accel_mph_per_sec === "number" ? summary.off_line_3yd_accel_mph_per_sec : null,
    cutTimeS: typeof summary.cut_time_s === "number" ? summary.cut_time_s : null,
    cutDecelMphPerSec: typeof summary.cut_decel_mph_per_sec === "number" ? summary.cut_decel_mph_per_sec : null,
    hipDropPctBodyHeight: typeof summary.hip_drop_pct_body_height === "number" ? summary.hip_drop_pct_body_height : null,
    cutConfidence: typeof summary.cut_confidence === "string" ? summary.cut_confidence : null,
    hipDropConfidence: typeof summary.hip_drop_confidence === "string" ? summary.hip_drop_confidence : null,
    calibrationConfidence: typeof summary.calibration_confidence === "number" ? summary.calibration_confidence : null,
    calibrationWarning: typeof summary.confidence_warning === "string" ? summary.confidence_warning : null,
    calibrationWarningDetail: typeof summary.confidence_warning_detail === "string" ? summary.confidence_warning_detail : null,
    repTrimMode: typeof summary.rep_trim_mode === "string" ? summary.rep_trim_mode : null,
    repStartFrame: typeof summary.rep_start_frame === "number" ? summary.rep_start_frame : null,
    repEndFrame: typeof summary.rep_end_frame === "number" ? summary.rep_end_frame : null,
    cutFrame: typeof summary.cut_frame === "number" ? summary.cut_frame : null,
  };
}

function scoreFromSummary(routeSummary: RouteSummary): number {
  let score = 70;
  if (routeSummary.peakSpeedMph !== null) score += Math.min(15, routeSummary.peakSpeedMph * 0.45);
  if (routeSummary.cutTimeS !== null) score += Math.max(0, 12 - routeSummary.cutTimeS * 18);
  if (routeSummary.hipDropPctBodyHeight !== null) score += Math.min(8, routeSummary.hipDropPctBodyHeight * 2);
  return Math.max(50, Math.min(99, Math.round(score)));
}

async function createJobFromUpload(buffer: Buffer, filename: string) {
  const jobId = randomUUID();
  const workdir = path.join(jobsRoot, jobId);
  await fsp.mkdir(workdir, { recursive: true });
  const safeName = filename.replace(/[^\w.\-()[\] ]+/g, "_");
  const inputPath = path.join(workdir, safeName);
  await fsp.writeFile(inputPath, buffer);

  const job: AnalysisJob = {
    id: jobId,
    status: "queued",
    message: "Preparing analysis...",
    workdir,
    inputPath,
    stdout: [],
    stderr: [],
    progressImagePath: path.join(workdir, "progress_latest.jpg"),
    progressFrame: null,
    totalFrames: null,
  };
  jobs.set(jobId, job);
  return job;
}

function startAnalysis(job: AnalysisJob) {
  job.status = "running";
  job.message = "Analyzing clip in-page.";

  const args = [analyzerScript, "--video", job.inputPath, "--mode", "side-view", "--auto-calibrate"];
  if (job.setup?.autoCalibrate === false && (!job.setup.calibrationPoints || job.setup.calibrationPoints.length < 4)) {
    const autoIndex = args.indexOf("--auto-calibrate");
    if (autoIndex >= 0) args.splice(autoIndex, 1);
  }
  if (job.setup?.calibrationPoints && job.setup.calibrationPoints.length >= 4) {
    const autoIndex = args.indexOf("--auto-calibrate");
    if (autoIndex >= 0) args.splice(autoIndex, 1);
    args.push("--calibration-json", JSON.stringify(job.setup.calibrationPoints));
  }
  if (job.setup?.targetPoint) {
    args.push("--target-point", `${job.setup.targetPoint.x},${job.setup.targetPoint.y}`);
  }
  args.push("--headless");
  if (job.progressImagePath) {
    args.push("--progress-image-path", job.progressImagePath);
  }
  if (typeof job.startFrame === "number" && Number.isFinite(job.startFrame)) args.push("--start-frame", String(job.startFrame));
  if (typeof job.endFrame === "number" && Number.isFinite(job.endFrame)) args.push("--end-frame", String(job.endFrame));
  if (typeof job.cutFrame === "number" && Number.isFinite(job.cutFrame)) args.push("--cut-frame", String(job.cutFrame));

  const child = spawn(
    pythonExec,
    args,
    {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
    }
  );

  child.stdout.on("data", (chunk) => {
    const text = chunk.toString();
    job.stdout.push(text);
    const frameMatch = text.match(/frame[: ]+(\d+)\s*\/\s*(\d+)/i);
    if (frameMatch) {
      job.progressFrame = Number(frameMatch[1]);
      job.totalFrames = Number(frameMatch[2]);
    }
    if (text.includes("Step 1: field calibration")) {
      job.message = "Preparing field calibration.";
    } else if (text.includes("Step 2: receiver selection")) {
      job.message = "Preparing receiver selection.";
    } else if (text.includes("Step 3: tracking")) {
      job.message = "Tracking receiver and pose in progress.";
    } else if (text.includes("Saved route summary CSV")) {
      job.message = "Finalizing outputs...";
    }
  });

  child.stderr.on("data", (chunk) => {
    job.stderr.push(chunk.toString());
  });

  child.on("close", async (code) => {
    if (code !== 0) {
      job.status = "failed";
      job.error = job.stderr.join("") || `Analyzer exited with code ${code}`;
      job.message = "Analysis failed.";
      return;
    }

    try {
      const stem = path.join(job.workdir, path.parse(job.inputPath).name);
      const summaryPath = `${stem}_summary.txt`;
      const cleanCsvPath = `${stem}_clean_metrics.csv`;
      const poseCsvPath = `${stem}_pose_points.csv`;
      const repCleanCsvPath = `${stem}_rep_clean_metrics.csv`;
      const summaryCsvPath = `${stem}_route_summary.csv`;
      const routeDebugPlotPath = `${stem}_route_debug.png`;
      const overlayPath = `${stem}_overlay.mp4`;

      const parsed = await parseSummaryFile(summaryPath);
      const summary = toRouteSummary(parsed);
      const result: AnalyzeResult = {
        mode: "side-view",
        score: scoreFromSummary(summary),
        image: fs.existsSync(routeDebugPlotPath) ? jobFileUrl(routeDebugPlotPath) : "",
        summary,
        metrics: buildMetricsFromSummary(summary),
        analyzedAt: new Date().toISOString(),
        routeDebugPlotUrl: fs.existsSync(routeDebugPlotPath) ? jobFileUrl(routeDebugPlotPath) : undefined,
        cleanCsvUrl: fs.existsSync(cleanCsvPath) ? jobFileUrl(cleanCsvPath) : undefined,
        posePointsCsvUrl: fs.existsSync(poseCsvPath) ? jobFileUrl(poseCsvPath) : undefined,
        repCleanCsvUrl: fs.existsSync(repCleanCsvPath) ? jobFileUrl(repCleanCsvPath) : undefined,
        summaryCsvUrl: fs.existsSync(summaryCsvPath) ? jobFileUrl(summaryCsvPath) : undefined,
      };

      if (fs.existsSync(overlayPath)) {
        job.message = "Analysis complete.";
      }
      job.result = result;
      job.status = "completed";
      job.message = "Analysis complete.";
    } catch (error) {
      job.status = "failed";
      job.error = error instanceof Error ? error.message : "Unable to parse analysis outputs.";
      job.message = "Analysis failed while reading outputs.";
    }
  });
}

async function startServer() {
  await fsp.mkdir(jobsRoot, { recursive: true });

  const app = express();
  const PORT = 3000;

  app.use(express.json());
  app.use("/analysis-files", express.static(jobsRoot));

  app.post(
    "/api/analyze",
    express.raw({ type: ["video/*", "application/octet-stream"], limit: "500mb" }),
    async (req, res) => {
      try {
        const body = req.body;
        if (!Buffer.isBuffer(body) || body.length === 0) {
          res.status(400).json({ error: "No video bytes were uploaded." });
          return;
        }

        const rawFilename = typeof req.header("X-Filename") === "string" ? req.header("X-Filename")! : "uploaded-video.mp4";
        const filename = decodeURIComponent(rawFilename);
        const job = await createJobFromUpload(body, filename);
        const startFrameHeader = req.header("X-Start-Frame");
        const endFrameHeader = req.header("X-End-Frame");
        const cutFrameHeader = req.header("X-Cut-Frame");
        const setupHeader = req.header("X-Analysis-Setup");
        job.startFrame = startFrameHeader ? Number(startFrameHeader) : null;
        job.endFrame = endFrameHeader ? Number(endFrameHeader) : null;
        job.cutFrame = cutFrameHeader ? Number(cutFrameHeader) : null;
        job.setup = setupHeader ? JSON.parse(decodeURIComponent(setupHeader)) : null;
        startAnalysis(job);
        res.json({
          jobId: job.id,
          status: job.status,
          message: job.message,
        });
      } catch (error) {
        res.status(500).json({
          error: error instanceof Error ? error.message : "Unable to start analysis.",
        });
      }
    }
  );

  app.get("/api/analyze/:jobId", (req, res) => {
    const job = jobs.get(req.params.jobId);
    if (!job) {
      res.status(404).json({ error: "Analysis job not found." });
      return;
    }
    res.json({
      jobId: job.id,
      status: job.status,
      message: job.message,
      progressPreviewUrl:
        job.progressImagePath && fs.existsSync(job.progressImagePath)
          ? `${jobFileUrl(job.progressImagePath)}?t=${Date.now()}`
          : undefined,
      progressFrame: job.progressFrame,
      totalFrames: job.totalFrames,
      result: job.result,
      error: job.error,
    });
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
