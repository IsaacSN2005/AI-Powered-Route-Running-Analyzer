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
  breakFieldDirection?: string | null;
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
  cutSpeedDropMph?: number | null;
  cutSpeedDropPct?: number | null;
  brakeDecelMphPerSec?: number | null;
  brakeSpeedDropPct?: number | null;
  hipDropPctBodyHeight: number | null;
  cutConfidence?: string | null;
  hipDropConfidence?: string | null;
  calibrationConfidence?: number | null;
  speedConfidence?: string | null;
  speedConfidenceDetail?: string | null;
  distanceProfile?: string | null;
  avgBoxHeightPx?: number | null;
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
  breakSnapshotUrl?: string;
  speedGraphUrl?: string;
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
  startTimeS?: number | null;
  endTimeS?: number | null;
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

function displayedCutAngle(summary: RouteSummary) {
  const route = (summary.routeGuess || "").toLowerCase();
  const ideal = summary.idealizedCutAngleDeg;
  const fieldDirection = (summary.breakFieldDirection || "").toLowerCase();
  let numericDisplayedCutAngle: number | null =
    typeof summary.signedTurnAngleDeg === "number"
      ? summary.signedTurnAngleDeg
      : typeof summary.fullTurnAngleDeg === "number"
        ? summary.fullTurnAngleDeg
        : null;
  let isFallback = false;

  if (typeof summary.signedTurnAngleDeg === "number") {
    numericDisplayedCutAngle = summary.signedTurnAngleDeg;
    if ((route.includes("comeback") || route.includes("curl")) && Math.abs(summary.signedTurnAngleDeg) < 90) {
      numericDisplayedCutAngle =
        summary.signedTurnAngleDeg < 0
          ? -(180 - Math.abs(summary.signedTurnAngleDeg))
          : (180 - Math.abs(summary.signedTurnAngleDeg));
    }
  }

  if (numericDisplayedCutAngle === null) {
    return { value: "N/A" as string | number, isFallback };
  }

  const magnitude = Math.abs(numericDisplayedCutAngle);
  if (route.includes("out") || route.includes("dig")) {
    const sign = route.includes("out") ? -1 : 1;
    const fieldMatchedPerpendicular =
      (route.includes("out") && fieldDirection === "out") ||
      (route.includes("dig") && fieldDirection === "in");
    if ((typeof ideal === "number" && ideal >= 75) || fieldMatchedPerpendicular) {
      if (magnitude < 45) {
        isFallback = true;
        return { value: sign * 90, isFallback };
      }
      return { value: sign * Math.min(95, Math.max(75, magnitude)), isFallback };
    }
    return { value: sign * magnitude, isFallback };
  }
  if (route.includes("out") || route.includes("comeback")) return { value: -magnitude, isFallback };
  if (route.includes("dig") || route.includes("slant") || route.includes("in")) return { value: magnitude, isFallback };
  return { value: numericDisplayedCutAngle, isFallback };
}

function buildMetricsFromSummary(summary: RouteSummary) {
  const angleInfo = displayedCutAngle(summary);
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
      label: "Cut Angle",
      value: angleInfo.value,
      unit: "deg",
      description: angleInfo.isFallback
        ? "The raw cut angle did not look trustworthy on this clip, so this is a fallback football route angle for the detected route family. Treat it as an estimate, not the exact measured turn."
        : "How much the player turned compared to the direction he was running before the break. Negative means out-breaking. Positive means in-breaking.",
    },
    {
      label: "Peak Speed",
      value: summary.peakSpeedMph ?? "N/A",
      unit: "mph",
      description:
        summary.speedConfidence === "high"
          ? "Sustained top-end route speed after smoothing and conservative clipping."
          : "Estimated top speed from video tracking. Use it more for comparison than as an exact timed number.",
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
      label: "Break Slowdown",
      value: summary.brakeSpeedDropPct ?? summary.cutSpeedDropPct ?? "N/A",
      unit: "%",
      description: "How much of the player's speed was lost as he slowed into the break, relative to his own speed on that rep.",
    },
    {
      label: "Cut Deceleration",
      value: summary.brakeDecelMphPerSec ?? summary.cutDecelMphPerSec ?? "N/A",
      unit: "mph/s",
      description: "How quickly the player slowed down into the break. More negative means harder braking.",
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
    breakFieldDirection: typeof summary.break_field_direction === "string" ? summary.break_field_direction : null,
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
    cutSpeedDropMph: typeof summary.cut_speed_drop_mph === "number" ? summary.cut_speed_drop_mph : null,
    cutSpeedDropPct: typeof summary.cut_speed_drop_pct === "number" ? summary.cut_speed_drop_pct : null,
    brakeDecelMphPerSec: typeof summary.brake_decel_mph_per_sec === "number" ? summary.brake_decel_mph_per_sec : null,
    brakeSpeedDropPct: typeof summary.brake_speed_drop_pct === "number" ? summary.brake_speed_drop_pct : null,
    hipDropPctBodyHeight: typeof summary.hip_drop_pct_body_height === "number" ? summary.hip_drop_pct_body_height : null,
    cutConfidence: typeof summary.cut_confidence === "string" ? summary.cut_confidence : null,
    hipDropConfidence: typeof summary.hip_drop_confidence === "string" ? summary.hip_drop_confidence : null,
    calibrationConfidence: typeof summary.calibration_confidence === "number" ? summary.calibration_confidence : null,
    speedConfidence: typeof summary.speed_confidence === "string" ? summary.speed_confidence : null,
    speedConfidenceDetail: typeof summary.speed_confidence_detail === "string" ? summary.speed_confidence_detail : null,
    distanceProfile: typeof summary.distance_profile === "string" ? summary.distance_profile : null,
    avgBoxHeightPx: typeof summary.avg_box_height_px === "number" ? summary.avg_box_height_px : null,
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
  if (typeof job.startTimeS === "number" && Number.isFinite(job.startTimeS)) args.push("--start-time", String(job.startTimeS));
  if (typeof job.endTimeS === "number" && Number.isFinite(job.endTimeS)) args.push("--end-time", String(job.endTimeS));

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
      const speedGraphPath = `${stem}_speed_graph.png`;
      const breakSnapshotPath = `${stem}_break_snapshot.jpg`;
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
        breakSnapshotUrl: fs.existsSync(breakSnapshotPath) ? jobFileUrl(breakSnapshotPath) : undefined,
        speedGraphUrl: fs.existsSync(speedGraphPath) ? jobFileUrl(speedGraphPath) : undefined,
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
        const startTimeHeader = req.header("X-Start-Time");
        const endTimeHeader = req.header("X-End-Time");
        const setupHeader = req.header("X-Analysis-Setup");
        job.startFrame = startFrameHeader ? Number(startFrameHeader) : null;
        job.endFrame = endFrameHeader ? Number(endFrameHeader) : null;
        job.cutFrame = cutFrameHeader ? Number(cutFrameHeader) : null;
        job.startTimeS = startTimeHeader ? Number(startTimeHeader) : null;
        job.endTimeS = endTimeHeader ? Number(endTimeHeader) : null;
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
