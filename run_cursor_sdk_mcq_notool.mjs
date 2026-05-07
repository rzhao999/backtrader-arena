#!/usr/bin/env node
// Run Backtrader MCQs through the Cursor SDK while failing on any tool call.
//
// Cursor does not document a true SDK/API "disable all tools" switch. This
// runner uses a strict local setup: no ambient settings, no inline MCP servers
// or subagents, sandbox enabled, and stream inspection that cancels the run if
// a tool call appears.

import { Agent } from "@cursor/sdk";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const ANSWER_RE = /<<<\s*([A-D])\s*>>>/i;
const DEFAULT_MODEL = "composer-2";
const DEFAULT_INPUT = "Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl";
const DEFAULT_OUTPUT_DIR = "cursor_sdk_notools_runs";
const DIFFICULTY_ORDER = new Map([
  ["easy", 0],
  ["medium", 1],
  ["hard", 2],
  ["unknown", 3],
]);

function usage() {
  console.error(`Usage:
  node run_cursor_sdk_mcq_notool.mjs [options]

Options:
  --input <path>       JSON or JSONL MCQ file (default: ${DEFAULT_INPUT})
  --output-dir <dir>   Output directory (default: ${DEFAULT_OUTPUT_DIR})
  --model <id>         Cursor model id (default: ${DEFAULT_MODEL})
  --timeout <seconds>  Per-question timeout (default: 120)

Requires:
  npm install @cursor/sdk
  export CURSOR_API_KEY="cursor_..."
`);
}

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    outputDir: DEFAULT_OUTPUT_DIR,
    model: DEFAULT_MODEL,
    timeoutMs: 120_000,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const readValue = () => {
      if (i + 1 >= argv.length || argv[i + 1].startsWith("--")) {
        throw new Error(`${arg} requires a value`);
      }
      i += 1;
      return argv[i];
    };

    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--input") {
      args.input = readValue();
    } else if (arg.startsWith("--input=")) {
      args.input = arg.slice("--input=".length);
    } else if (arg === "--output-dir") {
      args.outputDir = readValue();
    } else if (arg.startsWith("--output-dir=")) {
      args.outputDir = arg.slice("--output-dir=".length);
    } else if (arg === "--model") {
      args.model = readValue();
    } else if (arg.startsWith("--model=")) {
      args.model = arg.slice("--model=".length);
    } else if (arg === "--timeout") {
      args.timeoutMs = Number(readValue()) * 1000;
    } else if (arg.startsWith("--timeout=")) {
      args.timeoutMs = Number(arg.slice("--timeout=".length)) * 1000;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) {
    throw new Error("--timeout must be a positive number of seconds");
  }
  return args;
}

function safePathSegment(value) {
  const segment = String(value || "unknown")
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return segment || "unknown";
}

async function loadQuestions(inputPath) {
  const raw = await fs.readFile(inputPath, "utf8");
  const trimmed = raw.trim();
  if (!trimmed) return [];

  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return parsed.filter((item) => item && typeof item === "object");
    if (parsed && typeof parsed === "object") return [parsed];
  } catch {
    // Fall through to JSONL.
  }

  return trimmed
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`Invalid JSONL on line ${index + 1}: ${error.message}`);
      }
    })
    .filter((item) => item && typeof item === "object");
}

function buildPrompt(question) {
  return `You are a no-tools multiple-choice answerer. This is a one-shot Cursor SDK local agent run.

Do not use tools. Do not call web search, web fetch, Bash, Python, MCPs, file search, file read, file write, browser, workspace inspection, or any other external capability.

Use only the question text and your internal reasoning. Reply with exactly <<< X >>> where X is one of A, B, C, or D. No explanation, no code, no extra text.

${question}`;
}

function parseAnswer(text) {
  const match = ANSWER_RE.exec(text || "");
  return match ? match[1].toUpperCase() : null;
}

function normalizeDifficulty(questionObj) {
  const difficulty = questionObj.difficulty;
  return typeof difficulty === "string" && difficulty.trim() ? difficulty.trim() : "unknown";
}

function estimateTokens(text) {
  const value = String(text || "");
  if (!value) return 0;
  const chunks = value.match(/[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) || [];
  return Math.max(1, Math.ceil(chunks.length * 1.25));
}

function estimatedUsage(prompt, response, model) {
  const inputTokens = estimateTokens(prompt);
  const outputTokens = estimateTokens(response);
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    cache_read_tokens: null,
    cache_creation_tokens: null,
    model,
    source: "estimated",
  };
}

function pickNumber(obj, keys) {
  for (const key of keys) {
    const value = obj?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function normalizeUsage(payload, model) {
  if (!payload || typeof payload !== "object") return null;
  const usage = payload.usage && typeof payload.usage === "object" ? payload.usage : payload;
  const inputTokens = pickNumber(usage, [
    "input_tokens",
    "inputTokens",
    "prompt_tokens",
    "promptTokens",
    "promptTokenCount",
  ]);
  let outputTokens = pickNumber(usage, [
    "output_tokens",
    "outputTokens",
    "completion_tokens",
    "completionTokens",
    "candidatesTokenCount",
  ]);
  const totalTokens = pickNumber(usage, ["total_tokens", "totalTokens", "totalTokenCount"]);

  if (outputTokens === null && inputTokens !== null && totalTokens !== null) {
    outputTokens = totalTokens - inputTokens;
  }
  if (inputTokens === null && outputTokens === null && totalTokens === null) return null;

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
    cache_read_tokens: pickNumber(usage, [
      "cache_read_tokens",
      "cacheReadTokens",
      "cache_read_input_tokens",
      "cacheReadInputTokens",
      "cached_input_tokens",
      "cachedInputTokens",
    ]),
    cache_creation_tokens: pickNumber(usage, [
      "cache_creation_tokens",
      "cacheCreationTokens",
      "cache_write_input_tokens",
      "cacheWriteInputTokens",
    ]),
    model: typeof payload.model === "string" ? payload.model : model,
    source: "exact",
  };
}

function extractUsage(payload, model, seen = new WeakSet()) {
  if (!payload || typeof payload !== "object") return null;
  if (seen.has(payload)) return null;
  seen.add(payload);

  const direct = normalizeUsage(payload, model);
  if (direct) return direct;

  for (const value of Object.values(payload)) {
    if (value && typeof value === "object") {
      const nested = extractUsage(value, model, seen);
      if (nested) return nested;
    }
  }
  return null;
}

function eventLooksLikeToolUse(event) {
  if (!event || typeof event !== "object") return false;
  const type = String(event.type || event.kind || "");
  if (type.includes("tool")) return true;
  const serialized = JSON.stringify(event);
  return /"tool_call"|"toolCall"|"tool_use"|"toolUse"/.test(serialized);
}

function collectAssistantText(event) {
  if (event?.type !== "assistant") return "";
  let text = "";
  for (const block of event.message?.content || []) {
    if (block?.type === "text" && typeof block.text === "string") {
      text += block.text;
    }
  }
  return text;
}

function eventSummary(event) {
  const summary = {
    type: event?.type ?? null,
    kind: event?.kind ?? null,
    keys: event && typeof event === "object" ? Object.keys(event).sort() : [],
    looks_like_tool_use: eventLooksLikeToolUse(event),
  };
  if (event?.message?.content) {
    summary.message_content_types = event.message.content.map((block) => block?.type ?? null);
  }
  return summary;
}

async function waitWithTimeout(promise, timeoutMs, onTimeout) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(async () => {
          await onTimeout?.();
          reject(new Error(`Timed out after ${timeoutMs / 1000}s`));
        }, timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function runOne({ agent, questionObj, index, total, timeoutMs, outputRoot, model }) {
  const question = String(questionObj.question || "").trim();
  const expected = questionObj.answer ? String(questionObj.answer).toUpperCase() : null;
  const difficulty = normalizeDifficulty(questionObj);
  const qDir = path.join(outputRoot, `q_${String(index).padStart(3, "0")}`);
  await fs.mkdir(qDir, { recursive: true });

  const prompt = buildPrompt(question);
  await fs.writeFile(path.join(qDir, "prompt.txt"), prompt + "\n", "utf8");
  await fs.writeFile(path.join(qDir, "question_meta.json"), JSON.stringify(questionObj, null, 2) + "\n", "utf8");

  const started = Date.now();
  let run;
  let stdout = "";
  let stderr = "";
  let status = "ok";
  let toolAttempted = false;
  let usage = null;
  const eventSummaries = [];
  const toolEvents = [];

  try {
    run = await agent.send(prompt, { mcpServers: {} });
    const stream = run.stream();

    await waitWithTimeout(
      (async () => {
        for await (const event of stream) {
          const summary = eventSummary(event);
          eventSummaries.push(summary);
          usage = extractUsage(event, model) || usage;
          if (eventLooksLikeToolUse(event)) {
            toolAttempted = true;
            toolEvents.push(summary);
            status = "tool_attempted";
            stderr += `Tool use attempted and blocked: ${JSON.stringify(event)}\n`;
            if (run.supports?.("cancel")) await run.cancel();
            break;
          }
          stdout += collectAssistantText(event);
        }
        const result = await run.wait();
        usage = extractUsage(result, result?.model?.id || model) || usage;
        if (result?.status && result.status !== "finished" && status === "ok") {
          status = result.status;
        }
      })(),
      timeoutMs,
      async () => {
        if (run?.supports?.("cancel")) await run.cancel();
      },
    );
  } catch (error) {
    if (!toolAttempted) {
      status = "error";
      stderr += `${error.name || "Error"}: ${error.message || String(error)}\n`;
    }
  }

  const parsed = parseAnswer(stdout);
  const elapsedSec = Math.round((Date.now() - started) / 100) / 10;
  const recordUsage = usage || estimatedUsage(prompt, stdout, model);
  const record = {
    index,
    model: recordUsage.model ?? model,
    status,
    difficulty,
    elapsed_sec: elapsedSec,
    parsed_answer: parsed,
    expected_answer: expected,
    is_correct: expected ? parsed === expected : null,
    question_dir: qDir,
    usage: recordUsage,
    event_count: eventSummaries.length,
    event_types: [...new Set(eventSummaries.map((event) => event.type).filter(Boolean))],
    tool_event_count: toolEvents.length,
  };

  await fs.writeFile(path.join(qDir, "response.txt"), stdout, "utf8");
  await fs.writeFile(path.join(qDir, "stderr.txt"), stderr, "utf8");
  await fs.writeFile(
    path.join(qDir, "event_summaries.jsonl"),
    eventSummaries.map((event) => JSON.stringify(event)).join("\n") + "\n",
    "utf8",
  );
  await fs.writeFile(path.join(qDir, "tool_events.json"), JSON.stringify(toolEvents, null, 2) + "\n", "utf8");
  console.log(
    `[CursorSDKNoTools] [${index}/${total}] model=${record.model} status=${status} difficulty=${difficulty} parsed=${parsed} expected=${expected} correct=${record.is_correct} tool_events=${toolEvents.length} elapsed=${elapsedSec}s`,
  );
  return record;
}

function aggregateAccuracyByDifficulty(records) {
  const grouped = new Map();
  for (const record of records) {
    if (!record.expected_answer) continue;
    const difficulty = record.difficulty || "unknown";
    if (!grouped.has(difficulty)) grouped.set(difficulty, []);
    grouped.get(difficulty).push(record);
  }

  const entries = [...grouped.entries()].sort(([a], [b]) => {
    const ao = DIFFICULTY_ORDER.get(a.toLowerCase()) ?? DIFFICULTY_ORDER.size;
    const bo = DIFFICULTY_ORDER.get(b.toLowerCase()) ?? DIFFICULTY_ORDER.size;
    return ao - bo || a.localeCompare(b);
  });

  return Object.fromEntries(
    entries.map(([difficulty, items]) => {
      const answered = items.filter((record) => record.parsed_answer !== null);
      const correct = items.filter((record) => record.is_correct);
      return [
        difficulty,
        {
          gradable: items.length,
          answered: answered.length,
          correct: correct.length,
          accuracy_pct: items.length ? Math.round((correct.length / items.length) * 10000) / 100 : 0,
          answered_pct: items.length ? Math.round((answered.length / items.length) * 10000) / 100 : 0,
        },
      ];
    }),
  );
}

function aggregateUsage(records) {
  const usages = records.map((record) => record.usage).filter((usage) => usage && typeof usage === "object");
  const numericUsages = usages.filter((usage) =>
    ["input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_creation_tokens"].some(
      (key) => typeof usage[key] === "number",
    ),
  );
  if (numericUsages.length === 0) return null;

  const totals = {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
  };
  const counts = Object.fromEntries(Object.keys(totals).map((key) => [key, 0]));
  const models = new Set();

  for (const usage of numericUsages) {
    if (typeof usage.model === "string") models.add(usage.model);
    for (const key of Object.keys(totals)) {
      const value = usage[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        totals[key] += value;
        counts[key] += 1;
      }
    }
  }

  return {
    n_with_usage: numericUsages.length,
    models: [...models].sort(),
    totals,
    averages: Object.fromEntries(
      Object.keys(totals).map((key) => [`avg_${key}`, counts[key] ? Math.round((totals[key] / counts[key]) * 100) / 100 : null]),
    ),
  };
}

async function main() {
  const args = parseArgs(process.argv);
  if (!process.env.CURSOR_API_KEY) {
    throw new Error("Missing CURSOR_API_KEY");
  }

  const inputPath = path.resolve(args.input);
  const questions = await loadQuestions(inputPath);
  if (questions.length === 0) throw new Error(`No questions found in ${inputPath}`);

  const timestamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "").replace("T", "_");
  const runId = `${timestamp}_${safePathSegment(args.model)}`;
  const outputRoot = path.resolve(args.outputDir, runId);
  await fs.mkdir(outputRoot, { recursive: true });

  console.log(`[CursorSDKNoTools] Input: ${inputPath}`);
  console.log(`[CursorSDKNoTools] Questions: ${questions.length}`);
  console.log(`[CursorSDKNoTools] Output: ${outputRoot}`);
  console.log(`[CursorSDKNoTools] Model: ${args.model}`);

  const agent = await Agent.create({
    apiKey: process.env.CURSOR_API_KEY,
    model: { id: args.model },
    local: {
      cwd: process.cwd(),
      settingSources: [],
      sandboxOptions: { enabled: true },
    },
    mcpServers: {},
    agents: {},
  });

  const records = [];
  try {
    for (let i = 0; i < questions.length; i += 1) {
      const record = await runOne({
        agent,
        questionObj: questions[i],
        index: i + 1,
        total: questions.length,
        timeoutMs: args.timeoutMs,
        outputRoot,
        model: args.model,
      });
      records.push(record);
      await fs.appendFile(path.join(outputRoot, "summary.jsonl"), JSON.stringify(record) + "\n", "utf8");
    }
  } finally {
    await agent[Symbol.asyncDispose]?.();
  }

  const gradable = records.filter((record) => record.expected_answer);
  const correct = gradable.filter((record) => record.is_correct);
  const toolAttempts = records.filter((record) => record.status === "tool_attempted");
  const usageTotals = aggregateUsage(records);
  const modelsUsed = [...new Set(records.map((record) => record.model).filter(Boolean))].sort();
  const report = {
    agent: "CursorSDKNoTools",
    model: args.model,
    models_used: modelsUsed,
    input_file: inputPath,
    total_questions: records.length,
    gradable: gradable.length,
    correct: correct.length,
    accuracy_pct: gradable.length ? Math.round((correct.length / gradable.length) * 10000) / 100 : 0,
    accuracy_by_difficulty: aggregateAccuracyByDifficulty(records),
    tool_attempts: toolAttempts.length,
    per_question: records,
  };
  if (usageTotals) report.usage_totals = usageTotals;
  await fs.writeFile(path.join(outputRoot, "report.json"), JSON.stringify(report, null, 2) + "\n", "utf8");

  for (const [difficulty, stats] of Object.entries(report.accuracy_by_difficulty)) {
    console.log(
      `[CursorSDKNoTools] Accuracy (${difficulty}): ${stats.correct}/${stats.gradable} (${stats.accuracy_pct.toFixed(2)}%). Answered: ${stats.answered}/${stats.gradable} (${stats.answered_pct.toFixed(2)}%).`,
    );
  }
  console.log(
    `[CursorSDKNoTools] Model: ${args.model}. Models used: ${modelsUsed.length ? modelsUsed.join(", ") : "unknown"}. Accuracy: ${correct.length}/${gradable.length} (${report.accuracy_pct.toFixed(2)}%). Tool attempts: ${toolAttempts.length}.`,
  );
  if (usageTotals) {
    console.log(
      `[CursorSDKNoTools] Tokens: in=${usageTotals.totals.input_tokens} out=${usageTotals.totals.output_tokens} total=${usageTotals.totals.total_tokens}.`,
    );
  }
  if (toolAttempts.length > 0) process.exitCode = 1;
}

main().catch((error) => {
  console.error(`${error.name || "Error"}: ${error.message || String(error)}`);
  process.exitCode = 1;
});
