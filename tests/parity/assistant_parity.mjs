// ENG-25: Node-side half of the Python/JS assistant parity check.
//
// Loads the inline-JS engine `assistant_script()` embeds in every board
// page (extracted, by the Python caller, to a plain CommonJS `.cjs` file so
// its guarded `module.exports` fires under Node), a knowledge/corpus JSON
// blob (the SAME payload the page would embed, built by
// `nfl_ats.board_assistant.build_knowledge_for_board` plus the `"teams"`
// list `assistant_section()` adds before inlining it), and a JSON array of
// question strings -- then evaluates the JS engine's pure, DOM-free
// `answer(question, knowledge)` on every question and prints a JSON array
// of `{question, topic, text, anchors}` results to stdout.
//
// This file does no comparison itself: `tests/test_assistant_js_parity.py`
// runs the SAME questions through the Python reference engine
// (`nfl_ats.board_assistant.answer`) and diffs the two result sets. Kept a
// thin, dependency-free evaluator on purpose so a Node-availability skip on
// a given machine (see that test's `_node_executable`) is the only way this
// check goes missing, never a flaky harness bug.
//
// Usage: node assistant_parity.mjs <script.cjs> <knowledge.json> <questions.json>

import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

function fail(message) {
  process.stderr.write(message + "\n");
  process.exit(1);
}

const [scriptPath, knowledgePath, questionsPath] = process.argv.slice(2);
if (!scriptPath || !knowledgePath || !questionsPath) {
  fail(
    "usage: node assistant_parity.mjs <script.cjs> <knowledge.json> <questions.json>"
  );
}

const require = createRequire(import.meta.url);

let engine;
try {
  engine = require(scriptPath);
} catch (err) {
  fail(`failed to load engine script ${scriptPath}: ${err && err.stack ? err.stack : err}`);
}
if (!engine || typeof engine.answer !== "function") {
  fail(
    `${scriptPath} did not export a pure "answer" function -- assistant_script() must expose ` +
      "one via a guarded module.exports (see its docstring)"
  );
}

let knowledge;
let questions;
try {
  knowledge = JSON.parse(readFileSync(knowledgePath, "utf-8"));
} catch (err) {
  fail(`failed to read/parse knowledge JSON ${knowledgePath}: ${err}`);
}
try {
  questions = JSON.parse(readFileSync(questionsPath, "utf-8"));
} catch (err) {
  fail(`failed to read/parse questions JSON ${questionsPath}: ${err}`);
}
if (!Array.isArray(questions)) {
  fail(`${questionsPath} must be a JSON array of question strings`);
}

const results = questions.map((question) => {
  const resolved = engine.answer(question, knowledge);
  return {
    question,
    topic: resolved.topic,
    text: resolved.text,
    anchors: resolved.anchors || [],
  };
});

process.stdout.write(JSON.stringify(results));
