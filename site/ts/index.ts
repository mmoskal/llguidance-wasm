import initWasm from "llguidance-wasm";
import { ConstraintConfig } from "llguidance-wasm";
import {
  grm,
  gen,
  Generation,
  GenerationOptions,
  select,
  capture,
  lexeme,
  keyword,
  GrammarNode,
} from "guidance-ts";
import { constraintConfig, LLConstraint } from "./wrappers.js";

import * as webllm from "window-ai-ll-polyfill";
import {
  append,
  btn,
  elt,
  mkElt,
  rootElt,
  setError,
  setProgress,
  text,
} from "./chtml.js";

export class WaiSequence extends Generation {
  private tokens: number[];
  private ptr = 0;

  ffTokens = 0;
  sampledTokens = 0;
  genTime = 0;
  prefillTime = 0;

  constructor(
    private seq: webllm.AISequence,
    private ll: LLConstraint,
    options: GenerationOptions
  ) {
    super(options);
    this.tokens = Array.from(this.ll.processPrompt());
  }

  override async run() {
    if (this.started) throw new Error("Already started");
    this.started = true;
    const maxTokens = this.options.maxTokens ?? 100;
    for (let i = 0; i < maxTokens; ++i) {
      if (!(await this.advance())) break;
    }
    return;
  }

  destroy(): void {
    this.seq.destroy();
  }

  private async advance() {
    const t0 = performance.now();
    const isPrefill = this.ffTokens == 0;
    this.ffTokens += this.tokens.length - this.ptr;
    this.sampledTokens += 1;
    const adv = this.seq.advance(this.tokens.slice(this.ptr), 0);
    this.ptr = this.tokens.length;
    const mask = this.ll.samplingMask();
    await adv;
    const token = await this.seq.sample({
      temperature: this.ll.temperature,
      samplingMask: mask,
    });
    const r = this.ll.advanceParser(token);
    if (r.backtrack) {
      this.tokens.splice(-r.backtrack, r.backtrack);
      throw new Error("Backtracking not implemented");
    }
    this.tokens.push(...r.tokens);
    for (const res of this.ll.getResults()) {
      this.handleParserOutput(res);
    }
    if (isPrefill) {
      this.prefillTime += performance.now() - t0;
    } else {
      this.genTime += performance.now() - t0;
    }
    return !r.stop;
  }

  getStats() {
    const saved = Math.max(0, this.ffTokens - this.sampledTokens);
    const seconds = this.genTime / 1000;
    const tps = (this.sampledTokens - 1) / seconds;
    const prefill = this.prefillTime / 1000;
    return `Tokens: ${
      this.sampledTokens
    } + ${saved} saved, generation: ${tps.toFixed(
      1
    )} t/s; prefill ${prefill.toFixed(3)} s`;
  }
}

export class WaiModel {
  config: ConstraintConfig;

  constructor(private ai: webllm.AIModel) {}

  private computeTokenInfo() {
    // binary format description for all tokens in the tokenizer
    const nVocab = this.ai.vocabSize;
    const encoder = new TextEncoder();

    const encodedTokenizer = [];
    for (let i = 0; i < nVocab; i++) {
      let bytes = this.ai.tokenBytes(i);
      let isSpecial = false;
      if (bytes.length == 0) {
        isSpecial = true;
        bytes = encoder.encode(this.ai.tokenName(i));
      }
      if (bytes.length > 0xff) {
        throw new Error(
          `Token too long: ${JSON.stringify(this.ai.tokenName(i))} at ${i}`
        );
      }
      encodedTokenizer.push(isSpecial ? 0x40 : 0, bytes.length, ...bytes);
    }
    return new Uint8Array(encodedTokenizer);
  }

  private async init() {
    if (this.config) return;

    await initWasm();

    const ai = this.ai;
    const tokenInfo = this.computeTokenInfo();

    this.config = constraintConfig(
      {
        vocabSize: ai.vocabSize,
        eosToken: ai.eosToken,
        tokenInfo: () => tokenInfo,
        tokenizeExact: (text) => {
          const a = ai.tokenizeExact(text);
          return new Uint32Array(a.buffer, a.byteOffset, a.length);
        },
      },
      { console_log_level: 0, buffer_log_level: 2 },
      {
        ff_tokens: true,
        conditional_ff_tokens: true,
        backtrack: false,
        fork: false,
      }
    );
  }

  async generation(options: GenerationOptions): Promise<WaiSequence> {
    await this.init();
    const c = this.config.new_constraint(
      JSON.stringify(options.grammar.serialize())
    );
    const ll = new LLConstraint(c);
    if (options.prompt) {
      throw new Error("Prompt not implemented");
    }
    const seq = await this.ai.createSequence({ messages: options.messages });
    return new WaiSequence(seq, ll, options);
  }
}

let model: WaiModel;

async function deleteModel() {
  setProgress("Deleting model...");
  const keys = await window.caches.keys();
  for (const key of keys) {
    await window.caches.delete(key);
  }
  setProgress("Model deleted.");
}

async function generate() {
  setError("");
  const user = (elt("msg-user") as HTMLTextAreaElement).value;
  const gr = (elt("grammar") as HTMLTextAreaElement).value;
  const maxTokens = parseInt((elt("max-tokens") as HTMLInputElement).value);

  const args = [gen, select, grm, capture, lexeme, keyword];
  const argNames = ["gen", "select", "grm", "capture", "lexeme", "keyword"];
  let grammar: GrammarNode;
  try {
    grammar = new Function(...argNames, gr)(...args);
  } catch (e) {
    setError(e.message);
    return;
  }

  if (!(grammar instanceof GrammarNode)) {
    setError("Not a GrammarNode; use grm`...` at the top-level!");
    return;
  }

  try {
    grammar.serialize();
  } catch (e) {
    setError(e.message);
    return;
  }

  if (!model) {
    const engine = await webllm.CreateMLCEngine(
      "Phi-3-mini-4k-instruct-q4f16_1-MLC",
      {
        initProgressCallback: (progress) => {
          setProgress(progress.text);
          // console.log(progress);
        },
      }
    );

    model = new WaiModel(webllm.createAIModel(engine));
  }

  const seq = await model.generation({
    grammar: grammar,
    messages: [{ role: "user", content: user }],
    maxTokens,
  });
  elt("output").textContent = "";
  try {
    seq.onText = (t) => {
      if (!t.str) return;
      const e = text(t.str, t.is_generated ? "span.generated" : "span.forced");
      append(elt("output"), e);
    };
    await seq.run();
    elt("stats").textContent = seq.getStats();
  } finally {
    seq.destroy();
  }
}

async function checkWebGPU() {
  const webgpuReport =
    "Please visit <a href='https://webgpureport.org/'>webgpureport.org</a> to check your system's compatibility.";
  if (typeof navigator !== "undefined" && navigator.gpu !== void 0) {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (adapter == null) {
      setError({
        html: "Unable to find a compatible GPU. " + webgpuReport,
      });
      return false;
    }
  } else {
    setError({
      html: "WebGPU not supported. " + webgpuReport,
    });
    return false;
  }
  return true;
}

export async function main() {
  loadExample(examples[0]);
  setError("");
  setProgress("");

  if (!(await checkWebGPU())) return;

  setProgress("Press 'Generate' to download model and generate text");
  for (const ex of examples) {
    const t = ex;
    append(
      elt("examples"),
      btn(ex.name, "", () => {
        loadExample(t);
      })
    );
  }

  elt("generate").addEventListener("click", async (ev) => {
    ev.preventDefault();
    await generate();
  });
  elt("del-model").addEventListener("click", async (ev) => {
    ev.preventDefault();
    await deleteModel();
  });
}

interface Example {
  name: string;
  user: string;
  grammar: string;
}

function loadExample(ex: Example) {
  (elt("msg-user") as HTMLTextAreaElement).value = ex.user;
  (elt("grammar") as HTMLTextAreaElement).value = ex.grammar;
}

const examples = [
  {
    name: "JSON",
    user: "Please give me a JSON object.",
    grammar: `const item = gen("item", { listAppend: true, stop: '"' });
const valid_weapons = ["sword", "bow", "staff"];
return grm\`
{
  "id": "elf",
  "name": "\${gen("name", { stop: '"' })}",
  "age": \${gen("age", /[0-9]+/, { stop: "," })},
  "armor": "\${capture("armor", select("leather", "chainmail", "plate"))}",
  "weapon": "\${capture("weapon", select(...valid_weapons))}",
  "class": "\${gen("class", { stop: '"' })}",
  "mantra": "\${gen("mantra", { stop: '"' })}",
  "strength": \${gen("strength", /[0-9]+/, { stop: "," })},
  "items": ["\${item}", "\${item}", "\${item}"]
}\``,
  },

  {
    name: "Math",
    user: "Let's do some math!",
    grammar:
      "return grm`2 + 2 = ${gen(/[0-9]+/)}! and 3 + 3 = ${gen(/[0-9]+/)}!`",
  },

  {
    name: "Poem",
    user: "Write a poem about shouting",
    grammar: "return grm`${gen(/[a-z \\n]+/, { temperature: 0.8 })}`",
  },
];
