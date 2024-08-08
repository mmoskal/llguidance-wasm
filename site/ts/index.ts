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
} from "./chtml.js";

export class WaiSequence extends Generation {
  private tokens: number[];
  private ptr = 0;

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
    return !r.stop;
  }
}

export class WaiModel {
  config: ConstraintConfig;

  constructor(private ai: webllm.AIModel) {}

  private async init() {
    if (this.config) return;

    await initWasm();

    const ai = this.ai;
    const tokenInfo = await ai.tokenInfo();

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
          console.log(progress);
        },
      }
    );

    model = new WaiModel(new webllm.AIModel(engine));
  }

  const seq = await model.generation({
    grammar: grammar,
    messages: [{ role: "user", content: user }],
    maxTokens,
  });
  try {
    seq.onText = (t) => {
      elt("output").textContent = seq.getText();
    };
    await seq.run();
  } finally {
    seq.destroy();
  }
}

export async function main() {
  loadExample(examples[0]);
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
    name: "Math",
    user: "Let's do some math!",
    grammar: 'return grm`2 + 2 = ${gen(/[0-9]+/)}! and 3 + 3 = ${gen(/[0-9]+/)}!`',
  },
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
];
