import initWasm from "llguidance-wasm";
import { ConstraintConfig } from "llguidance-wasm";
import { grm, gen, Generation, GenerationOptions } from "guidance-ts";
import { constraintConfig, LLConstraint } from "./wrappers.js";

import * as webllm from "window-ai-ll-polyfill";
import { append, mkElt, rootElt, setProgress } from "./chtml.js";

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

export async function main() {
  const g = grm`2 + 2 = ${gen(/[0-9]+/)} `;
  const grmJson = g.serialize();
  console.log(grmJson);

  setProgress("Loading model...");

  const engine = await webllm.CreateMLCEngine(
    "Phi-3-mini-4k-instruct-q4f16_1-MLC",
    {
      initProgressCallback: (progress) => {
        setProgress(progress.text);
        console.log(progress);
      },
    }
  );

  const messages: webllm.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a helpful AI assistant." },
    { role: "user", content: "Hello!" },
  ];

  const reply = await engine.chat.completions.create({
    messages,
  });

  append(rootElt(), mkElt("div response", reply.choices[0].message.content));

  console.log(reply.choices[0].message);
  console.log(reply.usage);

  const model = new WaiModel(new webllm.AIModel(engine));
  const seq = await model.generation({
    grammar: grm`2 + 2 = ${gen(/[0-9]+/)}! and 3 + 3 = ${gen(/[0-9]+/)}!`,
    messages: [{ role: "user", content: "Let's do some math!" }],
  });
  seq.onText = (t) => {
    console.log(t.str);
  };
  await seq.run();
}
