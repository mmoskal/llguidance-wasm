import { ConstraintConfig } from "llguidance-wasm";
import { LlamaModelOptions, ModelIface, TokenizerInfo } from "node-llama-cpp";
import { grm, gen, GrammarNode } from "guidance-ts";
import {
  constraintConfig,
  ConstraintSettings,
  Sequence,
  JsTokenizer,
} from "./wrappers.js";

const TOKENIZER_PREFIX = "\x02";

export class NodeLlamaCppModel implements JsTokenizer {
  modelIface: ModelIface;
  tokInfo: TokenizerInfo;
  tokInfoBytes: Uint8Array;
  config: ConstraintConfig;
  tokenizerPrefixTokens: number[];

  constructor(options: LlamaModelOptions, settings?: ConstraintSettings) {
    this.modelIface = new ModelIface(options);
    this.tokInfo = this.modelIface.computeTokenizerInfo();
    const res: number[] = [];
    for (let i = 0; i < this.tokInfo.nVocab; i++) {
      const flags = this.tokInfo.isSpecialToken[i] ? 0x40 : 0x00;
      const bytes = this.tokInfo.tokenBytes[i];
      if (bytes.length > 0xff) {
        throw new Error("Token byte length is too long");
      }
      res.push(flags, bytes.length, ...bytes);
    }
    this.tokInfoBytes = new Uint8Array(res);
    this.config = constraintConfig(this, settings);
    this.tokenizerPrefixTokens = Array.from(this.tokenize(TOKENIZER_PREFIX));
  }

  seq(prompt: string, grammar: GrammarNode): Sequence {
    const c = this.config.new_constraint(JSON.stringify(grammar.serialize()));
    return new Sequence(this.tokenize(prompt), c);
  }

  async nextToken(seq: Sequence): Promise<boolean> {
    const tokenMask = seq.samplingMask();
    if (tokenMask === undefined) {
      return false;
    }
    const temperature = seq.temperature;
    const tok = await this.modelIface.nextToken(new Uint32Array(seq.tokens), {
      temperature,
      tokenMask,
    });
    const res = seq.advanceParser(tok);
    if (res.stop) {
      return false;
    }
    if (res.backtrack) {
      seq.tokens.splice(-res.backtrack, res.backtrack);
    }
    seq.tokens.push(...res.tokens);
    return true;
  }

  nVocab(): number {
    return this.tokInfo.nVocab;
  }
  eosToken(): number {
    return this.tokInfo.eosToken ?? -1;
  }
  bosToken(): number {
    return this.tokInfo.bosToken ?? -1;
  }
  tokenInfo(): Uint8Array {
    return this.tokInfoBytes;
  }
  tokenize(text: string): Uint32Array {
    return this.modelIface.tokenize(text);
  }
  tokenizeExact(text: string): Uint32Array {
    const r = this.tokenize(TOKENIZER_PREFIX + text);
    for (let i = 0; i < this.tokenizerPrefixTokens.length; i++) {
      if (r[i] !== this.tokenizerPrefixTokens[i]) {
        throw new Error("Tokenization mismatch");
      }
    }
    return r.slice(this.tokenizerPrefixTokens.length);
  }
}

export async function main() {
  testLogProbs();

  const model = new NodeLlamaCppModel(
    {
      modelPath:
        "node_modules/node-llama-cpp/test/.models/Phi-3.1-mini-4k-instruct-Q5_K_M.gguf",
    },
    { console_log_level: 0, buffer_log_level: 2 }
  );
  const seq = model.seq(
    "",
    grm`Q: 2 + 2 =\nA: ${gen(/\d+/)}\nQ: 3 + 3 =\nA: ${gen(/\d+/)}\n`
  );
  let maxTokens = 15;
  while (await model.nextToken(seq)) {
    console.log(seq.getResults());
    if (maxTokens-- <= 0) {
      break;
    }
  }

  console.log("Hello, TypeScript!");
}

function testLogProbs() {
  const nVocab = 256_000;
  const l = new Float32Array(nVocab);
  for (let i = 0; i < nVocab; i++) {
    l[i] = Math.random() * 30 - 10;
  }
  const mask = new Uint32Array(nVocab >>> 5);
  mask.fill(0xffff_ffff);

  let r = 0;
  const times: number[] = [];

  for (let iter = 0; iter < 30; ++iter) {
    const t0 = Date.now();
    for (let i = 0; i < 30; ++i) {
      const pos = Math.floor(Math.random() * nVocab);
      setAllowed(mask, pos, false);
    }
    const l2 = logProbs(l, mask);

    const pos = Math.floor(Math.random() * nVocab);
    r += l2[pos];
    times.push(Date.now() - t0);
  }

  console.log({ r, times });
}

function isAllowed(mask: Uint32Array, i: number): boolean {
  return mask[i >>> 5] & (1 << (i & 31)) ? true : false;
}

function setAllowed(mask: Uint32Array, i: number, allowed: boolean): void {
  if (allowed) {
    mask[i >>> 5] |= 1 << (i & 31);
  } else {
    mask[i >>> 5] &= ~(1 << (i & 31));
  }
}

function logProbs(l: Float32Array, mask: Uint32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < l.length; i++) {
    if (mask && !isAllowed(mask, i)) {
      l[i] = -Infinity;
    }
    if (l[i] > max) {
      max = l[i];
    }
  }
  let sum = 0;
  for (let i = 0; i < l.length; i++) {
    sum += Math.exp(l[i] - max);
  }
  const logSum = max + Math.log(sum);
  for (let i = 0; i < l.length; i++) {
    l[i] -= logSum;
  }
  return l;
}

main();
