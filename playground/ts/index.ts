import { ConstraintConfig } from "llguidance-wasm";
import { LlamaModelOptions, ModelIface, TokenizerInfo } from "node-llama-cpp";
import { grm, gen, GrammarNode } from "guidance-ts";
import {
  constraintConfig,
  ConstraintSettings,
  Sequence,
  JsTokenizer,
} from "./wrappers.js";

export class NodeLlamaCppModel implements JsTokenizer {
  modelIface: ModelIface;
  tokInfo: TokenizerInfo;
  tokInfoBytes: Uint8Array;
  config: ConstraintConfig;

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
}

export async function main() {
  const model = new NodeLlamaCppModel(
    {
      modelPath:
        "node_modules/node-llama-cpp/test/.models/Phi-3.1-mini-4k-instruct-Q5_K_M.gguf",
    },
    { console_log_level: 0, buffer_log_level: 2 }
  );
  const seq = model.seq(
    "",
    grm`2 + 2 = ${gen(/\d+/)} and 3 + 3 = ${gen(/\d+/)}\n`
  );
  while (await model.nextToken(seq)) {
    console.log(seq.getResults());
  }

  console.log("Hello, TypeScript!");
}

main();
