import { JsTokEnv, ConstraintConfig } from "llguidance-wasm";
import { LlamaModelOptions, ModelIface, TokenizerInfo } from "node-llama-cpp";

export interface InferenceCapabilities {
  /**
   * Unconditional splice is allowed.
   */
  ff_tokens: boolean;

  /**
   * Conditional (and unconditional) splices are allowed.
   */
  conditional_ff_tokens: boolean;

  /**
   * Backtracking is allowed.
   */
  backtrack: boolean;

  /**
   * More than one branch is allowed.
   */
  fork: boolean;
}

export interface ConstraintSettings {
  console_log_level?: number;
  buffer_log_level?: number;
}

export interface JsTokenizer {
  nVocab(): number;
  eosToken(): number;
  bosToken(): number;
  tokenInfo(): Uint8Array;
  tokenize(text: string): Uint32Array;
}

export function constraintConfig(
  jsTok: JsTokenizer,
  settings?: ConstraintSettings,
  capabilities?: InferenceCapabilities
): ConstraintConfig {
  if (!capabilities) {
    capabilities = {
      ff_tokens: true,
      conditional_ff_tokens: false,
      backtrack: true,
      fork: false,
    };
  }
  if (!settings) {
    settings = {};
  }
  const tokEnv = JsTokEnv.build(jsTok); // this takes many milliseconds
  return ConstraintConfig._from_json(
    tokEnv,
    JSON.stringify(capabilities),
    JSON.stringify(settings)
  );
}

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

const grammar: any = {
  grammars: [
    {
      nodes: [
        { Join: { sequence: [1, 2] } },
        { String: { literal: "So Im here to tell you that 2 + 2 = " } },
        { Join: { sequence: [3, 4] } },
        { Join: { sequence: [5, 6] } },
        { String: { literal: ".\n" } },
        { Join: { sequence: [9, 10] } },
        { Join: { sequence: [7] } },
        { Join: { sequence: [8], capture_name: "test2" } },
        {
          Gen: {
            body_rx: "\\d+",
            stop_rx: "",
            lazy: false,
            stop_capture_name: null,
            temperature: 0.0,
            max_tokens: 8,
          },
        },
        { Join: { sequence: [11] } },
        { String: { literal: " and 3 + 3 = " } },
        { Join: { sequence: [12], capture_name: "test" } },
        {
          Gen: {
            body_rx: "\\d+",
            stop_rx: "",
            lazy: false,
            stop_capture_name: null,
            temperature: 0.0,
            max_tokens: 8,
          },
        },
      ],
      rx_nodes: [],
    },
  ],
  max_tokens: 1000,
};

export function main() {
  const model = new NodeLlamaCppModel(
    {
      modelPath:
        "node_modules/node-llama-cpp/test/.models/Phi-3.1-mini-4k-instruct-Q5_K_M.gguf",
    },
    { console_log_level: 0, buffer_log_level: 2 }
  );
  const c = model.config.new_constraint(JSON.stringify(grammar));
  console.log(c.get_and_clear_logs());
  console.log("Hello, TypeScript!");
}

main();
