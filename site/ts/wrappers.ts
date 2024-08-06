import { JsTokEnv, ConstraintConfig, Constraint } from "llguidance-wasm";
import { api } from "guidance-ts";

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
  tokenizeExact(text: string): Uint32Array;
}

export type TokenId = number;

export interface AdvanceResult {
  stop: boolean;
  backtrack: number;
  tokens: TokenId[];
}

export class Sequence {
  tokens: TokenId[] = [];

  constructor(prompt: Uint32Array, public constraint: Constraint) {
    this.flush();
    const newTokens = this.constraint.process_prompt(prompt);
    this.tokens = Array.from(newTokens);
    this.flush();
  }

  private flush() {
    const logs = this.constraint.get_and_clear_logs();
    console.log(logs);
  }

  getResults(): api.ParserOutput[] {
    const r = JSON.parse(this.constraint.get_and_clear_results());
    this.flush();
    return r;
  }

  advanceParser(sampled: number): AdvanceResult {
    const r = JSON.parse(this.constraint.advance_parser(sampled));
    this.flush();
    return r;
  }

  samplingMask(): Uint32Array | undefined {
    const mask = this.constraint.sampling_mask();
    this.flush();
    return mask;
  }

  get temperature(): number {
    return this.constraint.temperature;
  }
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
