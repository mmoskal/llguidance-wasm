export type AIAssistantPromptRole = "system" | "user" | "assistant";

export interface AIAssistantPrompt {
  role: AIAssistantPromptRole;
  content: string;
}

export interface AITokenizationOptions {
  allowSpecial?: boolean;
}

export class AIModel {
  async createSequence(): Promise<AISequence> {
    return new AISequence();
  }

  get maxTokens(): number {
    return 0;
  }

  get vocabSize(): number {
    return 0;
  }

  get eosToken(): number {
    return 0;
  }

  get bosToken(): number | undefined {
    return undefined;
  }

  // binary format description for all tokens in the tokenizer
  async tokenInfo(): Promise<Uint8Array> {
    return new Uint8Array();
  }

  // if the last prompt has role:"assistant", it will be kept open, ie.
  // <|end|> or <end_of_turn> will not be added
  async tokenizePrompts(prompts: AIAssistantPrompt[]): Promise<Uint32Array> {
    return new Uint32Array();
  }

  // only encode the exact text
  tokenizeExact(text: string, options?: AITokenizationOptions): Uint32Array {
    return new Uint32Array();
  }
}

export class AISequence {
  private _tokens: number[];

  get maxTokens(): number {
    return 4096;
  }

  get tokensSoFar(): number {
    return this._tokens.length;
  }

  get tokensLeft(): number {
    return this.maxTokens - this.tokensSoFar;
  }

  get tokens(): number[] {
    return this._tokens.slice();
  }

  async advance2(newTokens: number[]): Promise<AILogits> {
    this._tokens = newTokens;
    return new AILogits();
  }

  async advance(appendTokens: number[], backtrack = 0): Promise<AILogits> {
    if (backtrack > 0) {
      this._tokens.splice(-backtrack, backtrack);
    }
    this._tokens.push(...appendTokens);
    return new AILogits();
  }

  async clone(): Promise<AISequence> {
    return new AISequence();
  }

  // release KV cache
  destroy(): void {}
}

export interface AISamplingOptions {
  temperature?: number;
  samplingMask?: Uint32Array;
}

function isAllowed(mask: Uint32Array, i: number): boolean {
  return mask[i >>> 5] & (1 << (i & 31)) ? true : false;
}

export class AILogits {
  async sample(options?: AISamplingOptions): Promise<number> {
    return 0;
  }

  async logProbs(mask?: Uint32Array): Promise<Float32Array> {
    const l = await this.logits();
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

  // 1 not surprised, 100 or more - very surprised
  async surprise(mask?: Uint32Array): Promise<number> {
    const l = await this.logits();
    let max = -Infinity;
    for (let i = 0; i < l.length; i++) {
      if (l[i] > max) {
        max = l[i];
      }
    }
    let sum = 0;
    let sumMasked = 0;
    for (let i = 0; i < l.length; i++) {
      sum += Math.exp(l[i] - max);
      if (mask && !isAllowed(mask, i)) {
        sumMasked += Math.exp(l[i] - max);
      }
    }
    return sum / sumMasked;
  }

  async logits(): Promise<Float32Array> {
    return new Float32Array();
  }
}
