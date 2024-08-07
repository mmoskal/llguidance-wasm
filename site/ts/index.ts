import { ConstraintConfig } from "llguidance-wasm";
import { grm, gen, GrammarNode } from "guidance-ts";
import {
  constraintConfig,
  ConstraintSettings,
  Sequence,
  JsTokenizer,
} from "./wrappers.js";



export async function main() {
  testLogProbs();

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
