use std::sync::Arc;

use llguidance_parser::{
    api::TopLevelGrammar,
    toktrie::{InferenceCapabilities, StepArg, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Logger, TokenParser,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    pub type JsTokenizer;

    #[wasm_bindgen(method)]
    fn nVocab(this: &JsTokenizer) -> u32;

    // -1 if absent
    #[wasm_bindgen(method)]
    fn eosToken(this: &JsTokenizer) -> i32;

    #[wasm_bindgen(method)]
    fn bosToken(this: &JsTokenizer) -> i32;

    // format:
    // [flags0, len0, byte0_0, byte0_1, ..., flags1, len1, byte1_0, byte1_1, ...]
    #[wasm_bindgen(method)]
    fn tokenInfo(this: &JsTokenizer) -> Vec<u8>;

    #[wasm_bindgen(method)]
    fn tokenize(this: &JsTokenizer, text: &str) -> Vec<u32>;
}

struct JsTokenizerEnv {
    js_tok: JsTokenizer,
    trie: TokTrie,
}

#[wasm_bindgen]
pub struct JsTokEnv {
    tok_env: TokEnv,
}

unsafe impl Send for JsTokenizerEnv {}
unsafe impl Sync for JsTokenizerEnv {}

#[wasm_bindgen]
impl JsTokEnv {
    pub fn build(tok: JsTokenizer) -> JsTokEnv {
        console_error_panic_hook::set_once();
        JsTokEnv {
            tok_env: Arc::new(Self::new(tok)),
        }
    }

    fn new(js_tok: JsTokenizer) -> JsTokenizerEnv {
        let mut words = Vec::new();
        let mut idx = 0;
        let info = js_tok.tokenInfo();
        while idx + 1 < info.len() {
            let flags = info[idx] as usize;
            let mut len = info[idx + 1] as usize;
            len |= (flags & 0xf) << 8;
            if flags & 0x80 != 0 {
                panic!("future extension flag set in tokenInfo");
            }
            let is_special = (flags & 0x40) != 0;
            let mut word = Vec::with_capacity(len);
            if !is_special {
                for i in 0..len {
                    word.push(info[idx + 2 + i]);
                }
            }
            words.push(word);
            idx += 2 + len;
        }
        let trie = TokTrie::from(
            &TokRxInfo {
                vocab_size: js_tok.nVocab() as u32,
                tok_eos: js_tok.eosToken() as u32,
            },
            &words,
        );
        JsTokenizerEnv { js_tok, trie }
    }
}

impl TokenizerEnv for JsTokenizerEnv {
    fn stop(&self) -> ! {
        panic!("stop");
    }

    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie
            .tokenize_with_greedy_fallback(s, |s| self.js_tok.tokenize(s))
    }
}

type JsResult<T> = Result<T, JsError>;

#[derive(Debug, Serialize, Deserialize)]
pub struct ConstraintSettings {
    pub console_log_level: Option<u32>,
    pub buffer_log_level: Option<u32>,
}

#[wasm_bindgen]
pub struct ConstraintConfig {
    tok_env: JsTokEnv,
    inf_caps: InferenceCapabilities,
    settings: ConstraintSettings,
}

fn js_err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{}", e))
}

#[wasm_bindgen]
impl ConstraintConfig {
    pub fn _from_json(
        tok_env: JsTokEnv,
        inference_caps: &str,
        settings: &str,
    ) -> JsResult<ConstraintConfig> {
        let inf_caps = serde_json::from_str::<InferenceCapabilities>(inference_caps)?;
        let settings = serde_json::from_str::<ConstraintSettings>(settings)?;
        Ok(ConstraintConfig {
            tok_env,
            inf_caps,
            settings,
        })
    }

    pub fn new_constraint(&self, grammar: &str) -> JsResult<Constraint> {
        let grammar = serde_json::from_str::<TopLevelGrammar>(grammar)?;
        let logger = Logger::new(
            self.settings.buffer_log_level.unwrap_or(1),
            self.settings.console_log_level.unwrap_or(1),
        );
        let parser = TokenParser::from_llguidance_json(
            self.tok_env.tok_env.clone(),
            grammar,
            logger,
            self.inf_caps.clone(),
        )
        .map_err(js_err)?;
        Ok(Constraint {
            parser,
            temperature: 0.0,
            step_arg: StepArg::empty(),
        })
    }
}

#[wasm_bindgen]
pub struct Constraint {
    parser: TokenParser,
    pub temperature: f32,
    step_arg: StepArg,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdvanceResult {
    pub stop: bool,
    pub backtrack: u32,
    pub tokens: Vec<TokenId>,
}

#[wasm_bindgen]
impl Constraint {
    pub fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        self.parser.process_prompt(prompt)
    }

    fn check_error(&self) -> JsResult<()> {
        if let Some(e) = self.parser.error_message() {
            Err(JsError::new(&e))
        } else {
            Ok(())
        }
    }

    pub fn get_and_clear_logs(&mut self) -> String {
        let r = self.parser.logger.get_and_clear_logs();
        r
    }

    pub fn sampling_mask(&mut self) -> JsResult<Option<Vec<u32>>> {
        self.check_error()?;

        let arg = std::mem::replace(&mut self.step_arg, StepArg::empty());
        let r = self.parser.mid_process(arg);
        self.check_error()?;

        if let Some(t) = r.temperature {
            self.temperature = t;
        }

        let r = if let Some(m) = &r.sample_mask {
            Some(m.as_slice().to_vec())
        } else if let Some(s) = r.unconditional_splice() {
            let trie = self.parser.token_env.tok_trie();
            let mask = trie.singleton_token_set(s.ff_tokens[0]);
            Some(mask.as_slice().to_vec())
        } else {
            None
        };
        Ok(r)
    }

    pub fn advance_parser(&mut self, sampled: TokenId) -> JsResult<String> {
        self.check_error()?;

        if !self.step_arg.tokens.is_empty() || !self.step_arg.sampled.is_none() {
            return Err(JsError::new("advance_parser called twice"));
        }

        let arg = StepArg {
            backtrack: 0,
            tokens: vec![sampled],
            sampled: Some(sampled),
        };

        let r = match self.parser.advance_parser(arg) {
            None => AdvanceResult {
                stop: true,
                backtrack: 0,
                tokens: vec![],
            },
            Some(r) => AdvanceResult {
                stop: false,
                backtrack: r.backtrack,
                tokens: r.ff_tokens,
            },
        };

        self.check_error()?;

        self.step_arg = StepArg {
            backtrack: r.backtrack,
            tokens: r.tokens.clone(),
            sampled: None,
        };

        Ok(serde_json::to_string(&r)?)
    }
}
