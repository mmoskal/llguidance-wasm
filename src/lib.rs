mod utils;

use std::sync::Arc;

use llguidance_parser::toktrie::{TokTrie, TokenId, TokenizerEnv};
use wasm_bindgen::prelude::*;

/*

export interface TokenizerInfo {
    nVocab: number;
    eosToken: number;
    bosToken?: number;
    isSpecialToken: boolean[];
    tokenBytes: Uint8Array[];
}

*/

#[wasm_bindgen]
extern "C" {
    pub type JsTokenizer;

    #[wasm_bindgen]
    fn newTokenizer() -> JsTokenizer;

    #[wasm_bindgen(method)]
    fn nVocab(this: &JsTokenizer) -> u32;

    // -1 if absent
    #[wasm_bindgen(method)]
    fn eosToken(this: &JsTokenizer) -> i32;

    #[wasm_bindgen(method)]
    fn bosToken(this: &JsTokenizer) -> i32;

    #[wasm_bindgen(method)]
    fn tokenInfo(this: &JsTokenizer) -> Vec<u8>;

    #[wasm_bindgen(method)]
    fn tokenize(this: &JsTokenizer, text: &str) -> Vec<u32>;
}

struct JsTokenizerEnv {
    js_tok: Arc<JsTokenizer>,
    trie: Arc<TokTrie>,
}

unsafe impl Send for JsTokenizerEnv {}

impl TokenizerEnv {
    fn new(js_tok: JsTokenizer) -> Self {
        let js_tok = Arc::new(js_tok);
        let trie = Arc::new(TokTrie::new(js_tok.nVocab() as usize));
        let env = JsTokenizerEnv { js_tok, trie };
        env.trie.init_tokenizer(&env);
        env
    }
}
impl TokenizerEnv for JsTokenizerEnv {
    fn stop(&self) -> ! {
        panic!("stop");
    }

    fn tok_trie(&self) -> &TokTrie {
        self.trie.as_ref()
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie
            .tokenize_with_greedy_fallback(s, |s| self.js_tok.tokenize(s))
    }
}

#[wasm_bindgen]
pub fn tokeni(tok: &JsTokenizer) {
    let t = tok.eosToken();
    let t2 = newTokenizer();
}
