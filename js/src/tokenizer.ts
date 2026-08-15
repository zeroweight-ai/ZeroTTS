/**
 * BPE tokenization — port of zerotts/tokenizer.py.
 *
 * Uses @huggingface/transformers' tokenizer over the same `tokenizer.json` the
 * Python package reads, so the merge table and vocab are identical by
 * construction.
 *
 * `normalizeText` is reimplemented here rather than delegated, and it must stay
 * byte-identical to the Python version. A drift does not throw — it changes
 * which BPE pieces a string produces, so the model receives a segmentation it
 * never saw in training and simply sounds worse.
 */

import { PreTrainedTokenizer } from '@huggingface/transformers';

export const SPECIAL_TOKENS: Record<string, number> = {
  '<pad>': 0, '<bos>': 1, '<eot>': 2, '<soa>': 3,
  '<slot>': 4, '<eoa>': 5, '<en>': 6, '<vi>': 7,
};

/**
 * NFC + whitespace collapse. Case and punctuation are preserved verbatim.
 *
 * NFC matters most for Vietnamese: decomposed input ('e' + combining marks)
 * tokenizes into different pieces than the precomposed 'ệ' the vocab was
 * trained on, scattering the tone across tokens. Whitespace runs collapse
 * because whitespace is its own token — a tab or newline has none, so it would
 * encode as <unk> and delete the word boundary it stands for.
 */
export function normalizeText(text: string): string {
  return text.normalize('NFC').replace(/\s+/g, ' ');
}

export class BpeTokenizer {
  private constructor(private tokenizer: PreTrainedTokenizer) {}

  static async create(tokenizerJson: unknown): Promise<BpeTokenizer> {
    // A minimal tokenizer_config; the real configuration lives in tokenizer.json.
    const tokenizer = new PreTrainedTokenizer(tokenizerJson as never, {} as never);
    const t = new BpeTokenizer(tokenizer);
    t.assertSpecialIds();
    return t;
  }

  /** The exported graphs hardcode these ids, so a mismatch means the tokenizer
   *  does not belong to these weights. Fail loudly rather than generate noise. */
  private assertSpecialIds(): void {
    for (const [token, want] of Object.entries(SPECIAL_TOKENS)) {
      const got = (this.tokenizer as unknown as { model: { tokens_to_ids: Map<string, number> } })
        .model?.tokens_to_ids?.get(token);
      if (got !== undefined && got !== want) {
        throw new Error(
          `tokenizer special id mismatch: ${token} is ${got}, must be ${want}. ` +
          `This tokenizer does not belong to these weights.`);
      }
    }
  }

  /** Text -> [BOS | body | EOT] as int64, body truncated to maxLength. */
  encode(text: string, maxLength = 512): BigInt64Array {
    const encoded = this.tokenizer.encode(normalizeText(text), { add_special_tokens: false });
    const body = encoded.slice(0, maxLength);
    const ids = [SPECIAL_TOKENS['<bos>'], ...body, SPECIAL_TOKENS['<eot>']];
    return BigInt64Array.from(ids.map((v) => BigInt(v)));
  }

  decode(ids: ArrayLike<number | bigint>): string {
    const keep: number[] = [];
    for (let i = 0; i < ids.length; i++) {
      const v = Number(ids[i]);
      if (v >= Object.keys(SPECIAL_TOKENS).length) keep.push(v);
    }
    return this.tokenizer.decode(keep, { skip_special_tokens: true });
  }
}
