/**
 * Vietnamese text normalization — port of `zerotts/text_norm/vi_normalizer.py`.
 *
 * Rewrites the written forms the model was never trained to voice (digits,
 * dates, clock times, versions, fractions, percentages, acronyms) into the
 * words a speaker would actually say:
 *
 *   "Ngày 23/8/2024 lúc 15h30, giảm 25%"
 *     -> "Ngày hai mươi ba tháng tám năm hai nghìn không trăm hai mươi tư
 *         lúc mười lăm giờ ba mươi phút, giảm hai mươi lăm phần trăm"
 *
 * **Keep this in step with the Python module.** The two are expected to produce
 * identical output for identical input; a divergence shows up as the browser
 * and the package speaking the same text differently, which nothing will catch
 * automatically. Structure, rule order and comments deliberately mirror it.
 *
 * Provenance: the expansion rules and `abbreviations.txt` are adapted from
 * soe-vinorm (MIT) — see NOTICE. None of its machinery is carried over: that
 * project tags tokens with a CRF and disambiguates acronyms with an ONNX
 * scorer. This is regex only.
 *
 * Two JS-specific notes:
 *   - There is no `re.VERBOSE`, so the scanner is assembled from an array of
 *     commented fragments and joined; the source stays readable, the pattern
 *     compiles compact.
 *   - The patterns use lookbehind, which needs a reasonably modern browser
 *     (Chrome 62+, Firefox 78+, Safari 16.4+).
 */

import abbreviationsRaw from '../../src/zerotts/text_norm/data/abbreviations.txt?raw';

// ── Vietnamese number words ─────────────────────────────────────────────────

const DIGIT: Record<string, string> = {
  '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
  '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín',
  ',': 'phẩy',
};

/** Position inside a 3-digit chunk: units, tens, hundreds. */
const UNIT_SINGLE = ['', 'mươi', 'trăm'];

/** Scale of each 3-digit chunk, least significant first. A number with more
 *  chunks than this falls back to digit-by-digit reading. */
const UNIT_TRIPLE = ['', 'nghìn', 'triệu', 'tỷ', 'nghìn tỷ', 'triệu tỷ', 'tỷ tỷ'];

const OP_WORDS: Record<string, string> = {
  '+': 'cộng', '-': 'trừ', '*': 'nhân', '/': 'chia', '^': 'mũ',
};

/**
 * Vietnamese number-pronunciation rules: 15 is "mười lăm" not "mười năm",
 * 21 "hai mươi mốt", 24 "hai mươi tư", 104 "một trăm linh tư".
 */
function applySandhi(text: string): string {
  return text
    .replace(/mười năm/g, 'mười lăm')
    .replace(/mươi năm/g, 'mươi lăm')
    .replace(/mươi bốn/g, 'mươi tư')
    .replace(/mươi một/g, 'mươi mốt')
    .replace(/linh bốn/g, 'linh tư');
}

/** Read a run of characters one symbol at a time ("2024" -> "hai không hai bốn"). */
export function expandDigit(digits: string): string {
  return [...digits.replace(/ /g, '')].map((c) => DIGIT[c] ?? c).join(' ');
}

/** Split a digit string into 3-digit chunks, most significant first. */
function splitChunks(number: string): string[] {
  const chunks: string[] = [];
  for (let i = number.length - 3; i >= 0; i -= 3) chunks.push(number.slice(i, i + 3));
  chunks.reverse();
  if (number.length % 3) chunks.unshift(number.slice(0, number.length % 3));
  return chunks;
}

/** Thrown when a number exceeds the scale table; caught by expandNumber. */
class TooLarge extends Error {}

/** One 3-digit chunk plus its scale word. An all-zero chunk is silent. */
function speakChunk(chunk: string, scaleIndex: number): string {
  if (chunk === '000') return '';

  let result = '';
  for (let pos = chunk.length - 1; pos >= 0; pos--) {
    if (pos === chunk.length - 1 && chunk[pos] === '0' && chunk.length > 1) {
      // trailing zero: "hai mươi", not "hai mươi không"
    } else if (pos === chunk.length - 2 && (chunk[pos] === '1' || chunk[pos] === '0')) {
      // Tens digit 1 -> "mười" (not "một mươi"); tens digit 0 -> "linh".
      if (pos === 0 && chunk[pos] === '0') {
        /* leading zero in a short chunk: silent */
      } else if (chunk[pos] === '1') {
        result = chunk[pos + 1] !== '0' ? `mười ${DIGIT[chunk[pos + 1]]}` : 'mười';
      } else {
        result = chunk[pos + 1] !== '0' ? `linh ${DIGIT[chunk[pos + 1]]}` : '';
      }
    } else {
      result = DIGIT[chunk[pos]] + ' ' + UNIT_SINGLE[chunk.length - pos - 1]
        + (result ? ' ' + result : '');
    }
  }

  if (scaleIndex >= UNIT_TRIPLE.length) throw new TooLarge();
  return [result.trim(), UNIT_TRIPLE[scaleIndex]].join(' ').trim();
}

/**
 * Speak an integer, a decimal, or a small arithmetic expression.
 *
 * Handles the sign ("-7" -> "trừ bảy"), '.' as a thousands separator, ',' as
 * the Vietnamese decimal comma ("12,5" -> "mười hai phẩy năm"), '.' as a
 * decimal point when at most two digits follow, and operators between numbers.
 */
export function expandNumber(number: string): string {
  const original = number;
  try {
    let sign = '';
    if (number[0] === '-' || number[0] === '+') {
      sign = number[0] === '+' ? 'cộng' : 'trừ';
      number = number.slice(1);
    }
    while (number.length > 1 && number[0] === '0' && /\d/.test(number[1])) {
      number = number.slice(1);
    }
    number = number.trim();

    // More than one number in the string -> an expression: speak each number,
    // then turn the operators between them into words.
    const matches = number.match(/[-+]?[0-9.,]+/g) ?? [];
    if (matches.length > 1 || (matches.length === 1 && matches[0] !== number)) {
      return number
        .replace(/\s*([-+]?[0-9.,]+)\s*/g, (_m, n: string) => ` ${expandNumber(n)} `)
        .trim()
        .replace(/[-+*/^]/g, (c) => OP_WORDS[c] ?? c);
    }

    number = number.replace(/[^0-9.,]/g, '');

    const count = (s: string, c: string) => [...s].filter((x) => x === c).length;
    let decimalPart = '';
    if (count(number, ',') === 1) {
      number = number.replace(/\./g, '');
      const parts = number.split(',');
      decimalPart = `phẩy ${expandDigit(parts[parts.length - 1])}`;
      number = parts.slice(0, -1).join('');
    } else if (count(number, '.') === 1 && number.slice(number.indexOf('.')).length <= 3) {
      number = number.replace(/,/g, '');
      const parts = number.split('.');
      decimalPart = `chấm ${expandDigit(parts[parts.length - 1])}`;
      number = parts.slice(0, -1).join('');
    } else {
      number = number.replace(/\./g, '');
    }

    const chunks = splitChunks(number);
    const spoken: string[] = [];
    chunks.forEach((chunk, i) => {
      const part = speakChunk(chunk, chunks.length - i - 1);
      if (part) spoken.push(part);
    });

    return `${sign} ${applySandhi(spoken.join(' '))} ${decimalPart}`.replace(/\s+/g, ' ').trim();
  } catch (err) {
    if (err instanceof TooLarge) return expandDigit(original);
    throw err;
  }
}

const num = (v: string) => expandNumber(v);

/** The fourth month is "tháng tư", never "tháng bốn". */
const month = (v: string) => (v.replace(/^0+/, '') === '4' ? 'tư' : expandNumber(v));

// ── abbreviations ───────────────────────────────────────────────────────────

let abbreviations: Map<string, string> | null = null;

/** Parse "ABBR:reading[,reading...]" lines. The first reading wins — the
 *  upstream project picks with an ONNX likelihood scorer, which is not ported. */
export function loadAbbreviations(): Map<string, string> {
  if (abbreviations) return abbreviations;
  const table = new Map<string, string>();
  for (const line of abbreviationsRaw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes(':')) continue;
    const idx = trimmed.indexOf(':');
    const abbr = trimmed.slice(0, idx);
    if (!table.has(abbr)) table.set(abbr, trimmed.slice(idx + 1).split(',')[0]);
  }
  abbreviations = table;
  return table;
}

function expandAbbreviation(token: string): string | null {
  const table = loadAbbreviations();
  for (const key of [token, token.replace(/[.-]/g, '')]) {
    const hit = table.get(key);
    if (hit) return hit;
  }
  const parts = token.split(/[.-]/).filter(Boolean);
  if (parts.length > 1 && parts.every((p) => p.length >= 2 && table.has(p))) {
    return parts.map((p) => table.get(p)!).join(' ');
  }
  return null;
}

// ── pattern scanner ─────────────────────────────────────────────────────────

const VN_UPPER = 'A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠ-Ỹ';
const VN_LOWER = 'a-zàáâãèéêìíòóôõùúýăđĩũơưạ-ỹ';
const DATE_CUES = 'ngày|mùng|mồng|hôm|sáng|trưa|chiều|tối|đêm|từ|đến|và|hoặc';

/** Abbreviations that qualify a following proper noun, so "TP. HCM" is one
 *  unit and the dot abbreviates rather than ends a sentence. A closed set on
 *  purpose: a general rule would eat the break in "làm ở FPT. Sau đó ...". */
const PREFIX_ABBR = 'TP|TX|TT|KP|[QPH]';

/** Roman numerals as Vietnamese ordinals, for "quý III" / "lần thứ IV". */
const ROMAN: Record<string, string> = {
  I: 'một', II: 'hai', III: 'ba', IV: 'bốn', V: 'năm',
  VI: 'sáu', VII: 'bảy', VIII: 'tám', IX: 'chín', X: 'mười',
};

/** Words marking the next token as an ordinal, so a roman numeral after one
 *  is a number rather than an acronym. */
const ROMAN_CUES = ['quý', 'thứ', 'khóa', 'kỳ', 'đợt', 'loại', 'chương', 'phần', 'thế kỷ'];

/** A letter run as capitals, unchanged: "ab" -> "AB". Deliberately not spaced
 *  into single letters and deliberately not a Vietnamese letter-name table —
 *  a capitalised run is already enough for the model to spell it. */
const spellLetters = (letters: string) => letters.toUpperCase();

/** Insert a space at every internal case boundary of a mixed-case token.
 *
 *  Case is tested with toUpperCase()/toLowerCase() comparisons, NOT a regex
 *  letter class: the precomposed Vietnamese letters live in Latin Extended
 *  Additional where upper and lower case interleave, so a range like `Ạ-Ỹ`
 *  also matches "ế" and would split "Chiếc" into "Chi ếc".
 *
 *  Only an ACRONYM is split off — an uppercase run of two or more. That is the
 *  difference between "Chat|GPT" and "MacBook": the second is an ordinary
 *  CamelCase brand read as one word ("YouTube", "TikTok"). A single leading
 *  lowercase letter does not open a boundary either, so "iPhone" survives. */
const isUp = (c: string) => !!c && c !== c.toLowerCase() && c === c.toUpperCase();
const isLow = (c: string) => !!c && c !== c.toUpperCase() && c === c.toLowerCase();

export function splitCamelCase(text: string): string {
  return text.split(/(\s+)/).map((token) => {
    if (!token || /^\s+$/.test(token)) return token;
    const chars = [...token];
    if (!(chars.some(isUp) && chars.some(isLow))) return token;
    const out: string[] = [];
    let lowerRun = 0;
    for (let i = 0; i < chars.length; i++) {
      const c = chars[i];
      if (isUp(c) && i > 0) {
        let run = 0;
        while (i + run < chars.length && isUp(chars[i + run])) run++;
        if (lowerRun >= 2 && run >= 2) out.push(' ');
        else if (isUp(chars[i - 1]) && isLow(chars[i + 1] ?? '')) out.push(' ');
      }
      lowerRun = isLow(c) ? lowerRun + 1 : 0;
      out.push(c);
    }
    return out.join('');
  }).join('');
}

/** URLs and emails are matched and skipped wholesale — normalizing inside one
 *  produces nonsense, and this module has no URL reader. */
const PROTECTED = new RegExp(
  [
    '(?:https?|ftp)://\\S+',
    'www\\.\\S+',
    '[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+',
    '\\b[\\w-]+(?:\\.[\\w-]+)*\\.(?:com|net|org|vn|io|edu|gov|info|dev|ai)\\b(?:/\\S*)?',
  ].join('|'),
  'gi',
);

// One ordered alternation: the first alternative that matches at a position
// wins, so more specific shapes come first (a full date before a month-year
// before a fraction; percent before a bare number).
const SCANNER = new RegExp([
  // time: HH:MM:SS / HHhMMmSS
  String.raw`(?<![\d:])(?<t_h>[01]?\d|2[0-3])[:hg](?<t_m>[0-5]?\d)[:mp](?<t_s>[0-5]?\d)(?![\d:])`,
  // date: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
  String.raw`(?<![\d/.\-])(?<d_d>0?[1-9]|[12]\d|3[01])(?<d_sep>[/.\-])(?<d_m>0?[1-9]|1[0-2])\k<d_sep>(?<d_y>[12]\d{3})(?![\d/-])(?!\.\d)`,
  // month-year: MM/YYYY, MM-YYYY
  String.raw`(?<![\d/.\-])(?<my_m>0?[1-9]|1[0-2])[/\-](?<my_y>1\d{3}|20\d{2}|21\d{2})(?![\d/-])(?!\.\d)`,
  // day-month: DD/MM after a date cue word only (else it is a fraction)
  String.raw`(?<=\b)(?<dm_cue>${DATE_CUES})(?<dm_gap>\s+)(?<dm_d>0?[1-9]|[12]\d|3[01])[/\-](?<dm_m>0?[1-9]|1[0-2])(?![\d/-])(?!\.\d)`,
  // time: HH:MM (2-digit minutes), HHhMM, HHgMM
  String.raw`(?<![\d:])(?<hm_h>[01]?\d|2[0-3]):(?<hm_m>[0-5]\d)(?![\d:])`,
  String.raw`(?<![\d:])(?<hg_h>[01]?\d|2[0-3])[hg](?<hg_m>[0-5]\d)(?![\dhg])`,
  // time: HH h ("15h", "9g")
  String.raw`(?<![\d:])(?<h_h>[01]?\d|2[0-3])[hg](?!\w)`,
  // version: v1.2 / V1.2.3, or a bare 3+ group dotted number
  String.raw`(?<![\w.])(?<vp>[vV])(?<v_num>\d+(?:\.\d+)+)(?!\w|\.\d|,\d)`,
  String.raw`(?<![\w.,])(?<v_bare>\d+(?:\.\d+){2,})(?!\w|\.\d|,\d)`,
  // fraction: a/b (dates already claimed above)
  String.raw`(?<![\w/.,])(?<f_a>\d+)\s*/\s*(?<f_b>\d+)(?![\w/,])(?!\.\d)`,
  // degree: 38°C / 38 °C — before the number branch, same reason as percent
  String.raw`(?<![\w.,])(?<deg_n>-?\d[\d.,]*)\s*°\s*(?<deg_u>[CF])?(?![${VN_LOWER}])`,
  // percent — BEFORE the number branch, or that claims the digits and leaves
  // a bare '%' behind
  String.raw`(?<![\w.,])(?<pct_num>[-+]?\d[\d.,]*?)\s*%(?!\w)`,
  // number: integer / decimal / thousand-separated / signed / arithmetic
  String.raw`(?<![\w.,])(?<n_num>[-+]?\d[\d.,]*(?:\s*[*^+]\s*[-+]?\d[\d.,]*|\s+[-/]\s+[-+]?\d[\d.,]*|[*^]\s*[-+]?\d[\d.,]*)*)(?![.,]?\d)`,
  // prefix abbreviation + proper noun: "TP. HCM" — the dot abbreviates
  String.raw`(?<![\w.])(?<pfx>${PREFIX_ABBR})\.(?=\s+[${VN_UPPER}])`,
  // acronym pair over a slash: USD/VND, KM/H — left verbatim, matched only so
  // the abbreviation branch cannot claim one half
  String.raw`(?<![\w/])(?<ap>[${VN_UPPER}]{1,6}/[${VN_UPPER}]{1,6})(?![\w/])`,
  // alphanumeric code: AB-1234, VN-215, SE1 — an identifier, not a quantity
  String.raw`(?<![\w-])(?<code_a>[${VN_UPPER}]{1,4})-?(?<code_n>\d{1,6})(?![\w-])`,
  // abbreviation: uppercase acronym, optionally dotted
  String.raw`(?<![\w.])(?<abbr>[${VN_UPPER}][${VN_UPPER}\d]+(?:\.[${VN_UPPER}][${VN_UPPER}\d]*)*)(?![${VN_LOWER}\d])`,
  // at sign — emails never reach here, PROTECTED claims them first
  String.raw`(?<at>@)`,
].join('|'), 'gu');

const THOUSANDS = /^\d{1,3}(?:\.\d{3})+$/;
const NUM_TRAIL = /[.,\s]+$/;

function speakTime(h: string, m?: string, s?: string): string {
  let out = `${num(h)} giờ`;
  // "17:00" is "mười bảy giờ" out loud, not "mười bảy giờ không phút" — a whole
  // hour drops its minutes UNLESS the text also spelled out seconds, in which
  // case "chín giờ không phút sáu giây" is what a speaker says.
  if (m != null && (s != null || m.replace(/0/g, '') !== '')) {
    out += ` ${num(m)} phút`;
  }
  if (s != null) out += ` ${num(s)} giây`;
  return out;
}

function expandMatch(m: RegExpExecArray): string {
  const g = (m.groups ?? {}) as Record<string, string | undefined>;
  const whole = m[0];

  if (g.t_h !== undefined) return speakTime(g.t_h, g.t_m, g.t_s);
  if (g.d_d !== undefined) {
    return `${num(g.d_d)} tháng ${month(g.d_m!)} năm ${num(g.d_y!)}`;
  }
  if (g.my_m !== undefined) {
    // Only add "tháng" when the text doesn't already say it right before.
    const before = m.input.slice(Math.max(0, m.index - 8), m.index).toLowerCase();
    const lead = /tháng\s*$/.test(before) ? '' : 'tháng ';
    return `${lead}${month(g.my_m)} năm ${num(g.my_y!)}`;
  }
  if (g.pfx !== undefined) return expandAbbreviation(g.pfx) ?? g.pfx;
  if (g.ap !== undefined) return g.ap;
  if (g.code_a !== undefined) {
    // The letters stay joined — a capitalised run is enough for the model to
    // spell them — but the digits are spaced so they are read one at a time:
    // "AB-1234" -> "AB 1 2 3 4", never "một nghìn hai trăm ba mươi tư".
    return `${spellLetters(g.code_a)} ${g.code_n!.split('').join(' ')}`;
  }
  if (g.deg_n !== undefined) {
    const unit = g.deg_u === 'C' ? ' xê' : g.deg_u === 'F' ? ' ép' : '';
    return `${expandNumber(g.deg_n)} độ${unit}`;
  }
  if (g.dm_d !== undefined) {
    const cue = g.dm_cue!.toLowerCase();
    if (cue === 'và' || cue === 'hoặc') {
      // Only a coordinating cue: require a genuine date cue nearby, else
      // "3 và 4/5" is arithmetic, not the 4th of May.
      const before = m.input.slice(Math.max(0, m.index - 40), m.index).toLowerCase();
      if (!new RegExp(`\\b(?:${DATE_CUES})\\b`, 'u').test(before)) return whole;
    }
    return `${g.dm_cue}${g.dm_gap}${num(g.dm_d)} tháng ${month(g.dm_m!)}`;
  }
  if (g.hm_h !== undefined) return speakTime(g.hm_h, g.hm_m);
  if (g.hg_h !== undefined) return speakTime(g.hg_h, g.hg_m);
  if (g.h_h !== undefined) return speakTime(g.h_h);
  if (g.v_num !== undefined) {
    // The "v"/"V" is kept as written, not read as a word — matching the Python
    // module, which leaves the prefix alone.
    return `${g.vp} ${g.v_num.split('.').map(num).join(' chấm ')}`;
  }
  if (g.v_bare !== undefined) {
    if (THOUSANDS.test(g.v_bare)) return expandNumber(g.v_bare);
    return g.v_bare.split('.').map(num).join(' chấm ');
  }
  if (g.f_a !== undefined) return `${num(g.f_a)} trên ${num(g.f_b!)}`;
  if (g.pct_num !== undefined) {
    return `${expandNumber(g.pct_num.replace(/[.,]+$/, ''))} phần trăm`;
  }
  if (g.n_num !== undefined) {
    let raw = g.n_num;
    let suffix = '';
    const trail = NUM_TRAIL.exec(raw);
    if (trail && !raw.slice(trail.index).replace(/ /g, '').replace(/[.,]+$/, '')) {
      suffix = raw.slice(trail.index);
      raw = raw.slice(0, trail.index);
    }
    if (!raw) return whole;
    // A long separator-less digit run is an identifier, not a quantity — a
    // phone number, an account number. Reading it as one huge cardinal is
    // worse than leaving it.
    if (/^[-+]?\d+$/.test(raw) && raw.replace(/^[-+]/, '').length > 8) return whole;
    // digit-letter-digit ("1m65", "3G4") is a compound written form, not a
    // quantity that happens to touch a unit.
    if (/^[^\W\d_]\d/u.test(m.input.slice(m.index + whole.length, m.index + whole.length + 2))) {
      return whole;
    }
    return expandNumber(raw) + suffix;
  }
  if (g.abbr !== undefined) {
    const token = g.abbr;
    if (token in ROMAN) {
      const before = m.input.slice(Math.max(0, m.index - 12), m.index).toLowerCase();
      if (ROMAN_CUES.some((cue) => new RegExp(`${cue}\\s*$`).test(before))) return ROMAN[token];
    }
    const expansion = expandAbbreviation(token);
    if (expansion !== null) {
      // A parenthesized acronym right after its own expansion is a gloss.
      const back = m.input.slice(Math.max(0, m.index - expansion.length - 4), m.index);
      if (back.trimEnd().endsWith('(') && back.toLowerCase().includes(expansion.toLowerCase())) {
        return spellLetters(token);
      }
    }
    return expansion ?? token;
  }
  if (g.at !== undefined) return 'a còng';
  return whole;
}

const isAlnum = (c: string) => !!c && /[\p{L}\p{N}]/u.test(c);

/** Expand every match in a stretch of text, keeping each expansion a separate
 *  word: a written form can sit flush against a unit or letter ("1.250.000đ"),
 *  and the spoken form must not, or the tokenizer sees one impossible word. */
function scan(text: string): string {
  SCANNER.lastIndex = 0;
  let out = '';
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = SCANNER.exec(text)) !== null) {
    if (m[0] === '') { SCANNER.lastIndex++; continue; }
    let expanded = expandMatch(m);
    if (expanded !== m[0]) {
      const before = m.index > 0 ? text[m.index - 1] : '';
      const after = text[m.index + m[0].length] ?? '';
      if (isAlnum(before) && !/^\s/.test(expanded)) expanded = ' ' + expanded;
      if (isAlnum(after) && !/\s$/.test(expanded)) expanded = expanded + ' ';
    }
    out += text.slice(last, m.index) + expanded;
    last = m.index + m[0].length;
  }
  return out + text.slice(last);
}

/**
 * Expand the supported non-standard forms into spoken Vietnamese.
 *
 * Punctuation, casing and every unsupported form are preserved verbatim, and
 * URLs/emails are skipped wholesale, so this is safe over already-normalized
 * text and is a no-op (beyond NFC) on text containing none of these forms.
 */
export function normalizeViText(text: string): string {
  if (!text || !text.trim()) return text;

  // NFC first: the dictionary keys and letter classes are precomposed, so
  // decomposed input would miss.
  text = text.normalize('NFC');

  const out: string[] = [];
  let last = 0;
  PROTECTED.lastIndex = 0;
  let p: RegExpExecArray | null;
  while ((p = PROTECTED.exec(text)) !== null) {
    out.push(scan(splitCamelCase(text.slice(last, p.index))));
    out.push(p[0]);
    last = p.index + p[0].length;
  }
  out.push(scan(splitCamelCase(text.slice(last))));

  // Expansions leave double spaces where a compact form used to be.
  return out.join('').replace(/[ \t]{2,}/g, ' ');
}
