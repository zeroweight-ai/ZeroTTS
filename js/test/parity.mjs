/**
 * Parity check: the JS normalizer must agree with the Python one, exactly.
 *
 * The two implementations are hand-maintained copies of the same rules, so
 * nothing but a test keeps them in step — and a divergence is invisible in
 * normal use, showing up only as the browser and the package speaking the same
 * sentence differently.
 *
 *   node --experimental-vm-modules js/test/parity.mjs        # uses js/test/cases.json
 *   python js/test/gen_cases.py > js/test/cases.json         # regenerate from Python
 *
 * Run it after touching either src/zerotts/text_norm/vi_normalizer.py or
 * js/src/textNorm.ts.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const { normalizeViText } = await import(join(here, '..', 'src', 'textNorm.ts'));

const cases = JSON.parse(readFileSync(join(here, 'cases.json'), 'utf8'));
const failures = [];
for (const c of cases) {
  const got = normalizeViText(c.in);
  if (got !== c.expected) failures.push({ in: c.in, python: c.expected, js: got });
}

console.log(`${cases.length - failures.length}/${cases.length} match the Python normalizer`);
for (const f of failures.slice(0, 20)) {
  console.log(`  in:     ${JSON.stringify(f.in)}`);
  console.log(`  python: ${JSON.stringify(f.python)}`);
  console.log(`  js:     ${JSON.stringify(f.js)}`);
}
process.exit(failures.length ? 1 : 0);
