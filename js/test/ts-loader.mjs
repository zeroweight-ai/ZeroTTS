/** Minimal Node loader so parity.mjs can import the .ts source directly.
 *  esbuild strips the types and inlines the `?raw` abbreviation table. */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { transformSync } from 'esbuild';

export async function resolve(specifier, context, next) {
  if (specifier.endsWith('?raw')) {
    const base = specifier.slice(0, -4);
    const resolved = await next(base, context);
    return { ...resolved, url: resolved.url + '?raw', shortCircuit: true };
  }
  return next(specifier, context);
}

export async function load(url, context, next) {
  if (url.endsWith('?raw')) {
    const raw = readFileSync(fileURLToPath(url.slice(0, -4)), 'utf8');
    return { format: 'module', source: `export default ${JSON.stringify(raw)};`, shortCircuit: true };
  }
  if (url.endsWith('.ts')) {
    const source = readFileSync(fileURLToPath(url), 'utf8');
    const { code } = transformSync(source, { loader: 'ts', format: 'esm', target: 'node18' });
    return { format: 'module', source: code, shortCircuit: true };
  }
  return next(url, context);
}
