/** Minimal Node loader so parity.mjs can import the .ts source directly.
 *  esbuild strips the types and inlines the `?raw` abbreviation table. */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { transformSync } from 'esbuild';

export async function resolve(specifier, context, next) {
  if (specifier.endsWith('?raw')) {
    const base = specifier.slice(0, -4);
    const resolved = await next(base, context);
    return { ...resolved, url: resolved.url + '?raw', shortCircuit: true };
  }
  // src/ imports its siblings without a file extension, which Node's ESM
  // resolver does not do on its own — only the bundler does. Without this,
  // anything that reaches past the first module fails on `./rng`.
  if (specifier.startsWith('.') && context.parentURL?.startsWith('file:')) {
    const abs = path.resolve(path.dirname(fileURLToPath(context.parentURL)), specifier);
    if (!existsSync(abs) && existsSync(abs + '.ts')) {
      return { url: pathToFileURL(abs + '.ts').href, format: 'module', shortCircuit: true };
    }
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
