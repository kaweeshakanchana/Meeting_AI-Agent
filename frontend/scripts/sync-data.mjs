import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..', '..');
const outDir = path.resolve(__dirname, '..', 'public', 'data');

fs.mkdirSync(outDir, { recursive: true });

const filesToSync = [
  { src: path.join(rootDir, 'answers.json'), dest: path.join(outDir, 'answers.json') },
  { src: path.join(rootDir, 'questions.json'), dest: path.join(outDir, 'questions.json') },
  { src: path.join(rootDir, 'data', 'bitemporal_graph.json'), dest: path.join(outDir, 'bitemporal_graph.json') },
  { src: path.join(rootDir, 'data', 'evaluation_results.json'), dest: path.join(outDir, 'evaluation_results.json') },
];

for (const { src, dest } of filesToSync) {
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dest);
    console.log(`[sync-data] Synced ${path.basename(src)} -> public/data/${path.basename(dest)}`);
  } else {
    console.warn(`[sync-data] Warning: Source file not found: ${src}`);
  }
}
