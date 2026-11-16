import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const pythonExe = 'C:\\Users\\YOGA\\AppData\\Local\\Programs\\Python\\Python313\\python.exe';

function liveAgentApiPlugin() {
  return {
    name: 'catalist-live-agent-api',
    configureServer(server) {
      server.middlewares.use('/api/live-query', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end(JSON.stringify({ error: 'Method not allowed' }));
          return;
        }
        let body = '';
        req.on('data', (chunk) => {
          body += chunk.toString();
        });
        req.on('end', () => {
          const child = spawn(
            pythonExe,
            ['-m', 'src.resolution.live_agent', '--json-stdin'],
            { cwd: projectRoot }
          );
          let stdout = '';
          let stderr = '';
          child.stdout.on('data', (d) => {
            stdout += d.toString();
          });
          child.stderr.on('data', (d) => {
            stderr += d.toString();
          });
          child.on('close', (code) => {
            res.setHeader('Content-Type', 'application/json');
            if (code !== 0) {
              res.statusCode = 500;
              res.end(
                JSON.stringify({
                  error: stderr || `Python live_agent exited with code ${code}`,
                })
              );
              return;
            }
            res.statusCode = 200;
            res.end(stdout);
          });
          child.stdin.write(body || '{}');
          child.stdin.end();
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), liveAgentApiPlugin()],
  server: {
    port: 5173,
    open: false,
    allowedHosts: true,
  },
});
