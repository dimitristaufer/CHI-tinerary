import { defineConfig } from 'vite';

export default defineConfig({
  base: '/CHI-tinerary/',
  plugins: [
    {
      name: 'dev-profile-fetch-proxy',
      configureServer(server) {
        server.middlewares.use('/__fetch', async (req, res) => {
          try {
            const requestUrl = new URL(req.url || '', 'http://localhost');
            const target = requestUrl.searchParams.get('url');
            if (!target) {
              res.statusCode = 400;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Missing url query parameter.' }));
              return;
            }

            let parsedTarget;
            try {
              parsedTarget = new URL(target);
            } catch {
              res.statusCode = 400;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Invalid target url.' }));
              return;
            }

            if (!['http:', 'https:'].includes(parsedTarget.protocol)) {
              res.statusCode = 400;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Only http/https targets are allowed.' }));
              return;
            }

            const host = parsedTarget.hostname.toLowerCase();
            const allowed =
              host.includes('scholar.google.') ||
              host === 'researchgate.net' ||
              host.endsWith('.researchgate.net') ||
              host === 'r.jina.ai';

            if (!allowed) {
              res.statusCode = 403;
              res.setHeader('content-type', 'application/json; charset=utf-8');
              res.end(JSON.stringify({ error: 'Host not allowed for dev proxy.' }));
              return;
            }

            const upstream = await fetch(parsedTarget.toString(), {
              method: 'GET',
              redirect: 'follow',
              headers: {
                'user-agent':
                  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'accept-language': 'en-US,en;q=0.9',
              },
            });

            const body = await upstream.text();
            res.statusCode = upstream.status;
            res.setHeader('content-type', 'text/plain; charset=utf-8');
            res.end(body);
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            res.statusCode = 502;
            res.setHeader('content-type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: `Dev proxy request failed: ${message}` }));
          }
        });
      },
    },
  ],
});
