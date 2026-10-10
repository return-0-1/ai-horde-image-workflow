/**
 * CivitAI API 反代 — Cloudflare Worker
 * 
 * 用法：部署后访问 https://<your-worker>.workers.dev/api/v1/models?...
 * 所有请求转发到 https://civitai.com
 */

export default {
  async fetch(request) {
    // 只允许 GET/HEAD（不需要 POST/PUT）
    if (!['GET', 'HEAD'].includes(request.method)) {
      return new Response('Method Not Allowed', { status: 405 });
    }

    const url = new URL(request.url);
    const targetUrl = 'https://civitai.com' + url.pathname + url.search;

    // 构造转发请求，保留必要的请求头
    const headers = new Headers();
    headers.set('User-Agent', 'AI-Horde-Workflow/1.0');
    if (request.headers.get('Accept')) {
      headers.set('Accept', request.headers.get('Accept'));
    }

    try {
      const response = await fetch(targetUrl, {
        method: request.method,
        headers: headers,
        cf: {
          // 关键：CivitAI 也在 Cloudflare 上，需要指定解析目标
          // 避免 Worker → Worker 循环
          resolveOverride: 'civitai.com',
        },
      });

      // 构造响应，添加 CORS 头
      const newResponse = new Response(response.body, response);
      newResponse.headers.set('Access-Control-Allow-Origin', '*');
      newResponse.headers.set('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
      newResponse.headers.set('Access-Control-Allow-Headers', 'Content-Type, User-Agent');
      
      // 缓存 5 分钟（搜索结果不常变）
      newResponse.headers.set('Cache-Control', 'public, max-age=300');

      return newResponse;
    } catch (e) {
      return new Response(JSON.stringify({ error: 'Proxy error: ' + e.message }), {
        status: 502,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  },
};
