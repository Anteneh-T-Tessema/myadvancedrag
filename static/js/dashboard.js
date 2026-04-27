/* Advanced RAG Studio — Dashboard JS */
const API = 'http://localhost:7891/api';

// ── State ──────────────────────────────────────────────────────
let currentConfig = {};

// ── Nav ────────────────────────────────────────────────────────
const views = ['query','ingest','router','hyde','hardware','config','stats'];
const titles = {
  query:  ['Query Explorer','Run queries through the full Advanced RAG pipeline with live stage tracing'],
  ingest: ['Ingest Documents','Add documents to the vector index using semantic, parent-child, or fixed chunking'],
  router: ['Semantic Router','Test intent classification — routes queries to the optimal pipeline target'],
  hyde:   ['HyDE Transform','Generate hypothetical documents to bridge the vocabulary gap in retrieval'],
  hardware: ['Hardware & Models','Inspect your hardware and manage local AI models via Ollama'],
  config: ['Pipeline Configuration','Tune every parameter of the Advanced RAG pipeline in real time'],
  stats:  ['Index Stats','Monitor ingested documents, chunk counts, and index health'],
};

document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', e => {
    e.preventDefault();
    const v = el.dataset.view;
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    el.classList.add('active');
    document.querySelectorAll('.view').forEach(s => s.classList.remove('active'));
    document.getElementById('view-' + v).classList.add('active');
    document.getElementById('page-title').textContent = titles[v][0];
    document.getElementById('page-subtitle').textContent = titles[v][1];
    if (v === 'router') loadRoutes();
    if (v === 'hardware') { loadHardware(); loadPopularModels(); }
    if (v === 'config') loadConfig();
    if (v === 'stats')  loadStats();
  });
});

// ── API helpers ─────────────────────────────────────────────────
async function api(path, method='GET', body=null) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  return r.json();
}

// ── Status ─────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const h = await api('/health');
    document.getElementById('status-dot').className = 'status-dot ok';
    document.getElementById('status-text').textContent = 'API Online';
    document.getElementById('index-size-footer').textContent = `${h.index_size} chunks indexed`;
    const model = h.ollama_models[0] || 'no model';
    document.getElementById('current-model-label').textContent = model;
    currentConfig.llm_model = model;
  } catch {
    document.getElementById('status-dot').className = 'status-dot err';
    document.getElementById('status-text').textContent = 'API Offline';
  }
}

document.getElementById('refresh-btn').addEventListener('click', checkStatus);

// ── Loading Overlay ─────────────────────────────────────────────
function showLoading(text, stages=[]) {
  document.getElementById('loading-text').textContent = text;
  document.getElementById('loading-stages').innerHTML = stages.map(s=>`<div>${s}</div>`).join('');
  document.getElementById('loading-overlay').classList.add('active');
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.remove('active');
}

// ── Example Queries ─────────────────────────────────────────────
document.querySelectorAll('.eq-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('query-input').value = btn.dataset.q;
  });
});

// ── Run Query ───────────────────────────────────────────────────
document.getElementById('run-query-btn').addEventListener('click', runQuery);
document.getElementById('query-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.metaKey) runQuery();
});

async function runQuery() {
  const q = document.getElementById('query-input').value.trim();
  if (!q) return;

  const useRouter  = document.getElementById('toggle-router').checked;
  const useHyde    = document.getElementById('toggle-hyde').checked;
  const useHybrid  = document.getElementById('toggle-hybrid').checked;
  const useMerge   = document.getElementById('toggle-merge').checked;
  const genAnswer  = document.getElementById('toggle-answer').checked;

  // Push temp config
  await api('/config', 'POST', {
    use_router: useRouter, use_hyde: useHyde,
    use_hybrid: useHybrid, use_auto_merge: useMerge,
  });

  showLoading('Running Advanced RAG Pipeline…', [
    useRouter ? '🔀 Semantic routing…' : '',
    useHyde   ? '✨ HyDE transformation…' : '',
    '🔍 Hybrid search (Dense + BM25 + RRF)…',
    useMerge  ? '🧩 Auto-merging parent chunks…' : '',
    genAnswer ? '🤖 Generating answer…' : '',
  ].filter(Boolean));

  try {
    const result = await api('/query', 'POST', { query: q, generate_answer: genAnswer });
    hideLoading();
    renderTrace(result);
    renderChunks(result.retrieved_chunks || []);
    if (genAnswer && result.answer) renderAnswer(result);
    checkStatus();
  } catch (err) {
    hideLoading();
    renderError('Query failed: ' + err.message);
  }
}

// ── Render Trace ────────────────────────────────────────────────
function renderTrace(result) {
  const el = document.getElementById('pipeline-trace');
  const ms = result.total_latency_ms;
  document.getElementById('total-latency').textContent = ms ? `${ms}ms total` : '—';

  const stageColors = {
    semantic_routing: '#8b5cf6',
    hyde_transformation: '#f59e0b',
    hybrid_search: '#22d3ee',
    llm_generation: '#10b981',
  };

  el.innerHTML = result.pipeline_stages.map(s => {
    const color = stageColors[s.stage] || '#7a8499';
    const label = s.stage.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
    let body = '';

    if (s.stage === 'semantic_routing' && s.result) {
      const r = s.result;
      const cls = 'target-' + r.target;
      body = `Route: <span class="route-target ${cls}">${r.target}</span>  Confidence: <b>${(r.confidence*100).toFixed(1)}%</b>  Fallback: ${r.fallback}`;
    } else if (s.stage === 'hyde_transformation' && s.result) {
      const r = s.result;
      body = r.success
        ? `✅ Generated in ${r.latency_ms}ms\n\n${r.hypothetical_doc}`
        : `⚠️ Failed: ${r.error} — using original query`;
    } else if (s.stage === 'hybrid_search') {
      body = `Retrieved ${s.results_count} chunks in ${s.latency_ms}ms\nSearch query: ${s.search_query_used}`;
    } else if (s.stage === 'llm_generation' && s.result) {
      body = `Model: ${s.result.model || '—'}  Latency: ${s.result.latency_ms}ms  Chunks used: ${s.result.context_chunks_used}`;
    } else {
      body = JSON.stringify(s.result || s, null, 2);
    }

    const time = s.result?.latency_ms ? `${s.result.latency_ms}ms` : (s.latency_ms ? `${s.latency_ms}ms` : '');

    return `<div class="pipeline-stage">
      <div class="stage-header" onclick="this.parentElement.classList.toggle('stage-collapsed')">
        <div class="stage-dot" style="background:${color}"></div>
        <div class="stage-label">${label}</div>
        <div class="stage-time">${time}</div>
      </div>
      <div class="stage-body"><pre>${body}</pre></div>
    </div>`;
  }).join('');

  if (!result.pipeline_stages.length) {
    el.innerHTML = '<div class="empty-state"><div class="empty-icon">⚡</div><div class="empty-title">No pipeline stages ran</div><div class="empty-desc">Make sure the demo corpus is loaded first.</div></div>';
  }
}

// ── Render Chunks ───────────────────────────────────────────────
function renderChunks(chunks) {
  const panel = document.getElementById('chunks-panel');
  const container = document.getElementById('chunks-container');
  const meta = document.getElementById('chunk-meta-row');

  if (!chunks.length) { panel.style.display='none'; return; }
  panel.style.display='block';

  const types = [...new Set(chunks.map(c=>c.chunk_type))];
  meta.innerHTML = `<span class="chunk-meta-pill">${chunks.length} chunks</span>` +
    types.map(t=>`<span class="chunk-meta-pill">${t}</span>`).join('');

  const maxRRF = Math.max(...chunks.map(c=>c.rrf_score));

  container.innerHTML = chunks.map((c,i) => {
    const typeCls = 'type-' + (c.chunk_type||'fixed');
    const barW = maxRRF > 0 ? Math.round((c.rrf_score/maxRRF)*100) : 0;
    const src = c.metadata?.source || c.doc_id?.slice(0,12) || '—';
    return `<div class="chunk-card">
      <div class="chunk-header">
        <span style="color:#7a8499;font-size:11px;font-family:var(--mono)">#${i+1}</span>
        <span class="chunk-type-badge ${typeCls}">${c.chunk_type}</span>
        <span class="chunk-source">${src}</span>
        ${c.parent_id ? `<span style="font-size:10px;color:#7a8499">↑ parent merged</span>` : ''}
      </div>
      <div class="chunk-content">${escHtml(c.content)}</div>
      <div class="chunk-scores">
        <div class="score-item"><div class="score-label">RRF Score</div><div class="score-value score-rrf">${c.rrf_score.toFixed(5)}</div></div>
        <div class="score-item"><div class="score-label">Dense</div><div class="score-value score-dense">${c.dense_score.toFixed(4)}</div></div>
        <div class="score-item"><div class="score-label">BM25</div><div class="score-value score-sparse">${c.sparse_score.toFixed(4)}</div></div>
        ${c.dense_rank ? `<div class="score-item"><div class="score-label">Dense Rank</div><div class="score-value">${c.dense_rank}</div></div>` : ''}
        ${c.sparse_rank ? `<div class="score-item"><div class="score-label">BM25 Rank</div><div class="score-value">${c.sparse_rank}</div></div>` : ''}
      </div>
      <div class="rrf-bar" style="width:${barW}%"></div>
    </div>`;
  }).join('');
}

// ── Render Answer ───────────────────────────────────────────────
function renderAnswer(result) {
  const panel = document.getElementById('answer-panel');
  const gen = result.pipeline_stages.find(s=>s.stage==='llm_generation')?.result || {};
  panel.style.display = 'block';
  document.getElementById('answer-meta').innerHTML =
    `<span>${gen.model||'—'}</span><span>${gen.latency_ms||0}ms</span><span>${gen.context_chunks_used||0} chunks</span>`;
  document.getElementById('answer-content').textContent = result.answer;
}

function renderError(msg) {
  document.getElementById('pipeline-trace').innerHTML =
    `<div style="padding:16px;color:var(--red);font-size:13px;">⚠️ ${msg}</div>`;
}

// ── Demo Corpus ─────────────────────────────────────────────────
async function loadDemo() {
  showLoading('Loading demo corpus…', ['Ingesting 7 RAG reference documents…','Building semantic index…','Fitting BM25…']);
  try {
    const r = await api('/ingest/demo', 'POST');
    hideLoading();
    checkStatus();
    alert(`✅ Loaded ${r.loaded} documents into the index.`);
  } catch(e) {
    hideLoading();
    alert('Failed to load demo: ' + e.message);
  }
}
document.getElementById('load-demo-btn').addEventListener('click', loadDemo);
document.getElementById('load-demo-btn-trace').addEventListener('click', loadDemo);

// ── Ingest ──────────────────────────────────────────────────────
document.getElementById('ingest-btn').addEventListener('click', async () => {
  const text = document.getElementById('ingest-text').value.trim();
  if (!text) return alert('Please enter document text.');
  const source = document.getElementById('ingest-source').value || 'manual_upload';
  const strategy = document.getElementById('ingest-strategy').value;
  showLoading('Ingesting document…');
  try {
    const r = await api('/ingest','POST',{text,source,strategy});
    hideLoading();
    checkStatus();
    const el = document.getElementById('ingest-result');
    el.innerHTML = `<div class="ingest-success">
      ${Object.entries({
        'Document ID': r.doc_id?.slice(0,18)+'…',
        'Source': r.source,
        'Strategy': r.strategy,
        'Chunks Created': r.chunk_count,
        'Tokens': r.token_count,
        'Latency': r.latency_ms + 'ms',
        'Ingested At': r.ingested_at,
      }).map(([k,v])=>`<div class="ingest-stat"><span class="ingest-stat-key">${k}</span><span class="ingest-stat-val">${v}</span></div>`).join('')}
    </div>`;
  } catch(e) {
    hideLoading();
    document.getElementById('ingest-result').innerHTML = `<div style="color:var(--red);padding:12px;">Error: ${e.message}</div>`;
  }
});

// ── Router ──────────────────────────────────────────────────────
async function loadRoutes() {
  const el = document.getElementById('routes-list');
  el.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const routes = await api('/router/routes');
    const targetColors = {
      vector_db:'primary',sql_agent:'yellow',code_agent:'accent',
      legal_agent:'router',calculator:'green',conversational:'muted',
      hybrid:'primary',summarizer:'red',
    };
    el.innerHTML = routes.map(r => `<div class="route-card">
      <div class="route-card-header">
        <div class="route-card-name">${r.name}</div>
        <span class="route-target target-${r.target}">${r.target}</span>
      </div>
      <div class="route-card-desc">${r.description}</div>
      <div class="route-examples">${r.examples.slice(0,3).map(e=>`<span class="route-example">${e}</span>`).join('')}</div>
    </div>`).join('');
  } catch(e) {
    el.innerHTML = `<div style="color:var(--red);padding:12px;">${e.message}</div>`;
  }
}

document.getElementById('router-test-btn').addEventListener('click', async () => {
  const q = document.getElementById('router-query').value.trim();
  if (!q) return;
  const area = document.getElementById('router-result-area');
  area.style.display = 'block';
  area.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const r = await api('/router/test','POST',{query:q});
    const scores = Object.entries(r.all_scores).sort((a,b)=>b[1]-a[1]);
    area.innerHTML = `<div class="router-decision-card">
      <div class="router-decision-header">
        <div>
          <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px">Routed to</div>
          <span class="route-target target-${r.target}" style="font-size:14px;padding:4px 14px">${r.target}</span>
        </div>
        <div style="text-align:right">
          <div style="font-size:12px;color:var(--text-muted)">Confidence</div>
          <div style="font-size:22px;font-weight:700;font-family:var(--mono)">${(r.confidence*100).toFixed(1)}%</div>
        </div>
      </div>
      <div class="confidence-bar-wrap">
        <div class="conf-bar-label"><span>${r.matched_route}</span><span>${(r.confidence*100).toFixed(1)}%</span></div>
        <div class="conf-bar"><div class="conf-bar-fill" style="width:${Math.round(r.confidence*100)}%"></div></div>
      </div>
      <div style="margin-top:12px;font-size:11px;color:var(--text-muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px">All Scores</div>
      <div class="all-scores-grid">
        ${scores.map(([name,val])=>`<div class="score-row">
          <div class="score-row-name">${name}</div>
          <div class="score-row-val">${val.toFixed(4)}</div>
        </div>`).join('')}
      </div>
      ${r.fallback ? `<div style="margin-top:10px;font-size:12px;color:var(--yellow)">⚠️ Below threshold — fell back to HYBRID route</div>` : ''}
    </div>`;
  } catch(e) {
    area.innerHTML = `<div style="color:var(--red);padding:12px;">${e.message}</div>`;
  }
});

// ── HyDE ───────────────────────────────────────────────────────
document.getElementById('hyde-btn').addEventListener('click', async () => {
  const q = document.getElementById('hyde-query').value.trim();
  if (!q) return;
  const el = document.getElementById('hyde-result');
  el.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const r = await api('/hyde/transform','POST',{query:q});
    document.getElementById('hyde-latency').textContent = r.latency_ms + 'ms';
    if (r.success) {
      el.innerHTML = `<div style="margin-bottom:10px;font-size:12px;color:var(--text-muted)">
        Model: <b>${r.model_used}</b> · Generated in <b>${r.latency_ms}ms</b>
      </div>
      <div class="hyde-doc-output">${escHtml(r.hypothetical_doc)}</div>`;
    } else {
      el.innerHTML = `<div style="color:var(--yellow);padding:12px;">
        ⚠️ Ollama unavailable: ${r.error}<br>
        <small style="color:var(--text-muted)">Falling back to original query for retrieval.</small>
      </div>`;
    }
  } catch(e) {
    el.innerHTML = `<div style="color:var(--red);padding:12px;">${e.message}</div>`;
  }
});

// ── Config ──────────────────────────────────────────────────────
async function loadConfig() {
  const el = document.getElementById('config-grid');
  el.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const cfg = await api('/config');
    currentConfig = cfg;
    el.innerHTML = `
      <div class="config-item">
        <div class="config-label">Embed Model</div>
        <input class="form-input" id="cfg-embed_model" value="${cfg.embed_model}">
        <div class="config-desc">Sentence transformer for indexing and query embedding</div>
      </div>
      <div class="config-item">
        <div class="config-label">LLM Model (Ollama)</div>
        <input class="form-input" id="cfg-llm_model" value="${cfg.llm_model}">
        <div class="config-desc">Local model for HyDE and answer generation</div>
      </div>
      <div class="config-item">
        <div class="config-label">Top-K Results</div>
        <input class="form-input" id="cfg-top_k" type="number" min="1" max="20" value="${cfg.top_k}">
        <div class="config-desc">Number of chunks returned per search</div>
      </div>
      <div class="config-item">
        <div class="config-label">RRF K (smoothing)</div>
        <input class="form-input" id="cfg-rrf_k" type="number" min="1" max="200" value="${cfg.rrf_k}">
        <div class="config-desc">Reciprocal Rank Fusion constant (default: 60)</div>
      </div>
      <div class="config-item">
        <div class="config-label">Router Threshold</div>
        <input class="form-input" id="cfg-router_threshold" type="number" step="0.01" min="0" max="1" value="${cfg.router_threshold}">
        <div class="config-desc">Min confidence to commit to a route (vs fallback)</div>
      </div>
      <div class="config-item">
        <div class="config-label">HyDE Temperature</div>
        <input class="form-input" id="cfg-hyde_temperature" type="number" step="0.05" min="0" max="1" value="${cfg.hyde_temperature}">
        <div class="config-desc">LLM temperature for hypothetical doc generation</div>
      </div>
      <div class="config-item">
        <div class="config-label">Chunking Strategy</div>
        <select class="form-select" id="cfg-chunking_strategy">
          <option value="semantic" ${cfg.chunking_strategy==='semantic'?'selected':''}>Semantic</option>
          <option value="parent_child" ${cfg.chunking_strategy==='parent_child'?'selected':''}>Parent-Child</option>
          <option value="fixed" ${cfg.chunking_strategy==='fixed'?'selected':''}>Fixed-size</option>
        </select>
        <div class="config-desc">Default strategy for new document ingestion</div>
      </div>`;
  } catch(e) {
    el.innerHTML = `<div style="color:var(--red);padding:12px;">${e.message}</div>`;
  }
}

document.getElementById('save-config-btn').addEventListener('click', async () => {
  const keys = ['embed_model','llm_model','top_k','rrf_k','router_threshold','hyde_temperature','chunking_strategy'];
  const newCfg = {};
  keys.forEach(k => {
    const el = document.getElementById('cfg-'+k);
    if (!el) return;
    const v = el.value;
    newCfg[k] = (el.type==='number') ? parseFloat(v) : v;
  });
  const fb = document.getElementById('config-feedback');
  try {
    await api('/config','POST', newCfg);
    fb.className = 'config-feedback ok';
    fb.textContent = '✅ Configuration applied. Re-index documents to use new embed model.';
    checkStatus();
  } catch(e) {
    fb.className = 'config-feedback err';
    fb.textContent = '⚠️ ' + e.message;
  }
});

document.getElementById('reset-config-btn').addEventListener('click', () => {
  api('/config','POST',{}).then(()=>loadConfig());
});

// ── Stats ───────────────────────────────────────────────────────
async function loadStats() {
  const grid = document.getElementById('stats-grid');
  const docsList = document.getElementById('docs-list');
  try {
    const s = await api('/stats');
    const ix = s.index_stats;
    grid.innerHTML = `
      <div class="stat-card primary">
        <div class="stat-label">Total Chunks</div>
        <div class="stat-value">${ix.total_chunks}</div>
        <div class="stat-sub">Indexed vectors</div>
      </div>
      <div class="stat-card green">
        <div class="stat-label">Documents</div>
        <div class="stat-value">${s.documents_ingested}</div>
        <div class="stat-sub">Ingested docs</div>
      </div>
      <div class="stat-card yellow">
        <div class="stat-label">BM25 Index</div>
        <div class="stat-value">${ix.bm25_active ? 'ON' : 'OFF'}</div>
        <div class="stat-sub">Sparse retrieval</div>
      </div>
      <div class="stat-card router">
        <div class="stat-label">RRF K</div>
        <div class="stat-value">${ix.rrf_k}</div>
        <div class="stat-sub">Smoothing constant</div>
      </div>`;

    if (!s.documents.length) {
      docsList.innerHTML = '<div class="empty-state"><div class="empty-icon">📂</div><div class="empty-title">No documents indexed</div></div>';
      return;
    }
    docsList.innerHTML = `<table class="docs-table">
      <thead><tr><th>Source</th><th>Strategy</th><th>Chunks</th><th>Tokens</th><th>Latency</th><th>Ingested At</th></tr></thead>
      <tbody>${s.documents.map(d=>`<tr>
        <td>${d.source}</td>
        <td>${d.strategy}</td>
        <td>${d.chunk_count}</td>
        <td>${d.token_count}</td>
        <td>${d.latency_ms}ms</td>
        <td>${d.ingested_at}</td>
      </tr>`).join('')}</tbody>
    </table>`;
  } catch(e) {
    grid.innerHTML = `<div style="color:var(--red);grid-column:1/-1;padding:12px;">${e.message}</div>`;
  }
}

document.getElementById('clear-index-btn').addEventListener('click', async () => {
  if (!confirm('Clear all indexed chunks? This cannot be undone.')) return;
  await api('/stats/clear','POST');
  loadStats();
  checkStatus();
});

// ── Hardware & Models ──────────────────────────────────────────
async function loadHardware() {
  const el = document.getElementById('hardware-profile-content');
  el.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const hw = await api('/hardware');
    el.innerHTML = `
      <div class="tier-badge tier-${hw.hardware_tier}">${hw.hardware_tier.replace(/_/g, ' ')}</div>
      
      <div class="hw-section-title">CPU</div>
      <div class="hw-stat-row"><span class="hw-stat-key">Model</span><span class="hw-stat-val">${hw.cpu.brand}</span></div>
      <div class="hw-stat-row"><span class="hw-stat-key">Cores</span><span class="hw-stat-val">${hw.cpu.physical_cores}P / ${hw.cpu.logical_cores}L</span></div>
      
      <div class="hw-section-title">Memory</div>
      <div class="hw-stat-row"><span class="hw-stat-key">Total RAM</span><span class="hw-stat-val">${hw.memory.total_gb} GB</span></div>
      
      <div class="hw-section-title">Storage</div>
      <div class="hw-stat-row"><span class="hw-stat-key">Free Space</span><span class="hw-stat-val">${hw.disk.free_gb} GB</span></div>
      
      <div class="hw-section-title">Ollama</div>
      <div class="hw-stat-row"><span class="hw-stat-key">Status</span><span class="hw-stat-val">${hw.ollama_installed ? 'Installed' : 'Not Found'}</span></div>
      ${hw.ollama_installed ? `<div class="hw-stat-row"><span class="hw-stat-key">Version</span><span class="hw-stat-val">${hw.ollama_version}</span></div>` : ''}
      
      <div class="hw-section-title">Recommendations</div>
      <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">Optimal models for your hardware:</div>
      <div style="display: flex; flex-wrap: wrap; gap: 6px;">
        ${hw.recommended_llm_models.map(m => `<span class="route-example" style="color: var(--accent)">${m}</span>`).join('')}
      </div>

      ${hw.warnings.length ? `<div style="margin-top: 20px;">${hw.warnings.map(w => `<div style="color: var(--yellow); font-size: 12px; margin-bottom: 4px;">${w}</div>`).join('')}</div>` : ''}
    `;
  } catch (e) {
    el.innerHTML = `<div style="color:var(--red);padding:12px;">${e.message}</div>`;
  }
}

async function loadPopularModels() {
  const el = document.getElementById('popular-models-list');
  el.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const popular = await api('/models/popular');
    const local = (await api('/models')).models;
    
    el.innerHTML = popular.map(m => {
      const isDownloaded = local.some(l => l.includes(m.name));
      return `
        <div class="model-card">
          <div class="model-info">
            <div class="model-name">${m.name}</div>
            <div class="model-desc">${m.desc}</div>
            <div class="model-meta">~${m.size_gb} GB ${isDownloaded ? '· <span style="color:var(--green)">Downloaded</span>' : ''}</div>
          </div>
          <button class="btn ${isDownloaded ? 'btn-outline' : 'btn-primary'} btn-mini" 
                  onclick="pullModel('${m.name}')" 
                  ${isDownloaded ? 'disabled' : ''}>
            ${isDownloaded ? 'Available' : 'Pull'}
          </button>
        </div>
      `;
    }).join('');
  } catch (e) {
    el.innerHTML = `<div style="color:var(--red);padding:12px;">${e.message}</div>`;
  }
}

window.pullModel = async function(modelName) {
  showLoading(`Pulling model ${modelName}...`, ['This may take several minutes depending on your internet speed.']);
  try {
    const r = await api('/models/pull', 'POST', { model: modelName });
    hideLoading();
    alert(`✅ ${r.message}. Check Ollama logs or wait for it to appear in the list.`);
    loadPopularModels();
    checkStatus();
  } catch (e) {
    hideLoading();
    alert('Failed to pull model: ' + e.message);
  }
};

document.getElementById('pull-custom-model-btn').addEventListener('click', () => {
  const name = document.getElementById('custom-model-name').value.trim();
  if (name) pullModel(name);
});

// ── Utility ─────────────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Boot ────────────────────────────────────────────────────────
checkStatus();
setInterval(checkStatus, 30000);
