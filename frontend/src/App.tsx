import { useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/$/, "");type Result = {
  category: string;
  anomaly_score: number;
  report: string;
  retrieved_documents: Array<{citation?: string; title?: string; source?: string; text?: string; rrf_score?: number}>;
  llm_provider?: string;
  llm_error?: string | null;
  warning?: string;
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState("bottle");
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);

  async function investigate(event: React.FormEvent) {
    event.preventDefault();
    if (!file) { setError("Choose a PNG or JPEG image first."); return; }
    setBusy(true); setError(""); setResult(null);
    const body = new FormData();
    body.append("file", file);
    body.append("category", category);
    body.append("query", query);
    try {
      const response = await fetch(`${API}/investigate`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
      setResult(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the API.");
    } finally { setBusy(false); }
  }

  return <main className="shell">
    <header><div className="eyebrow">INDUSTRIAL AI · INSPECTION WORKSPACE</div><h1>Visual Defect <span>Copilot</span></h1>
      <p>Investigate visual anomalies with computer vision and evidence-grounded technical retrieval.</p></header>
    <section className="panel">
      <form onSubmit={investigate}>
        <label>Product category<input value={category} onChange={e => setCategory(e.target.value)} placeholder="e.g. bottle" /></label>
        <label>Inspection image<input type="file" accept="image/png,image/jpeg" onChange={e => setFile(e.target.files?.[0] || null)} /></label>
        <label>Investigation question <textarea value={query} onChange={e => setQuery(e.target.value)} placeholder="e.g. What inspection procedures cover surface cracks?" rows={3} /></label>
        <button disabled={busy}>{busy ? "Investigating…" : "Run investigation"}</button>
      </form>
      {error && <p className="error">{error}</p>}
    </section>
    {result && <section className="panel output">
      <div className="metrics"><div><small>CATEGORY</small><strong>{result.category}</strong></div><div><small>ANOMALY SCORE</small><strong>{result.anomaly_score.toFixed(4)}</strong></div><div><small>REPORT ENGINE</small><strong>{result.llm_provider || "unknown"}</strong></div></div>
      {result.llm_error && <p className="notice">{result.llm_error}</p>}
      <h2>Investigation report</h2><pre>{result.report}</pre>
      <h2>Retrieved evidence</h2>
      {result.retrieved_documents?.length ? result.retrieved_documents.map((d, i) => <article className="evidence" key={d.citation || i}><b>{d.citation} · {d.title || "Untitled"}</b><small>{d.source}</small><p>{d.text}</p></article>) : <p>No relevant documents retrieved.</p>}
      <p className="notice">{result.warning}</p>
    </section>}
    <footer>Research prototype · Verify all findings against physical and process evidence.</footer>
  </main>;
}
