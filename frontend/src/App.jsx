import { useEffect, useState } from "react";

const PAGES = [
  ["upload", "อัปโหลดภาพ"],
  ["result", "ผลการตรวจ"],
  ["compare", "เปรียบเทียบโมเดล"],
  ["experiments", "ผลการทดลอง"],
];

const METRICS = [
  ["image_auroc", "Image AUROC"],
  ["pixel_auroc", "Pixel AUROC"],
  ["au_pro", "AU-PRO"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
];

const MODEL_NAMES = { patchcore: "PatchCore", padim: "PaDiM" };
const fixed = (value, digits = 3) => (typeof value === "number" ? value.toFixed(digits) : "-");

function UploadPage({ onResult }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function inspect() {
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/inspect", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "ตรวจภาพไม่สำเร็จ");
      onResult({ ...data, fileName: file.name });
    } catch (problem) {
      setError(problem.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel narrow">
      <h2>อัปโหลดภาพยาเม็ด</h2>
      <p className="muted">ใช้ภาพยาหนึ่งเม็ดต่อภาพ โมเดลนี้เรียนรู้จากยาเม็ดชนิดเดียวใน MVTec AD Pill</p>
      <label className="drop">
        {preview ? <img src={preview} alt="ภาพที่เลือก" /> : <span>เลือกไฟล์ภาพ PNG หรือ JPG</span>}
        <input
          type="file"
          accept="image/png,image/jpeg"
          onChange={(event) => setFile(event.target.files[0] || null)}
        />
      </label>
      {error && <p className="error" role="alert">{error}</p>}
      <button disabled={!file || busy} onClick={inspect}>
        {busy ? "กำลังตรวจ..." : "ตรวจภาพ"}
      </button>
    </section>
  );
}

function ResultPage({ result, goToUpload }) {
  if (!result) {
    return (
      <section className="panel narrow">
        <h2>ยังไม่มีผลการตรวจ</h2>
        <button onClick={goToUpload}>ไปอัปโหลดภาพ</button>
      </section>
    );
  }
  const abnormal = result.is_anomalous;
  return (
    <section className="panel">
      <div className="verdict-row">
        <div>
          <p className="muted">{result.fileName}</p>
          <h2 className={abnormal ? "bad" : "good"}>{abnormal ? "ผิดปกติ" : "ปกติ"}</h2>
        </div>
        <dl className="stats">
          <div><dt>Score</dt><dd>{fixed(result.score, 2)}</dd></div>
          <div><dt>Threshold</dt><dd>{fixed(result.threshold, 2)}</dd></div>
          <div><dt>เวลา</dt><dd>{fixed(result.latency_ms, 0)} ms</dd></div>
        </dl>
      </div>
      <div className="images">
        <figure><img src={result.image} alt="ภาพต้นฉบับ" /><figcaption>ภาพที่ตรวจ</figcaption></figure>
        <figure><img src={result.heatmap} alt="Heatmap จุดผิดปกติ" /><figcaption>Heatmap สีแดงคือบริเวณที่ต่างจากยาปกติมาก</figcaption></figure>
      </div>
      <p className="muted">ผลนี้มาจากงานต้นแบบ ไม่ใช่การรับรองคุณภาพยาจริง</p>
    </section>
  );
}

function ComparePage({ results }) {
  const models = Object.entries(results.main);
  if (!models.length) return <EmptyResults />;
  const defects = Object.keys(models[0][1].by_defect);
  return (
    <>
      <section className="panel">
        <h2>ผลบน test set</h2>
        <p className="muted">ตั้ง threshold จากภาพปกติใน validation แล้วประเมินบน test หลังล็อกค่า</p>
        <div className="scroll">
          <table>
            <thead>
              <tr><th>Metric</th>{models.map(([name]) => <th key={name}>{MODEL_NAMES[name]}</th>)}</tr>
            </thead>
            <tbody>
              {METRICS.map(([key, label]) => (
                <tr key={key}>
                  <td>{label}</td>
                  {models.map(([name, run]) => <td key={name}>{fixed(run.overall[key])}</td>)}
                </tr>
              ))}
              <tr>
                <td>False positive / negative</td>
                {models.map(([name, run]) => (
                  <td key={name}>{run.overall.false_positive} / {run.overall.false_negative}</td>
                ))}
              </tr>
              <tr>
                <td>เวลาต่อภาพบน T4</td>
                {models.map(([name, run]) => <td key={name}>{fixed(run.overall.latency_ms, 1)} ms</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel">
        <h2>Recall และ AU-PRO ตามตำหนิ</h2>
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>ตำหนิ</th><th>ภาพ</th>
                {models.map(([name]) => <th key={name}>{MODEL_NAMES[name]} Recall</th>)}
                {models.map(([name]) => <th key={name}>{MODEL_NAMES[name]} AU-PRO</th>)}
              </tr>
            </thead>
            <tbody>
              {defects.map((defect) => (
                <tr key={defect}>
                  <td>{defect}</td>
                  <td>{models[0][1].by_defect[defect].images}</td>
                  {models.map(([name, run]) => <td key={name}>{fixed(run.by_defect[defect].recall)}</td>)}
                  {models.map(([name, run]) => <td key={name}>{fixed(run.by_defect[defect].au_pro)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function ExperimentsPage({ results }) {
  const baseline = results.main.patchcore;
  const runs = Object.entries(results.exploratory);
  if (!baseline) return <EmptyResults />;
  return (
    <section className="panel">
      <h2>การทดลองปรับ PatchCore</h2>
      <p className="notice">
        การทดลองเหล่านี้เลือกหลังเห็นผล test จึงเป็น exploratory ไม่ใช่หลักฐานประสิทธิภาพอิสระ
      </p>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>การตั้งค่า</th><th>Image AUROC</th><th>Pixel AUROC</th><th>AU-PRO</th><th>F1</th><th>เวลา</th>
            </tr>
          </thead>
          <tbody>
            <tr className="baseline">
              <td>ผลหลัก 256, coreset 0.1</td>
              <td>{fixed(baseline.overall.image_auroc)}</td>
              <td>{fixed(baseline.overall.pixel_auroc)}</td>
              <td>{fixed(baseline.overall.au_pro)}</td>
              <td>{fixed(baseline.overall.f1)}</td>
              <td>{fixed(baseline.overall.latency_ms, 1)} ms</td>
            </tr>
            {runs.map(([name, run]) => (run.error ? (
              <tr key={name}><td>{name}</td><td colSpan="5">รันไม่สำเร็จ</td></tr>
            ) : (
              <tr key={name}>
                <td>{run.config.size}, coreset {run.config.coreset}</td>
                <td>{fixed(run.overall.image_auroc)}</td>
                <td>{fixed(run.overall.pixel_auroc)}</td>
                <td>{fixed(run.overall.au_pro)}</td>
                <td>{fixed(run.overall.f1)}</td>
                <td>{fixed(run.overall.latency_ms, 1)} ms</td>
              </tr>
            )))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EmptyResults() {
  return <section className="panel narrow"><h2>ยังโหลดผลการทดลองไม่ได้</h2></section>;
}

export default function App() {
  const [page, setPage] = useState("upload");
  const [result, setResult] = useState(null);
  const [results, setResults] = useState({ main: {}, exploratory: {} });

  useEffect(() => {
    fetch("/api/results")
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => data && setResults(data))
      .catch(() => {});
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Pill Inspection</h1>
        <nav aria-label="หน้าหลัก">
          {PAGES.map(([id, label]) => (
            <button
              key={id}
              className={page === id ? "tab active" : "tab"}
              aria-current={page === id ? "page" : undefined}
              onClick={() => setPage(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {page === "upload" && (
          <UploadPage onResult={(data) => { setResult(data); setPage("result"); }} />
        )}
        {page === "result" && <ResultPage result={result} goToUpload={() => setPage("upload")} />}
        {page === "compare" && <ComparePage results={results} />}
        {page === "experiments" && <ExperimentsPage results={results} />}
      </main>
    </div>
  );
}
