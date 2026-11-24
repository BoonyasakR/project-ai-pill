// นำเข้า useEffect (ทำงานข้างเคียงหลัง render) และ useState (เก็บสถานะ) จาก React | input: - | output: ฟังก์ชัน hook ทั้งสอง
import { useEffect, useState } from "react";

// รายการหน้าเมนู: แต่ละรายการคือ [รหัสหน้า, ข้อความบนปุ่ม] | input: - | output: PAGES = array
const PAGES = [
  // หน้าอัปโหลดภาพ
  ["upload", "อัปโหลดภาพ"],
  // หน้าผลการตรวจ
  ["result", "ผลการตรวจ"],
  // หน้าเปรียบเทียบโมเดล
  ["compare", "เปรียบเทียบโมเดล"],
  // หน้าผลการทดลอง
  ["experiments", "ผลการทดลอง"],
];

// รายการเมตริกที่แสดงในตาราง: [ชื่อ key ในข้อมูล, ป้ายที่แสดง] | input: - | output: METRICS = array
const METRICS = [
  // AUROC ระดับรูป
  ["image_auroc", "Image AUROC"],
  // AUROC ระดับพิกเซล
  ["pixel_auroc", "Pixel AUROC"],
  // AU-PRO
  ["au_pro", "AU-PRO"],
  // ความแม่นยำ
  ["precision", "Precision"],
  // สัดส่วนที่จับได้
  ["recall", "Recall"],
  // ค่าเฉลี่ยฮาร์โมนิกของ precision กับ recall
  ["f1", "F1"],
];

// แปลงรหัสโมเดลเป็นชื่อที่แสดง | input: - | output: MODEL_NAMES = object
const MODEL_NAMES = { patchcore: "PatchCore", padim: "PaDiM" };
// ข้อมูลของยาแต่ละชนิดที่ให้เลือก: ป้ายชื่อ และคำแนะนำการใช้ | input: - | output: PILL_TYPES = object
const PILL_TYPES = {
  // ยา MVTec
  mvtec: {
    // ป้ายชื่อที่แสดงในรายการเลือก
    label: "MVTec Pill (ยาเม็ดรีสีขาว)",
    // คำแนะนำภาพที่เหมาะกับชนิดนี้
    hint: "ใช้ภาพยาหนึ่งเม็ดหรือหลายเม็ดที่วางไม่ทับกันบนพื้นสีเข้ม",
  },
  // ยา pillQC
  pillqc: {
    // ป้ายชื่อที่แสดงในรายการเลือก
    label: "pillQC (ยาเม็ดกลมสีชมพู)",
    // คำแนะนำภาพ พร้อมข้อจำกัดว่ายังจับตำหนิได้น้อย
    hint: "ใช้ภาพยาหนึ่งเม็ดบนพื้นสีเขียวอมฟ้าแบบชุดข้อมูล pillQC หรือหลายเม็ดที่วางไม่ทับกันบนพื้นสีเข้ม ยาชนิดนี้ยังจับตำหนิได้น้อย: ราว 1 ใน 4 เม็ดในภาพเดี่ยว และแทบไม่ได้เลยในภาพหลายเม็ด",
  },
};
// ฟังก์ชันจัดรูปตัวเลขทศนิยม | input: value = ค่าที่จะแสดง, digits = จำนวนทศนิยม (ค่าเริ่มต้น 3) | output: ข้อความเลขทศนิยม หรือ "-" ถ้า value ไม่ใช่ตัวเลข
const fixed = (value, digits = 3) => (typeof value === "number" ? value.toFixed(digits) : "-");

// คอมโพเนนต์หน้าอัปโหลดภาพ | input: props.onResult = ฟังก์ชันที่เรียกเมื่อได้ผลตรวจ | output: องค์ประกอบ UI ของหน้าอัปโหลด
function UploadPage({ onResult }) {
  // สถานะไฟล์ที่เลือก (เริ่มต้นไม่มี) | input: - | output: file, setFile
  const [file, setFile] = useState(null);
  // สถานะชนิดยาที่เลือก (เริ่มต้น mvtec) | input: - | output: pillType, setPillType
  const [pillType, setPillType] = useState("mvtec");
  // สถานะ URL ตัวอย่างรูปที่เลือก | input: - | output: preview, setPreview
  const [preview, setPreview] = useState(null);
  // สถานะกำลังตรวจอยู่หรือไม่ | input: - | output: busy, setBusy
  const [busy, setBusy] = useState(false);
  // สถานะข้อความผิดพลาด | input: - | output: error, setError
  const [error, setError] = useState("");

  // ทำงานเมื่อไฟล์เปลี่ยน เพื่อสร้าง URL ตัวอย่างรูป
  useEffect(() => {
    // ถ้ายังไม่มีไฟล์ ไม่ต้องทำอะไรและไม่มีฟังก์ชันล้าง | input: file | output: undefined
    if (!file) return undefined;
    // สร้าง URL ชั่วคราวของไฟล์ในเบราว์เซอร์ | input: file | output: url = ข้อความ
    const url = URL.createObjectURL(file);
    // เก็บ URL ไว้แสดงตัวอย่าง | input: url | output: preview ถูกอัปเดต
    setPreview(url);
    // คืนฟังก์ชันล้าง: คืนหน่วยความจำของ URL เมื่อไฟล์เปลี่ยนหรือคอมโพเนนต์ถูกถอด | input: url | output: ฟังก์ชัน
    return () => URL.revokeObjectURL(url);
    // ให้ effect นี้ทำงานใหม่เมื่อ file เปลี่ยน
  }, [file]);

  // ฟังก์ชันส่งภาพไปตรวจที่ API | input: ไฟล์และชนิดยาที่เลือกไว้ | output: เรียก onResult พร้อมผลตรวจ หรือตั้งข้อความผิดพลาด
  async function inspect() {
    // ตั้งสถานะกำลังตรวจ | input: true | output: busy = true
    setBusy(true);
    // ล้างข้อความผิดพลาดเดิม | input: "" | output: error = ""
    setError("");
    // เริ่มบล็อกที่อาจเกิดข้อผิดพลาด
    try {
      // สร้างข้อมูลฟอร์มสำหรับส่งไฟล์ | input: - | output: body = FormData
      const body = new FormData();
      // ใส่ไฟล์รูปในฟอร์ม | input: file | output: body มี field "file"
      body.append("file", file);
      // ใส่ชนิดยาในฟอร์ม | input: pillType | output: body มี field "pill_type"
      body.append("pill_type", pillType);
      // ส่ง POST ไปที่ /api/inspect | input: body | output: response
      const response = await fetch("/api/inspect", { method: "POST", body });
      // อ่านเนื้อหาตอบกลับเป็น JSON | input: response | output: data = object
      const data = await response.json();
      // ถ้าเซิร์ฟเวอร์ตอบ error ให้โยนข้อผิดพลาดพร้อมข้อความจากเซิร์ฟเวอร์ (หรือข้อความสำรอง) | input: response.ok, data.detail | output: throw Error
      if (!response.ok) throw new Error(data.detail || "ตรวจภาพไม่สำเร็จ");
      // ส่งผลตรวจพร้อมชื่อไฟล์และชนิดยาให้หน้าแม่ | input: data, file.name, pillType | output: เรียก onResult
      onResult({ ...data, fileName: file.name, pillType });
    // ดักข้อผิดพลาดทั้งหมด (เครือข่ายหรือจากเซิร์ฟเวอร์)
    } catch (problem) {
      // แสดงข้อความผิดพลาด | input: problem.message | output: error ถูกอัปเดต
      setError(problem.message);
    // ส่วนที่ทำเสมอไม่ว่าสำเร็จหรือไม่
    } finally {
      // ปิดสถานะกำลังตรวจ | input: false | output: busy = false
      setBusy(false);
    }
  }

  // คืนองค์ประกอบ UI ของหน้านี้
  return (
    // กล่องหน้าอัปโหลดแบบแคบ
    <section className="panel narrow">
      {/* หัวข้อหน้า */}
      <h2>อัปโหลดภาพยาเม็ด</h2>
      {/* ป้ายกำกับของรายการเลือกชนิดยา */}
      <label className="field">
        ชนิดยา
        {/* รายการเลือกชนิดยา: ค่าที่เลือกคือ pillType และเมื่อเปลี่ยนจะอัปเดตสถานะ */}
        <select value={pillType} onChange={(event) => setPillType(event.target.value)}>
          {/* สร้างตัวเลือกจากทุกชนิดยาใน PILL_TYPES | input: PILL_TYPES | output: รายการ <option> */}
          {Object.entries(PILL_TYPES).map(([id, type]) => <option key={id} value={id}>{type.label}</option>)}
        </select>
      </label>
      {/* ข้อความอธิบาย: คำแนะนำของชนิดยาที่เลือก ตามด้วยหมายเหตุว่าระบบไม่ระบุชนิดให้อัตโนมัติ */}
      <p className="muted">
        {PILL_TYPES[pillType].hint} แต่ละชนิดใช้ memory bank และ threshold ของตัวเอง ระบบไม่ระบุชนิดยาให้อัตโนมัติ
      </p>
      {/* พื้นที่เลือกไฟล์ (กดเพื่อเลือก) */}
      <label className="drop">
        {/* ถ้ามีตัวอย่างรูปให้แสดงรูป ไม่มีก็แสดงข้อความชวนเลือกไฟล์ | input: preview | output: <img> หรือ <span> */}
        {preview ? <img src={preview} alt="ภาพที่เลือก" /> : <span>เลือกไฟล์ภาพ PNG, JPG หรือ WebP</span>}
        {/* ช่องเลือกไฟล์ (ซ่อนด้วย CSS) */}
        <input
          // ชนิดของ input คือไฟล์
          type="file"
          // รับเฉพาะ PNG, JPEG, WebP
          accept="image/png,image/jpeg,image/webp"
          // เมื่อเลือกไฟล์ ให้เก็บไฟล์แรก (หรือ null ถ้าไม่มี) | input: event.target.files | output: file ถูกอัปเดต
          onChange={(event) => setFile(event.target.files[0] || null)}
        />
      </label>
      {/* แสดงข้อความผิดพลาดเมื่อมี (role="alert" ให้โปรแกรมอ่านหน้าจอประกาศทันที) */}
      {error && <p className="error" role="alert">{error}</p>}
      {/* ปุ่มตรวจภาพ: ปิดเมื่อยังไม่มีไฟล์หรือกำลังตรวจ, กดแล้วเรียก inspect */}
      <button disabled={!file || busy} onClick={inspect}>
        {/* ข้อความบนปุ่มเปลี่ยนตามสถานะ | input: busy | output: ข้อความ */}
        {busy ? "กำลังตรวจ..." : "ตรวจภาพ"}
      </button>
    </section>
  );
}

// คอมโพเนนต์แสดงข้อความเตือน | input: props.warnings = array ของข้อความเตือน (หรือไม่มี) | output: <p> หนึ่งอันต่อข้อความ (หรือ undefined ถ้าไม่มี)
function Warnings({ warnings }) {
  // ถ้ามี warnings ให้แปลงแต่ละข้อความเป็นย่อหน้า (role="status" ให้โปรแกรมอ่านหน้าจอรับรู้) | input: warnings | output: array ขององค์ประกอบ
  return warnings?.map((text) => <p className="notice" role="status" key={text}>{text}</p>);
}

// คอมโพเนนต์แสดงผลตรวจแบบหลายเม็ด | input: props.result = ผลตรวจที่มี pills | output: องค์ประกอบ UI ของผลตรวจหลายเม็ด
function SceneResult({ result }) {
  // นับจำนวนเม็ดที่ผิดปกติ | input: result.pills | output: bad = จำนวนเต็ม
  const bad = result.pills.filter((pill) => pill.is_anomalous).length;
  // คืนองค์ประกอบ UI
  return (
    // กล่องผลลัพธ์
    <section className="panel">
      {/* แถวสรุปผลและตัวเลขสถิติ */}
      <div className="verdict-row">
        {/* ฝั่งซ้าย: ชื่อไฟล์และคำตัดสิน */}
        <div>
          {/* ชื่อไฟล์และชื่อชนิดยา (ใช้ ?. เผื่อไม่พบชนิด) */}
          <p className="muted">{result.fileName} · {PILL_TYPES[result.pillType]?.label}</p>
          {/* คำตัดสิน: สีแดงถ้ามีเม็ดผิดปกติ พร้อมจำนวน, สีเขียวถ้าปกติทุกเม็ด | input: bad | output: ข้อความ */}
          <h2 className={bad ? "bad" : "good"}>{bad ? `ผิดปกติ ${bad} เม็ด` : "ปกติทุกเม็ด"}</h2>
        </div>
        {/* รายการสถิติ */}
        <dl className="stats">
          {/* จำนวนเม็ดที่พบ */}
          <div><dt>พบยา</dt><dd>{result.pills.length} เม็ด</dd></div>
          {/* threshold ทศนิยม 2 ตำแหน่ง */}
          <div><dt>Threshold</dt><dd>{fixed(result.threshold, 2)}</dd></div>
          {/* เวลาที่ใช้ แปลงจากมิลลิวินาทีเป็นวินาที ทศนิยม 1 ตำแหน่ง */}
          <div><dt>เวลา</dt><dd>{fixed(result.latency_ms / 1000, 1)} วินาที</dd></div>
        </dl>
      </div>
      {/* แสดงคำเตือน (ถ้ามี) | input: result.warnings */}
      <Warnings warnings={result.warnings} />
      {/* ภาพรวมที่วาดกรอบและหมายเลข */}
      <figure className="scene">
        {/* ภาพที่ตรวจ (data URL จากเซิร์ฟเวอร์) */}
        <img src={result.image} alt="ภาพที่ตรวจ พร้อมหมายเลขและกรอบของยาแต่ละเม็ด" />
        {/* คำอธิบายสีของกรอบ */}
        <figcaption>กรอบเขียวคือปกติ กรอบแดงคือผิดปกติ</figcaption>
      </figure>
      {/* กริดแสดง heatmap ของแต่ละเม็ด */}
      <div className="pills">
        {/* วนทุกเม็ด สร้าง figure หนึ่งอันต่อเม็ด | input: result.pills | output: รายการ <figure> */}
        {result.pills.map((pill) => (
          // รูปและคำอธิบายของเม็ดนี้ (key คือหมายเลขเม็ด)
          <figure key={pill.number}>
            {/* heatmap ของเม็ดนี้ */}
            <img src={pill.heatmap} alt={`Heatmap ยาเม็ดที่ ${pill.number}`} />
            {/* คำอธิบาย: หมายเลข ผล คะแนน และหมายเหตุสีแปลก */}
            <figcaption>
              {/* ข้อความ "เม็ดที่ N:" */}
              เม็ดที่ {pill.number}:{" "}
              {/* คำว่า ผิดปกติ/ปกติ มีสีตามผล | input: pill.is_anomalous | output: <span> */}
              <span className={pill.is_anomalous ? "bad" : "good"}>{pill.is_anomalous ? "ผิดปกติ" : "ปกติ"}</span>
              {/* ช่องว่างคั่น แล้วตามด้วยคะแนนในวงเล็บ และหมายเหตุถ้ามีสีที่ยาปกติไม่มี */}
              {" "}({fixed(pill.score, 2)}){pill.odd_colour && " มีสีที่ยาปกติไม่มี"}
            </figcaption>
          </figure>
        ))}
      </div>
      {/* ข้อความเตือนว่าเป็นงานต้นแบบ */}
      <p className="muted">ผลนี้มาจากงานต้นแบบ ไม่ใช่การรับรองคุณภาพยาจริง</p>
    </section>
  );
}

// คอมโพเนนต์หน้าผลการตรวจ | input: props.result = ผลตรวจล่าสุด (หรือ null), props.goToUpload = ฟังก์ชันกลับไปหน้าอัปโหลด | output: องค์ประกอบ UI ของผลตรวจ
function ResultPage({ result, goToUpload }) {
  // ถ้ายังไม่มีผลตรวจ
  if (!result) {
    // แสดงข้อความและปุ่มกลับไปอัปโหลด
    return (
      // กล่องแบบแคบ
      <section className="panel narrow">
        {/* หัวข้อบอกว่ายังไม่มีผล */}
        <h2>ยังไม่มีผลการตรวจ</h2>
        {/* ปุ่มไปหน้าอัปโหลด */}
        <button onClick={goToUpload}>ไปอัปโหลดภาพ</button>
      </section>
    );
  }
  // ถ้าผลมีรายการ pills (ภาพหลายเม็ด) ให้ใช้คอมโพเนนต์ผลหลายเม็ด | input: result | output: <SceneResult>
  if (result.pills) return <SceneResult result={result} />;
  // ค่าบอกว่าผิดปกติหรือไม่ | input: result.is_anomalous | output: abnormal = boolean
  const abnormal = result.is_anomalous;
  // คืนองค์ประกอบ UI ของผลภาพเม็ดเดียว
  return (
    // กล่องผลลัพธ์
    <section className="panel">
      {/* แถวสรุปผลและตัวเลขสถิติ */}
      <div className="verdict-row">
        {/* ฝั่งซ้าย: ชื่อไฟล์และคำตัดสิน */}
        <div>
          {/* ชื่อไฟล์และชื่อชนิดยา */}
          <p className="muted">{result.fileName} · {PILL_TYPES[result.pillType]?.label}</p>
          {/* คำตัดสิน: ผิดปกติ (สีแดง) หรือ ปกติ (สีเขียว) | input: abnormal | output: ข้อความ */}
          <h2 className={abnormal ? "bad" : "good"}>{abnormal ? "ผิดปกติ" : "ปกติ"}</h2>
        </div>
        {/* รายการสถิติ */}
        <dl className="stats">
          {/* คะแนนความผิดปกติ ทศนิยม 2 ตำแหน่ง */}
          <div><dt>Score</dt><dd>{fixed(result.score, 2)}</dd></div>
          {/* threshold ทศนิยม 2 ตำแหน่ง */}
          <div><dt>Threshold</dt><dd>{fixed(result.threshold, 2)}</dd></div>
          {/* เวลาที่ใช้เป็นมิลลิวินาที (ไม่มีทศนิยม) */}
          <div><dt>เวลา</dt><dd>{fixed(result.latency_ms, 0)} ms</dd></div>
        </dl>
      </div>
      {/* แสดงคำเตือน (ถ้ามี) | input: result.warnings */}
      <Warnings warnings={result.warnings} />
      {/* ถ้าพบสีแปลก แสดงข้อความว่าแจ้งผิดปกติเพราะสี ไม่ว่าคะแนนจะเกินเกณฑ์หรือไม่ */}
      {result.odd_colour && <p className="notice" role="status">พบสีที่ยาปกติไม่มี จึงแจ้งว่าผิดปกติ ไม่ว่า Score จะเกิน Threshold หรือไม่</p>}
      {/* กริดรูปสองรูปเคียงกัน */}
      <div className="images">
        {/* ภาพต้นฉบับพร้อมคำอธิบาย */}
        <figure><img src={result.image} alt="ภาพต้นฉบับ" /><figcaption>ภาพที่ตรวจ</figcaption></figure>
        {/* heatmap พร้อมคำอธิบายว่าสีแดงคือบริเวณที่ต่างจากยาปกติมาก */}
        <figure><img src={result.heatmap} alt="Heatmap จุดผิดปกติ" /><figcaption>Heatmap สีแดงคือบริเวณที่ต่างจากยาปกติมาก</figcaption></figure>
      </div>
      {/* ข้อความเตือนว่าเป็นงานต้นแบบ */}
      <p className="muted">ผลนี้มาจากงานต้นแบบ ไม่ใช่การรับรองคุณภาพยาจริง</p>
    </section>
  );
}

// คอมโพเนนต์หน้าเปรียบเทียบโมเดล | input: props.results = ผลการทดลองจาก /api/results (มี main) | output: องค์ประกอบ UI ตารางเปรียบเทียบ
function ComparePage({ results }) {
  // แปลงผลหลักเป็น array ของ [ชื่อโมเดล, ผล] | input: results.main | output: models = array
  const models = Object.entries(results.main);
  // ถ้าไม่มีผลเลย แสดงหน้าว่าง | input: models | output: <EmptyResults>
  if (!models.length) return <EmptyResults />;
  // รายชื่อชนิดความบกพร่อง (ใช้ของโมเดลแรก) | input: models[0][1].by_defect | output: defects = array ของข้อความ
  const defects = Object.keys(models[0][1].by_defect);
  // คืนองค์ประกอบ UI (fragment ครอบสองกล่อง)
  return (
    <>
      {/* กล่องที่ 1: ผลรวมบนชุดทดสอบ */}
      <section className="panel">
        {/* หัวข้อ */}
        <h2>ผลบน test set</h2>
        {/* คำอธิบายวิธีตั้ง threshold */}
        <p className="muted">ตั้ง threshold จากภาพปกติใน validation แล้วประเมินบน test หลังล็อกค่า</p>
        {/* ห่อตารางให้เลื่อนแนวนอนได้ */}
        <div className="scroll">
          {/* ตารางเปรียบเทียบเมตริก */}
          <table>
            {/* ส่วนหัวตาราง */}
            <thead>
              {/* แถวหัว: คอลัมน์ Metric แล้วตามด้วยชื่อโมเดลแต่ละตัว | input: models | output: <th> หลายอัน */}
              <tr><th>Metric</th>{models.map(([name]) => <th key={name}>{MODEL_NAMES[name]}</th>)}</tr>
            </thead>
            {/* ส่วนเนื้อหาตาราง */}
            <tbody>
              {/* หนึ่งแถวต่อเมตริก | input: METRICS | output: รายการ <tr> */}
              {METRICS.map(([key, label]) => (
                // แถวของเมตริกนี้
                <tr key={key}>
                  {/* ชื่อเมตริก */}
                  <td>{label}</td>
                  {/* ค่าของเมตริกนี้ของแต่ละโมเดล ทศนิยม 3 ตำแหน่ง | input: run.overall[key] | output: <td> หลายอัน */}
                  {models.map(([name, run]) => <td key={name}>{fixed(run.overall[key])}</td>)}
                </tr>
              ))}
              {/* แถวจำนวน false positive / false negative */}
              <tr>
                {/* ชื่อแถว */}
                <td>False positive / negative</td>
                {/* ค่าของแต่ละโมเดล | input: models | output: <td> หลายอัน */}
                {models.map(([name, run]) => (
                  // เซลล์แสดง "FP / FN"
                  <td key={name}>{run.overall.false_positive} / {run.overall.false_negative}</td>
                ))}
              </tr>
              {/* แถวเวลาต่อภาพ */}
              <tr>
                {/* ชื่อแถว */}
                <td>เวลาต่อภาพบน T4</td>
                {/* เวลาของแต่ละโมเดลเป็น ms ทศนิยม 1 ตำแหน่ง | input: run.overall.latency_ms | output: <td> หลายอัน */}
                {models.map(([name, run]) => <td key={name}>{fixed(run.overall.latency_ms, 1)} ms</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      {/* กล่องที่ 2: ผลแยกตามตำหนิ */}
      <section className="panel">
        {/* หัวข้อ */}
        <h2>Recall และ AU-PRO ตามตำหนิ</h2>
        {/* ห่อตารางให้เลื่อนแนวนอนได้ */}
        <div className="scroll">
          {/* ตารางแยกตามตำหนิ */}
          <table>
            {/* ส่วนหัวตาราง */}
            <thead>
              {/* แถวหัว */}
              <tr>
                {/* คอลัมน์ตำหนิและจำนวนภาพ */}
                <th>ตำหนิ</th><th>ภาพ</th>
                {/* หัวคอลัมน์ Recall ของแต่ละโมเดล | input: models | output: <th> หลายอัน */}
                {models.map(([name]) => <th key={name}>{MODEL_NAMES[name]} Recall</th>)}
                {/* หัวคอลัมน์ AU-PRO ของแต่ละโมเดล | input: models | output: <th> หลายอัน */}
                {models.map(([name]) => <th key={name}>{MODEL_NAMES[name]} AU-PRO</th>)}
              </tr>
            </thead>
            {/* ส่วนเนื้อหาตาราง */}
            <tbody>
              {/* หนึ่งแถวต่อชนิดตำหนิ | input: defects | output: รายการ <tr> */}
              {defects.map((defect) => (
                // แถวของตำหนินี้
                <tr key={defect}>
                  {/* ชื่อตำหนิ */}
                  <td>{defect}</td>
                  {/* จำนวนภาพของตำหนินี้ (ใช้ของโมเดลแรก) */}
                  <td>{models[0][1].by_defect[defect].images}</td>
                  {/* Recall ของแต่ละโมเดล | input: run.by_defect[defect].recall | output: <td> หลายอัน */}
                  {models.map(([name, run]) => <td key={name}>{fixed(run.by_defect[defect].recall)}</td>)}
                  {/* AU-PRO ของแต่ละโมเดล | input: run.by_defect[defect].au_pro | output: <td> หลายอัน */}
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

// คอมโพเนนต์หน้าผลการทดลองปรับ PatchCore | input: props.results = ผลจาก /api/results (มี main และ exploratory) | output: องค์ประกอบ UI ตารางผลการทดลอง
function ExperimentsPage({ results }) {
  // ผลหลักของ PatchCore ใช้เป็นแถวฐาน | input: results.main.patchcore | output: baseline = object หรือ undefined
  const baseline = results.main.patchcore;
  // รายการการทดลองเสริมเป็น array ของ [ชื่อ, ผล] | input: results.exploratory | output: runs = array
  const runs = Object.entries(results.exploratory);
  // ถ้าไม่มีผลหลัก แสดงหน้าว่าง | input: baseline | output: <EmptyResults>
  if (!baseline) return <EmptyResults />;
  // คืนองค์ประกอบ UI
  return (
    // กล่องผลลัพธ์
    <section className="panel">
      {/* หัวข้อ */}
      <h2>การทดลองปรับ PatchCore</h2>
      {/* ข้อความเตือนว่าเป็นการทดลองเชิงสำรวจ */}
      <p className="notice">
        การทดลองเหล่านี้เลือกหลังเห็นผล test จึงเป็น exploratory ไม่ใช่หลักฐานประสิทธิภาพอิสระ
      </p>
      {/* คำอธิบายแถวที่มี "ตรวจสี" และแถวที่เว็บใช้ */}
      <p className="muted">
        แถวที่มี "ตรวจสี" เทรนบน Apple M3 และแจ้งผิดปกติเมื่อ PatchCore เกิน threshold หรือพบสีที่ยาปกติไม่มี
        เว็บใช้แถว 512 + ตรวจสี
      </p>
      {/* ห่อตารางให้เลื่อนแนวนอนได้ */}
      <div className="scroll">
        {/* ตารางผลการทดลอง */}
        <table>
          {/* ส่วนหัวตาราง */}
          <thead>
            {/* แถวหัว */}
            <tr>
              {/* ชื่อคอลัมน์ทั้งหก */}
              <th>การตั้งค่า</th><th>Image AUROC</th><th>Pixel AUROC</th><th>AU-PRO</th><th>F1</th><th>เวลา</th>
            </tr>
          </thead>
          {/* ส่วนเนื้อหาตาราง */}
          <tbody>
            {/* แถวผลหลัก (เน้นด้วยคลาส baseline) */}
            <tr className="baseline">
              {/* คำอธิบายการตั้งค่า */}
              <td>ผลหลัก 256, coreset 0.1</td>
              {/* Image AUROC */}
              <td>{fixed(baseline.overall.image_auroc)}</td>
              {/* Pixel AUROC */}
              <td>{fixed(baseline.overall.pixel_auroc)}</td>
              {/* AU-PRO */}
              <td>{fixed(baseline.overall.au_pro)}</td>
              {/* F1 */}
              <td>{fixed(baseline.overall.f1)}</td>
              {/* เวลาเป็น ms ทศนิยม 1 ตำแหน่ง */}
              <td>{fixed(baseline.overall.latency_ms, 1)} ms</td>
            </tr>
            {/* หนึ่งแถวต่อการทดลองเสริม: ถ้าล้มเหลวแสดงข้อความ ไม่งั้นแสดงตัวเลข | input: runs | output: รายการ <tr> */}
            {runs.map(([name, run]) => (run.error ? (
              // แถวของการทดลองที่รันไม่สำเร็จ (ผสานคอลัมน์ 5 ช่อง)
              <tr key={name}><td>{name}</td><td colSpan="5">รันไม่สำเร็จ</td></tr>
            ) : (
              // แถวของการทดลองที่สำเร็จ
              <tr key={name}>
                {/* การตั้งค่า: ขนาดรูป, coreset และคำว่า "+ ตรวจสี" ถ้าเปิดการตรวจสี */}
                <td>{run.config.size}, coreset {run.config.coreset}{run.config.colour_check && " + ตรวจสี"}</td>
                {/* Image AUROC */}
                <td>{fixed(run.overall.image_auroc)}</td>
                {/* Pixel AUROC */}
                <td>{fixed(run.overall.pixel_auroc)}</td>
                {/* AU-PRO */}
                <td>{fixed(run.overall.au_pro)}</td>
                {/* F1 */}
                <td>{fixed(run.overall.f1)}</td>
                {/* เวลาเป็น ms ถ้ามีค่า ไม่งั้นแสดง "-" */}
                <td>{run.overall.latency_ms ? `${fixed(run.overall.latency_ms, 1)} ms` : "-"}</td>
              </tr>
            )))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// คอมโพเนนต์แสดงเมื่อโหลดผลการทดลองไม่ได้ | input: - | output: กล่องข้อความแจ้งว่ายังโหลดผลไม่ได้
function EmptyResults() {
  // คืนกล่องข้อความ
  return <section className="panel narrow"><h2>ยังโหลดผลการทดลองไม่ได้</h2></section>;
}

// คอมโพเนนต์หลักของแอป | input: - | output: โครงหน้าเว็บ (หัวเมนู และเนื้อหาตามหน้าที่เลือก)
export default function App() {
  // สถานะหน้าที่เปิดอยู่ (เริ่มต้นหน้าอัปโหลด) | input: - | output: page, setPage
  const [page, setPage] = useState("upload");
  // สถานะผลตรวจล่าสุด (เริ่มต้นไม่มี) | input: - | output: result, setResult
  const [result, setResult] = useState(null);
  // สถานะผลการทดลอง (เริ่มต้นว่าง) | input: - | output: results, setResults
  const [results, setResults] = useState({ main: {}, exploratory: {} });

  // ทำงานครั้งเดียวตอนเปิดหน้า เพื่อโหลดผลการทดลอง
  useEffect(() => {
    // ขอข้อมูลจาก /api/results
    fetch("/api/results")
      // ถ้าตอบสำเร็จแปลงเป็น JSON ไม่งั้นได้ null | input: response | output: data หรือ null
      .then((response) => (response.ok ? response.json() : null))
      // ถ้าได้ข้อมูลให้เก็บลงสถานะ | input: data | output: results ถูกอัปเดต
      .then((data) => data && setResults(data))
      // ถ้าเกิดข้อผิดพลาดให้เงียบไว้ (ใช้ค่าว่างต่อไป)
      .catch(() => {});
    // [] = ทำงานครั้งเดียวตอนเริ่ม
  }, []);

  // คืนโครงหน้าเว็บ
  return (
    // กล่องรวมทั้งแอป
    <div className="app">
      {/* ส่วนหัวของหน้า */}
      <header>
        {/* ชื่อแอป */}
        <h1>Pill Inspection</h1>
        {/* แถบเมนูหลัก (aria-label ให้โปรแกรมอ่านหน้าจอรู้ชื่อ) */}
        <nav aria-label="หน้าหลัก">
          {/* สร้างปุ่มหนึ่งปุ่มต่อหนึ่งหน้า | input: PAGES | output: รายการ <button> */}
          {PAGES.map(([id, label]) => (
            // ปุ่มเมนูของหน้านี้
            <button
              // กุญแจของรายการใน React
              key={id}
              // ปุ่มของหน้าที่เปิดอยู่ใช้คลาส active
              className={page === id ? "tab active" : "tab"}
              // บอกโปรแกรมอ่านหน้าจอว่าเป็นหน้าปัจจุบัน
              aria-current={page === id ? "page" : undefined}
              // เมื่อกดให้สลับไปหน้านี้ | input: id | output: page ถูกอัปเดต
              onClick={() => setPage(id)}
            >
              {/* ข้อความบนปุ่ม */}
              {label}
            </button>
          ))}
        </nav>
      </header>
      {/* ส่วนเนื้อหาหลัก */}
      <main>
        {/* หน้าอัปโหลด: เมื่อได้ผลให้เก็บผลแล้วสลับไปหน้าผลการตรวจ */}
        {page === "upload" && (
          <UploadPage onResult={(data) => { setResult(data); setPage("result"); }} />
        )}
        {/* หน้าผลการตรวจ: ส่งผลล่าสุดและฟังก์ชันกลับไปหน้าอัปโหลด */}
        {page === "result" && <ResultPage result={result} goToUpload={() => setPage("upload")} />}
        {/* หน้าเปรียบเทียบโมเดล */}
        {page === "compare" && <ComparePage results={results} />}
        {/* หน้าผลการทดลอง */}
        {page === "experiments" && <ExperimentsPage results={results} />}
      </main>
    </div>
  );
}
