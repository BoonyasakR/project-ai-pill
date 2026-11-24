// นำเข้า createRoot: ฟังก์ชันสร้างจุดเริ่มแสดงผลของ React | input: - | output: ฟังก์ชัน createRoot
import { createRoot } from "react-dom/client";
// นำเข้าคอมโพเนนต์หลักของแอป | input: - | output: App
import App from "./App.jsx";
// นำเข้าไฟล์ CSS เพื่อให้สไตล์มีผลทั้งแอป | input: - | output: สไตล์ถูกโหลด
import "./styles.css";

// สร้างจุดเริ่มแสดงผลที่ element id="root" แล้ววาดคอมโพเนนต์ App ลงไป | input: element #root | output: หน้าเว็บถูกแสดง
createRoot(document.getElementById("root")).render(<App />);
