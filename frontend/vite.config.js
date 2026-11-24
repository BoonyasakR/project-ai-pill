// นำเข้าปลั๊กอิน React ของ Vite (แปลง JSX และรองรับ hot reload) | input: - | output: ฟังก์ชันปลั๊กอิน react
import react from "@vitejs/plugin-react";
// นำเข้า defineConfig: ตัวช่วยกำหนดค่า Vite พร้อม type hint | input: - | output: ฟังก์ชัน defineConfig
import { defineConfig } from "vite";

// ส่งออกค่ากำหนดของ Vite | input: object การตั้งค่า | output: config ที่ Vite ใช้
export default defineConfig({
  // ใช้ปลั๊กอิน React | input: - | output: รายการปลั๊กอิน
  plugins: [react()],
  // ตั้งค่าเซิร์ฟเวอร์ตอนพัฒนา
  server: {
    // ส่งคำขอที่ขึ้นต้นด้วย /api ต่อไปยัง backend ที่ http://127.0.0.1:8000 | input: คำขอ /api/... | output: คำขอถูกส่งต่อ
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
