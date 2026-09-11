# 🚀 SIH 2026: WINNING BLUEPRINT — SwarVed AI (Aegis Call Guard)

> **FOR INTERNAL TEAM ALIGNMENT | READ END-TO-END BEFORE THE SPRINT**

---

### 📌 1. THE PROBLEM STATEMENT DETAILS
- **PS Sponsoring Body**: Ministry of Home Affairs (MHA) / Indian Cyber Crime Coordination Centre (I4C) / MeitY.
- **PS Title**: **AI-Powered Real-Time Detection & Prevention of Voice Cloning Impersonation Attacks & Digital Arrest Scams**.
- **Ground Reality in India**: Cybercriminals are stealing hundreds of crores daily from Indian families using two lethal methods:
  1. **AI Voice Cloning**: Creating 3-second audio clones of relatives: *"Papa, urgent ₹50,000 sent karo, accident ho gaya hai."*
  2. **Digital Arrest Video Calls**: Fake WhatsApp video calls claiming to be from CBI/NCB/Crime Branch, placing victims under psychological "arrest" for hours while draining their bank accounts.

---

### 💡 2. OUR SOLUTION: SwarVed AI (Aegis Call Guard)
**SwarVed AI** is an active, real-time on-device call security guard operating at the incoming audio layer. 
- **100% Passive & Zero-Effort**: Works automatically on incoming unknown calls / WhatsApp video calls (just like Truecaller).
- **1.5-Second Active Interception**: Detects synthetic AI voice signatures in background audio within the first 3-second audio window.
- **Visual Emergency Overlay**: Flashes a red screen banner over the phone:  
  🚨 `CRITICAL ALERT: AI SYNTHETIC VOICE DETECTED! 98.4% Probability of Impersonation Scam. DO NOT TRANSFER MONEY.`
- **1-Tap Transaction Lock & Police Reporting**: Freezes UPI payment apps for 30 minutes (preventing panic money transfers) and auto-dispatches the scammer's audio fingerprint + IP to the **National Cyber Crime Helpline (1930 I4C Portal)**.

---

### 🛡️ 3. WHY OUR SOLUTION OBLITERATES 99% OF COMPETING TEAMS

| Feature | 99% Of Competing Teams | SwarVed AI (Our GOATed Solution) |
| :--- | :--- | :--- |
| **Form Factor** | Web page `.mp3` file uploader | **Native On-Device Active Call Guard** |
| **Inference Latency** | 3–8 Seconds (Cloud server lag) | **3s Detection Window (Local C++ ONNX, On-Device — no cloud cost)** |
| **Internet Need** | Requires fast Wi-Fi/4G | **100% Offline Capable (Zero cloud costs)** |
| **Voice Sample Need** | Requires pre-recorded family audio | **Zero-Shot Vocoder Phase-Glitch Detection** |
| **Actionability** | Text notification on screen | **Active Red Overlay + 1-Tap UPI Freeze + 1930 Dispatch** |

---

### 🏗️ 4. SYSTEM ARCHITECTURE & TECH STACK

- **Layer 1: Audio Capture**: Android `AudioRecord` / Accessibility Audio Loopback / MediaProjection API.
- **Layer 2: On-Device ML Engine**: Quantized Int8 Conformer / QuartzNet Model (<45ms execution via C++ ONNX Runtime Mobile). Detects Neural Vocoder Phase Glitches & Glottal Pulse Anomalies (ElevenLabs, Tacotron, Bark artifacts).
- **Layer 3: Verification Pipeline**: Stage 1 Acoustic AI Score + Stage 2 Call Metadata Risk Score.
- **Layer 4: UI & Enforcement**: Android `SYSTEM_ALERT_WINDOW` Red Warning Banner + 1-Tap UPI App Freeze + 1930 National Cyber Crime Reporting API payload generator.

**Tech Stack**: Kotlin / C++ JNI / Flutter + ONNX Runtime Mobile + Python FastAPI + PostgreSQL + WebSockets.

---

### 👥 5. TEAM ROLE ALLOCATION & MODULE BREAKDOWN

1. **Member 1 (AI/ML Lead)**:
   - Train & quantize the Audio Deepfake / Neural Vocoder Glitch Detection model in PyTorch $\rightarrow$ Export to ONNX (Int8).
2. **Member 2 (Android / C++ Systems Lead)**:
   - Build Android native JNI bridge (`onnxruntime-android`), live audio loopback buffer (`AudioRecord`), and System Alert Window overlay.
3. **Member 3 (Full-Stack & DPI Integrations Lead)**:
   - Build FastAPI backend microservice, real-time WebSockets, 1930 I4C Cyber Portal complaint API payload generator, and UPI lock simulation.
4. **Member 4 (UI/UX & Pitch/Presenter Lead)**:
   - Build high-fidelity dark-mode mobile dashboard, system architecture slide deck, and master the 30-second live stage demo.

---

### 🎬 6. THE 30-SECOND STAGE DEMO SCRIPT (THE JURY WOW MOMENT)

1. **Step 1 (The Call)**: A teammate calls the demo phone live from the jury desk using an AI voice generator saying: *"Papa, urgent ₹50,000 sent karo, accident ho gaya hai."*
2. **Step 2 (The Interception — within 3 seconds)**: Within the first 3-second audio window, the phone screen turns bright RED with an emergency chime, flashing:  
   🚨 **"AI SYNTHETIC VOICE DETECTED! DO NOT TRANSFER MONEY."**
3. **Step 3 (The Action)**: Team taps **"Lock UPI Transactions"** $\rightarrow$ UPI apps lock, and an official complaint payload is generated live for the **1930 National Cyber Helpline**!
