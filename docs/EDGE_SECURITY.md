# Raspberry Pi Edge Hardening & Next.js Backend Security Guide (2026)

This guide addresses physical edge security, device authentication, and anti-tamper protections for **Argus** and its camera/hardware ingestion scripts running on Raspberry Pi devices communicating with a **Next.js** backend API.

---

## 1. System Architecture & Threat Model

```
┌────────────────────────────────────────────────────────┐
│ Edge Device (Raspberry Pi at Weighbridge)              │
│ 1. Hardware Sensor & Camera Capture                    │
│ 2. Argus ANPR Engine (YOLO v11 + RapidOCR)             │
│ 3. Ingestion Client (Compiled with Nuitka)             │
│    Holds unique credentials (e.g., Device ID: pi-05)   │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTPS POST /api/telemetry
                           ▼
┌────────────────────────────────────────────────────────┐
│ Cloud / On-Prem Backend (Next.js)                      │
│ - Reverse Proxy (Nginx / Caddy / Cloudflare)           │
│ - Next.js Route Handlers (app/api/telemetry/route.ts)   │
│ - Database (Stores device identities & scoped roles)   │
└────────────────────────────────────────────────────────┘
```

### Key Context & Clarifications
1. **Per-Device Credentials**: Each Raspberry Pi has its own **unique** username and password / token. There is **no shared global master password**.
2. **Backend**: Built with **Next.js** (App Router Route Handlers / Middleware).
3. **Role Enforcement**: The backend verifies `user.role == 'device'` to enforce that devices can only `POST` telemetry and cannot read other records or perform administrative tasks.

---

## 2. Device Authentication in Next.js: Passwords vs API Keys vs `device.crt` (mTLS)

Since each Pi already has its own unique username and password, **the blast radius is isolated**: if Pi #5 is compromised, you can disable account `pi-05` in your Next.js database without affecting Pi #1 through #4.

However, how does this compare to `device.crt` (mTLS) and scoped API keys in a Next.js environment?

### Detailed Comparison

| Criterion | Unique Credentials + Argon2 + JWT Refresh | Per-Device API Key (`x-device-id` + `x-device-key`) | Mutual TLS (`device.crt` / mTLS) |
| :--- | :--- | :--- | :--- |
| **Protocol Flow in Next.js** | `POST /api/auth/login` (Argon2) $\rightarrow$ issues short-lived JWT + Refresh Token $\rightarrow$ `POST /api/telemetry` verifies JWT. | Single-step: Pi sends `x-device-id` and `x-device-key` directly in request headers. | Handshake-level: Identity verified during TLS handshake before HTTP request begins. |
| **Operational Feasibility** | **High**: Weighment is a 30–60s physical cycle; 100ms Argon2 cost on login/refresh is completely imperceptible. | **High**: Completely stateless; zero token refresh logic needed on edge. | **Maximum Security**: Reverse proxy filters connections at network perimeter. |
| **Next.js Server Load** | **Low on Telemetry**: Telemetry requests only verify fast JWT signatures ($<0.01$ ms). Argon2 is only computed on login/refresh. | **Ultra-lightweight**: Fast SHA-256 hash ($0.005$ ms) on every request. | Zero Node.js load: Reverse proxy drops unauthorized connections before touching Next.js. |
| **Edge Reliability on Network Drops** | Moderate: Pi needs retry logic to handle token expiration if the network drops for longer than the JWT lifetime. | Highly robust: 100% stateless; automatically recovers and resumes posting when network returns. | Maximum: Connection level; no tokens or sessions to expire or maintain. |
| **Credential Entropy** | High if machine-generated; Argon2id provides state-of-the-art memory-hardness against GPU cracking. | High: 256-bit CSPRNG cryptographic string (`argus_live_...`). | Cryptographic: Asymmetric 2048-bit RSA or ECC P-256 keypair. |

---

### Your Architecture: Argon2id + JWT Access & Refresh Tokens

You are using **Argon2** for password hashing and a **JWT access token + refresh token** architecture in Next.js. 

Here is why that works well in your operational reality:

1. **Weighment is a Physical Process (100ms is Completely Negligible)**:
   - A truck takes **30 to 60 seconds** to pull onto the weighbridge, settle its suspension, record tare/gross weight, and clear the gate.
   - A 100ms execution time for Argon2 during initial device boot or token refresh represents less than **0.2%** of a single weighment cycle. It has zero noticeable impact on weighbridge throughput.

2. **Telemetry POSTs Do NOT Run Argon2**:
   - Because you use JWT access tokens, the high-throughput `POST /api/telemetry` endpoint **does not execute Argon2 on every vehicle weighment**.
   - It only verifies the cryptographic signature of the JWT (using HMAC-SHA256 or Ed25519), which takes **under 0.05 milliseconds**.
   - Argon2 is only computed once during initial startup login and periodically when the refresh token is rotated.

3. **Edge Client Hardening for JWT Refresh**:
   - To make your edge ingestion script resilient during intermittent 4G/5G connections at the weighbridge:
     - **Preemptive Refresh**: Refresh the JWT access token when it reaches 70–80% of its lifespan, rather than waiting for an HTTP 401 error.
     - **Clock Drift Tolerance**: Ensure the Raspberry Pi synchronizes time via NTP (`chrony` or `systemd-timesyncd`). If the Pi's RTC drifts while offline, JWT timestamp validation (`nbf`, `exp`) can fail prematurely. Allow a 30–60 second clock skew window in your Next.js JWT verification options.

At first glance, both approaches look like an identifier and a secret. However, in backend engineering and IoT design, they operate completely differently:

1. **Hashing Algorithm & Next.js Event Loop Performance (SHA-256 vs bcrypt)**:
   - **Passwords** are designed for humans. Because humans pick predictable passwords, backend systems run slow, CPU-intensive algorithms like **bcrypt** or **argon2id** with work factor 10–12. Verifying a single password consumes ~100ms of 100% CPU thread time. In Next.js (single-threaded Node.js event loop), multiple devices authenticating simultaneously will stall incoming requests.
   - **Device Keys** are 256-bit high-entropy random strings (e.g. 32 bytes generated via CSPRNG). Because a 256-bit secret is computationally impossible to brute-force ($2^{256}$ search space), your Next.js server does **not** need slow bcrypt. It hashes the key using **SHA-256** (which takes $0.005$ milliseconds—**20,000x faster**) and verifies it using `crypto.timingSafeEqual`.

2. **Store-and-Forward Reconnection Spikes**:
   - Because Argus processes images in RAM and buffers to disk only during network outages, once internet connectivity is restored, the Pi flushes a burst of queued records (e.g., 50 to 200 readings).
   - If using username/password with session renewal or per-request verification, your Next.js server will choke on CPU-bound bcrypt hashing during reconnect bursts. With `x-device-key`, the server validates all 200 requests in a fraction of a millisecond.

3. **Stateless M2M Protocol vs Session Desynchronization**:
   - Traditional password authentication is stateful: the device must log in, store a JWT or session cookie, track expiration timestamps, and execute refresh flows.
   - At an unattended weighbridge subject to power cuts and intermittent 4G/5G signal, session tokens expire while the network is down. The device script then fails with `401 Unauthorized` and requires complex relogin recovery code.
   - An API key in the headers (`x-device-id` + `x-device-key`) is **100% stateless**. Every request is self-contained. When the network reconnects, the device simply posts immediately with zero negotiation.

4. **Domain Separation (Fleet Assets vs Human Accounts)**:
   - Modeling edge devices as "Users" inside your Next.js user database exposes them to human account logic: email verification, forgot-password emails, OAuth logins, and brute-force account lockouts.
   - Storing devices in a dedicated `Device` table isolates hardware assets cleanly: you can track `deviceId`, `hashedKey`, `hardwareSerial`, `firmwareVersion`, and `lastHeartbeat` without polluting your user authentication tables.

5. **Automated Secret Scanning**:
   - Device keys use structured prefixes (e.g. `argus_live_sec_...`). If an engineer or field technician accidentally commits a `.env` file or device log to GitHub, automated secret scanners instantly catch the token and notify you. Generic passwords cannot be scanned this way.

---

### Verdict: Is `device.crt` (mTLS) Better for Next.js?

**Yes, for zero-trust network perimeter defense, but with architectural tradeoffs:**

- **How mTLS works with Next.js**:
  Next.js itself (Node.js or Vercel serverless) typically does not terminate raw client certificates directly. Instead, you put **Nginx**, **Caddy**, or **Cloudflare API Shield** in front of Next.js:
  1. The reverse proxy terminates TLS and checks `device.crt` against your internal Certificate Authority (CA).
  2. If valid, the reverse proxy passes the verified device Common Name to Next.js via a trusted internal header:
     `x-forwarded-client-cert-cn: pi-05`.
  3. If invalid or missing, the reverse proxy **terminates the connection immediately**. Your Next.js app never processes a single byte from unauthorized callers.

- **The Pragmatic Alternative for Next.js (Per-Device API Keys)**:
  If running Next.js on managed platforms (like Vercel or Railway) where configuring custom mTLS reverse proxies is cumbersome, **Per-Device API Keys** (`x-device-id` + `x-device-token`) are vastly superior to a username/password session login.

#### Example Next.js Route Handler Implementation
```typescript
// app/api/telemetry/route.ts
import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import crypto from "crypto";

export async function POST(req: NextRequest) {
  const deviceId = req.headers.get("x-device-id");
  const deviceSecret = req.headers.get("x-device-key");

  if (!deviceId || !deviceSecret) {
    return NextResponse.json({ error: "Missing device credentials" }, { status: 401 });
  }

  // Fetch device from database
  const device = await db.device.findUnique({ where: { id: deviceId } });
  if (!device || device.role !== "device" || !device.isActive) {
    return NextResponse.json({ error: "Unauthorized or disabled device" }, { status: 403 });
  }

  // Constant-time token verification to prevent timing attacks
  const providedHash = crypto.createHash("sha256").update(deviceSecret).digest("hex");
  if (!crypto.timingSafeEqual(Buffer.from(providedHash), Buffer.from(device.hashedToken))) {
    return NextResponse.json({ error: "Invalid device key" }, { status: 403 });
  }

  // Process plate & hardware telemetry...
  const payload = await req.json();
  await db.reading.create({
    data: {
      deviceId: device.id,
      plate: payload.plate,
      vehicleType: payload.vehicleType,
      confidence: payload.confidence,
    },
  });

  return NextResponse.json({ success: true });
}
```

---

## 3. Challenge 2: The "Config File & Nuitka" Reality

### Can you put the unique username/password in a config file when compiling with Nuitka?

> [!CAUTION]
> **No.** Nuitka compiles Python code (`.py`) into native C/C++ machine code (`.so` or binary), but it **does not compile or encrypt external config files**.

1. **If credentials are in `.env` or `config.json`**:
   - The config file is an uncompiled, plain-text file stored on the disk.
   - Anyone mounting the SD card on a PC can open `.env` in Notepad and read `PI_05_PASSWORD=...`.
   - The attacker can then use those credentials to send fabricated weighment records to your Next.js API from their own laptop.
2. **If credentials are hardcoded in Python before running Nuitka**:
   - String literals inside compiled binaries reside in the `.rodata` section.
   - Running `strings my_compiled_app | grep -i pass` or opening the binary in disassemblers (Ghidra, IDA Pro) or `nuitka-static-unpacker` will extract the credentials in seconds.

### How to Mitigate This
- **Full Disk Encryption (LUKS)**: If the SD card / eMMC is encrypted, the config file cannot be read offline.
- **Hardware-Sealed Keys**: When using mTLS, private keys can be generated inside an ATECC608 secure element or TPM 2.0. The private key never exists as a file on disk and cannot be copied.

---

## 4. Challenge 3: Physical SD Card vs Soldering vs CM4/CM5

### Storage Lifecycle: In-Memory (RAM) vs Network Outage Spooling

Argus processes all images, crops, and OCR passes **in-memory (RAM)**. Images are never written to the SD card under normal operating conditions. 

However, during **network outages**, the ingestion service activates an offline queue (store-and-forward) and buffers readings/images to disk until connectivity is restored.

### Why Soldering the SD Card is Counterproductive (Even with In-Memory Operation)

While keeping images in RAM eliminates constant write wear during normal uptime, soldering the MicroSD card remains problematic:

1. **Power Cut Spooling & Corruption**:
   - In industrial weighbridge environments, power fluctuations or abrupt cuts are common.
   - If a sudden power loss occurs precisely while the offline store-and-forward queue is flushing or writing buffered records to the SD card, consumer FAT/ext4 SD card controllers frequently corrupt their internal partition tables.
   - If the card is physically soldered or epoxied, **the entire Raspberry Pi board is ruined** and requires desoldering or complete replacement.
2. **Physical Attackers Aren't Stopped**:
   - A thief or rogue operator with a soldering iron or hot air rework pencil can desolder the card in 60 seconds, or attach a micro-clip logic analyzer / SD ribbon probe directly to the exposed SD bus test points on the Raspberry Pi PCB without ever desoldering the socket.

### The Production Migration Path: Compute Module (CM4 / CM5)
Adopting the **Raspberry Pi Compute Module** for your next iteration is the ideal path:
- **Factory-Soldered eMMC Storage**: Eliminates the physical SD socket entirely with industrial BGA flash soldered at the factory.
- **Power-Loss Immune Controllers**: Industrial eMMC silicon includes hardware power-fail protection circuits to prevent partition corruption during power drops.
- **Tamper-Evident Enclosures**: CM4/CM5 carrier boards easily mount inside sealed, locked DIN-rail metal enclosures with chassis intrusion microswitches.

---

## 5. 2026 State-of-the-Art Raspberry Pi Hardening Stack

To protect both your code and credentials on the edge:

```
┌────────────────────────────────────────────────────────┐
│ 1. Code Compilation (Nuitka)                           │
│    Compile Argus & ingestion code into native binaries │
├────────────────────────────────────────────────────────┤
│ 2. Full Disk Encryption (LUKS + dm-crypt)              │
│    Root partition encrypted; unusable if extracted     │
├────────────────────────────────────────────────────────┤
│ 3. Network-Bound Decryption (Clevis + Tang)            │
│    Pi automatically unlocks only on company network/VPN│
├────────────────────────────────────────────────────────┤
│ 4. Hardware Root of Trust (Raspberry Pi 5 OTP)         │
│    SoC Boot ROM checks signed bootloader via customer  │
│    keys burned into silicon fuses                      │
├────────────────────────────────────────────────────────┤
│ 5. Next.js Perimeter Security                          │
│    Reverse proxy mTLS or unique per-device API tokens  │
│    scoped strictly to POST /api/telemetry              │
└────────────────────────────────────────────────────────┘
```

### 1. Dedicated Hardware Cryptographic Chips (TPM 2.0 vs Secure Elements)

Adding a dedicated silicon security module moves encryption keys and private certificates off the SD card completely:

#### A. Infineon OPTIGA™ TPM 2.0 (SLB 9672) / LetsTrust-TPM / ANAVI
- **2026 Verification**: The **OPTIGA™ SLB 9672** is the modern 2025/2026 revision (superseding the legacy SLB 9670) and is fully supported on the Raspberry Pi 5 via SPI using Linux kernel 6.6 / 6.12+.
- **Configuration**: Enabled cleanly in Raspberry Pi OS `/boot/firmware/config.txt` using:
  ```ini
  dtoverlay=tpm-slb9670
  ```
- **Mechanical Clearance on Pi 5**: Because the Raspberry Pi 5 uses the official Active Cooler fan/heatsink, mounting a TPM HAT directly over the 40-pin GPIO requires a **tall stacking header or 11mm standoff** to clear the fan shroud.
- **Disk Encryption Binding**: Use `systemd-cryptenroll` to bind the LUKS volume key to the TPM's internal PCR (Platform Configuration Register) measurements:
  ```bash
  systemd-cryptenroll --tpm2-device=auto --tpm2-pcrs=0+2+7 /dev/mmcblk0p2
  ```
  The disk will only decrypt if firmware, bootloader, and kernel remain untampered.

#### B. Microchip ATECC608B (Secure Element)
- **Interface**: Communicates over I2C (`dtoverlay=i2c-crypto-atecc608b`).
- **Use Case**: Best for storing asymmetric private keys in hardware. The private key **cannot be exported or read** by anyone mounting the SD card. Instead, the Pi requests the chip to sign JWT tokens or TLS handshakes inside the tamper-resistant silicon.

---

### 2. Enclosure Security & Tamper Switch Zeroization (GPIO)

Mounting the Raspberry Pi inside a locked, sealed DIN-rail enclosure with an intrusion switch provides physical defense against tampering.

#### The Circuit & Detection Logic
- **Hardware**: A **Normally Closed (NC) microswitch** mounted inside the enclosure lid, wired between a GPIO pin (e.g. GPIO 17) and GND.
- **Normal State**: The closed enclosure lid depresses the switch, holding the circuit CLOSED to GND.
- **Breach State**: If someone removes the lid, the switch opens, pulling GPIO 17 HIGH via the internal pull-up resistor.
- **Software Zeroization Daemon**:
  ```python
  from gpiozero import Button
  import os, subprocess

  def on_tamper_detected():
      # 1. Securely shred local credential files
      subprocess.run(["shred", "-u", "-z", "/path/to/.env", "/path/to/tokens.json"])
      # 2. Immediately close and dismount encrypted LUKS partitions
      subprocess.run(["cryptsetup", "close", "secure_storage"])
      # 3. Force kernel emergency reboot into un-decrypted locked state
      os.system("reboot -f")

  tamper_switch = Button(17, pull_up=True, bounce_time=0.05)
  tamper_switch.when_pressed = on_tamper_detected
  ```

#### The Critical "Unpowered Attack" Threat (2026 Reality)
> [!WARNING]
> **The Unpowered Attack Trap**: If an attacker simply **disconnects the power cord or battery first**, the Raspberry Pi is off. Opening the lid will **not** trigger a GPIO interrupt or execute the wipe script.

**How 2026 Architecture Solves This**:
1. **Volatile Key Protection via LUKS**: When power is cut, RAM immediately loses all data. Since the LUKS encryption key lives only in volatile RAM, disconnecting power **automatically locks the filesystem**. The attacker cannot read the unpowered SD card.
2. **Battery-Backed Tamper HSMs (e.g., Zymbit ZYMKEY 4i / HSM6)**: For absolute military-grade defense, modules like the Zymkey include an autonomous coin-cell battery-powered perimeter loop. Even when the Raspberry Pi is completely powered down, cutting or opening the perimeter loop **permanently destroys the internal master key in silicon**.

---

### Automated Headless Boot via Clevis + Tang (NBDE)
To ensure the encrypted disk decrypts automatically upon power-on without requiring a human to type a password:
1. The root partition is encrypted using **LUKS**.
2. On boot, the lightweight `initramfs` brings up networking and queries your internal **Tang server** (running on your weighbridge local network or secure VPN).
3. The Tang server validates network parameters and releases the key to unlock LUKS into RAM.
4. **If stolen**: Once taken off the authorized network, the Pi cannot reach the Tang server. The SD card/eMMC remains unreadable ciphertext.

---

## 6. Recommended Action Plan for Your Setup

| Priority | Step | Description |
| :--- | :--- | :--- |
| **Immediate** | **Retain Argon2 & Harden Edge JWT Refresh** | Keep Argon2 and JWT tokens. Implement preemptive token refresh (at 75% lifespan) and NTP time sync to avoid clock-skew rejection. |
| **Immediate** | **Compile Code with Nuitka** | Compile your Python ingestion script and Argus into native `.so` / ELF binaries. |
| **Next Step** | **Deploy LUKS Full Disk Encryption** | Encrypt root partition using LUKS so that local `.env` and token caches cannot be extracted offline. Unattended boot via Clevis/Tang or TPM 2.0 HAT. |
| **Hardware** | **Enclosure Switch & Compute Module** | Enclose the hardware in a locked DIN-rail box with an NC tamper microswitch, and transition to a Compute Module (CM4/CM5 with eMMC). |
