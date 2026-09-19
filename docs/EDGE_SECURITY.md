# Raspberry Pi Edge Hardening & Next.js Backend Security Guide (2026)

This guide addresses physical edge security, device authentication, and anti-tamper protections for **Argus** and its camera/hardware ingestion scripts running on Raspberry Pi devices communicating with a **Next.js** backend API.

---

## 1. System Architecture & Threat Model

```
┌────────────────────────────────────────────────────────┐
│ Edge Device (Raspberry Pi at Weighbridge)              │
│ 1. Hardware Sensor & Camera Capture                    │
│ 2. Argus ANPR Engine (YOLO26 + RapidOCR)               │
│ 3. Ingestion Client (Compiled with Nuitka)             │
│    Credentials: EDGE_USERNAME + EDGE_PASSWORD          │
│    Session: Managed via requests.Session()             │
└──────────────────────────┬─────────────────────────────┘
                           │ 1. POST /api/auth/login (Auth Session)
                           │ 2. HTTPS POST /api/entries (Session Cookie)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Cloud / On-Prem Backend (Next.js)                      │
│ - Session Auth Handler (app/api/auth/login/route.ts)   │
│ - Next.js Ingestion Route (app/api/entries/route.ts)   │
│ - Database (Stores user credentials & session states)  │
└────────────────────────────────────────────────────────┘
```

### Key Context & Clarifications
1. **Credentials & Session Architecture**: Each Raspberry Pi edge ingestion client authenticates using a dedicated **Username & Password** account. Once authenticated, Next.js issues an authenticated session (via encrypted `HttpOnly` cookie or session token).
2. **Session Persistence on Edge**: The Python client maintains state using `requests.Session()`, which stores the session cookie and automatically attaches it to subsequent requests.
3. **Automated Session Resilience (4G Drops & Expiry)**: Unattended weighbridges frequently experience network disconnects, server restarts, or session timeouts. If `POST /api/entries` receives a `401 Unauthorized`, the client automatically re-authenticates via `POST /api/auth/login` and seamlessly retries the upload.
4. **Backend**: Next.js App Router route handlers with encrypted session cookies (e.g., `iron-session`, NextAuth, or session database table).
5. **Role Enforcement**: Edge service accounts have `role: "EDGE_DEVICE"`, scoped strictly to `POST /api/entries` and unable to access operator UI or administrative routes.

---

## 2. Device Authentication in Next.js: Username & Password with Session Management

Using username and password paired with robust **session management** allows edge devices to securely authenticate against standard Next.js authentication stacks while maintaining high reliability in industrial IoT environments.

### How Username/Password + Session Management Operates on Edge Devices

1. **Initial Login**:
   - The edge client reads `EDGE_USERNAME` and `EDGE_PASSWORD` from local configuration (`.env`).
   - It sends a `POST /api/auth/login` request with `{ "username": "...", "password": "..." }`.
   - Next.js verifies the credentials against the database (using password hashing like Argon2id or bcrypt) and sets an encrypted session cookie (`HttpOnly; Secure; SameSite=Lax`).
2. **Amortizing CPU Cost (Event Loop Protection)**:
   - Password hashing with Argon2/bcrypt is intentionally CPU-intensive (~100ms per verification).
   - In single-threaded Node.js (Next.js), if the Pi had to transmit and verify a password on *every* weighment upload, bursting 50–100 queued records after a 4G reconnect would freeze the server's event loop for 5–10 seconds.
   - **Session management completely solves this**: The heavy password hash runs **only once** on initial login. All subsequent requests validate the lightweight session cookie ($<0.01$ ms), keeping Next.js ultra-fast and responsive.
3. **Session Auto-Recovery on Network Drops & Backend Restarts**:
   - At unattended weighbridges with intermittent 4G connectivity, or when the Next.js server is redeployed, existing sessions may expire or be cleared.
   - If an expired session is sent, Next.js responds with `401 Unauthorized`.
   - The production Python client wraps API calls with an **automatic re-login guardrail**: when receiving a 401, it immediately re-authenticates via `login()` and transparently retries `post_entry()`.
4. **Offline Store-and-Forward Integration**:
   - While the network is down, readings are queued locally in RAM (or encrypted local spool).
   - Once connectivity returns, the client logs in once, establishes an active session, and flushes all queued records without repeated login overhead.

---

### Production Edge Python Ingestion Client (`requests.Session()`)

The edge ingestion client uses `requests.Session()` to persist cookies across requests, combined with an automatic re-login loop on `401 Unauthorized`:

```python
import logging
import requests
from typing import Any

logger = logging.getLogger("argus.edge_client")

class WeighbridgeSessionClient:
    """
    Production edge ingestion client using username/password with
    automatic session management and 401 re-login recovery.
    """

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self._is_logged_in = False

    def login(self) -> bool:
        """
        Authenticates against Next.js with username & password to establish a session.
        Stores the resulting session cookie automatically in self.session.
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/api/auth/login",
                json={"username": self.username, "password": self.password},
                timeout=10,
            )
            if resp.status_code == 200:
                self._is_logged_in = True
                logger.info("Edge session established successfully with Next.js.")
                return True

            logger.error("Authentication failed: HTTP %s - %s", resp.status_code, resp.text)
            return False
        except requests.RequestException as exc:
            logger.error("Network error during session login: %s", exc)
            return False

    def post_entry(self, entry_payload: dict[str, Any], retry_on_401: bool = True) -> requests.Response:
        """
        Posts a weighment entry to Next.js API using the active session.
        If the session has expired (HTTP 401), automatically re-authenticates and retries.
        """
        if not self._is_logged_in:
            if not self.login():
                raise ConnectionError("Cannot post entry: initial edge authentication failed")

        try:
            resp = self.session.post(
                f"{self.base_url}/api/entries",
                json=entry_payload,
                timeout=10,
            )

            # Auto-relogin on session expiration or backend restart
            if resp.status_code == 401 and retry_on_401:
                logger.warning("Session expired (HTTP 401). Re-authenticating...")
                if self.login():
                    return self.post_entry(entry_payload, retry_on_401=False)

            return resp
        except requests.RequestException as exc:
            logger.error("Network error posting entry: %s", exc)
            raise
```

---

### Next.js Route Handler Implementation

#### 1. Login Route (`app/api/auth/login/route.ts`)
Validates username and password, then sets an encrypted session cookie:

```typescript
// app/api/auth/login/route.ts
import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import bcrypt from "bcrypt";
import { createSession } from "@/lib/session"; // iron-session, NextAuth, or custom cookie

export async function POST(req: NextRequest) {
  const { username, password } = await req.json();

  if (!username || !password) {
    return NextResponse.json(
      { error: "Username and password required" },
      { status: 400 }
    );
  }

  // 1. Fetch user by username
  const user = await db.user.findUnique({
    where: { username },
  });

  if (!user || !user.isActive) {
    return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
  }

  // 2. Verify hashed password (Argon2 / bcrypt)
  const isValid = await bcrypt.compare(password, user.passwordHash);
  if (!isValid) {
    return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
  }

  // 3. Create encrypted session cookie (HttpOnly, Secure, SameSite=Lax)
  const response = NextResponse.json({
    success: true,
    user: { id: user.id, username: user.username, role: user.role },
  });

  await createSession(response, {
    userId: user.id,
    role: user.role,
    deviceId: user.deviceId,
  });

  return response;
}
```

#### 2. Ingestion Route (`app/api/entries/route.ts`)
Validates the active session cookie and ensures the user role is authorized:

```typescript
// app/api/entries/route.ts
import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getSession } from "@/lib/session";

export async function POST(req: NextRequest) {
  // 1. Validate session from request cookies
  const session = await getSession(req);

  if (!session || !session.userId) {
    return NextResponse.json(
      { error: "Unauthorized: Active session required" },
      { status: 401 }
    );
  }

  // 2. Role-based access control: ensure edge account or admin
  if (session.role !== "EDGE_DEVICE" && session.role !== "ADMIN") {
    return NextResponse.json(
      { error: "Forbidden: Insufficient privileges for ingestion" },
      { status: 403 }
    );
  }

  // 3. Ingest weighment & plate entry
  const payload = await req.json();
  const entry = await db.entry.create({
    data: {
      deviceId: session.deviceId,
      userId: session.userId,
      plate: payload.plate,
      vehicleType: payload.vehicleType,
      confidence: payload.confidence,
    },
  });

  return NextResponse.json({ success: true, id: entry.id }, { status: 201 });
}
```

---

### Data Modeling: Accounts, Roles, & Devices

Using a unified `User` model with dedicated roles (`EDGE_DEVICE`, `OPERATOR`, `ADMIN`) provides full domain separation while keeping authentication uniform:

```prisma
// Recommended Prisma Schema
model User {
  id           String    @id @default(cuid())
  username     String    @unique
  passwordHash String
  role         Role      @default(OPERATOR)
  deviceId     String?   // Links edge account to physical gate (e.g. "pi-lane-01")
  isActive     Boolean   @default(true)
  entries      Entry[]
  createdAt    DateTime  @default(now())
}

enum Role {
  OPERATOR
  ADMIN
  EDGE_DEVICE
}

model Entry {
  id           String    @id @default(cuid())
  plate        String
  vehicleType  String?
  confidence   Float?

  // Origin & Audit Trail
  deviceId     String?   // e.g. "pi-lane-01"
  userId       String?   // Ingested by edge user or operator
  user         User?     @relation(fields: [userId], references: [id])
  verifiedById String?   // Set if an operator reviewed/corrected the plate

  createdAt    DateTime  @default(now())
}
```

> [!TIP]
> **Handling Operator Overrides**: In real weighbridge operations, a camera automatically detects a plate under an edge session (`userId` of the Pi, `deviceId: "pi-01"`). If a human operator later corrects a character on their dashboard, setting `verifiedById` provides a complete, legally compliant audit trail without needing redundant status flags.

---

## 3. Challenge 2: The "Config File & Nuitka" Reality

### Can you put credentials in a config file when compiling with Nuitka?

> [!CAUTION]
> **No.** Nuitka compiles Python code (`.py`) into native C/C++ machine code (`.so` or binary), but it **does not compile or encrypt external config files**.

Storing edge credentials (`EDGE_USERNAME` and `EDGE_PASSWORD`) in plaintext on an unencrypted SD card exposes them to physical extraction risks:

1. **If credentials are in `.env` or `config.json`**:
   - The config file is an uncompiled, plain-text file stored on the disk.
   - Anyone mounting the SD card on a PC can open `.env` in Notepad and read `EDGE_PASSWORD=...`.
   - The attacker can then use those credentials to authenticate against your Next.js API from their own machine.
2. **If credentials are hardcoded in Python before running Nuitka**:
   - String literals inside compiled binaries reside in the `.rodata` section.
   - Running `strings my_compiled_app | grep -i pass` or opening the binary in disassemblers (Ghidra, IDA Pro) will extract credentials in seconds.

### How to Mitigate This
- **Restrict File Permissions**: Enforce `chmod 600 .env` owned by `root:root` so only the service process can read the credentials.
- **Full Disk Encryption (LUKS)**: If the SD card / eMMC is encrypted, the config file cannot be read offline.
- **Hardware-Sealed Keys (TPM 2.0 / ATECC608)**: Move beyond static symmetric secrets to asymmetric keys generated inside a secure element. The private key never exists as a file on disk and cannot be copied.

### 3.1 Compiling Application Code with Nuitka (Zero Source on Edge)

To prevent reverse-engineering of proprietary weighbridge business rules and plate normalization heuristics while keeping heavy dependencies uncompiled:

1. **Compilation Strategy (`--nofollow-imports`)**:
   - Heavy dependencies (`torch`, `ultralytics`, `rapidocr`, `onnxruntime`, `cv2`) are pre-compiled native C/C++ wheels. Recompiling them with Nuitka is slow, error-prone, and unnecessary.
   - Nuitka is invoked with `--module --include-package=app --nofollow-imports --lto=yes --python-flag=no_docstrings`.
   - Only `app/` is compiled into a single native shared object (`app.cpython-314-aarch64-linux-gnu.so`).

2. **Automated `release-arm64` Deployment Branch**:
   - GitHub Actions (`.github/workflows/nuitka-arm64.yml`) runs on native `ubuntu-26.04-arm`.
   - It compiles `app/`, deletes all `app/**/*.py` source files, and force-pushes the compiled binary tree to an orphan `release-arm64` branch.

3. **Raspberry Pi `git pull` Workflow**:
   ```bash
   # One-time setup on Pi:
   git clone -b release-arm64 https://github.com/dheereshag/argus.git /opt/argus
   cd /opt/argus && cp .env.example .env && uv sync --no-dev

   # Updating Pi in production (zero .py source files downloaded):
   cd /opt/argus
   git pull origin release-arm64
   uv sync --no-dev
   sudo systemctl restart argus
   ```
   On the Pi, Python loads `app` directly from the native `.so` shared library. No `.py` source code resides on the device.

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
│    Session-based auth with encrypted HttpOnly cookies, │
│    strict role scoping (EDGE_DEVICE), and rate limits  │
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
| **Immediate** | **Deploy Username/Password with Session Management** | Configure dedicated edge user accounts (`EDGE_DEVICE` role), `requests.Session()` with automatic 401 re-login recovery, and encrypted session cookies in Next.js. |
| **Immediate** | **Compile Code with Nuitka** | Compile your Python ingestion script and Argus into native `.so` / ELF binaries. |
| **Next Step** | **Deploy LUKS Full Disk Encryption** | Encrypt root partition using LUKS so that local `.env` and credential caches cannot be extracted offline. Unattended boot via Clevis/Tang or TPM 2.0 HAT. |
| **Hardware** | **Enclosure Switch & Compute Module** | Enclose the hardware in a locked DIN-rail box with an NC tamper microswitch, and transition to a Compute Module (CM4/CM5 with eMMC). |

---

## 7. 2026 Next-Gen Alternatives & Evolution Roadmap

While the baseline architecture (Per-Device API Keys + Nuitka + LUKS) provides a solid, practical foundation for MVP and pilot deployments, the **2026 state-of-the-art for enterprise IoT & edge fleets** offers four superior architectural alternatives that eliminate the remaining physical and network vulnerabilities:

```
┌────────────────────────────────────────────────────────┐
│ 1. Asymmetric Private Key JWT (RFC 7523 / DPoP)        │
│    Zero static secrets exist; private key in TPM       │
├────────────────────────────────────────────────────────┤
│ 2. Zero-Trust Private Mesh (Tailscale / Cloudflare)    │
│    Next.js endpoints completely hidden from internet   │
├────────────────────────────────────────────────────────┤
│ 3. Immutable Read-Only OS (dm-verity)                  │
│    Kernel halts if a single byte of code is modified   │
├────────────────────────────────────────────────────────┤
│ 4. FIDO Device Onboard (FDO)                           │
│    Zero-touch factory enrollment; no manual configs    │
└────────────────────────────────────────────────────────┘
```

---

### Alternative 1: Asymmetric "Private Key JWT" (RFC 7523) instead of Static Credentials

#### The Limitation of Shared Static Secrets on Edge Devices
Any static credential (whether a password or API key) is a shared secret stored on the device. If extracted from unencrypted storage or RAM, an attacker can authenticate from another machine. Furthermore, **a TPM cannot protect a symmetric secret or password**—TPM chips are designed specifically to protect asymmetric cryptographic keypairs.

#### The 2026 Solution: Private Key JWT (RFC 7523 / DPoP)
Instead of a shared API key, each Raspberry Pi is assigned an **asymmetric keypair** (ECC NIST P-256 or Ed25519):
1. **Hardware Silicon Key Storage**: The **Private Key** is generated inside the **TPM 2.0 (SLB 9672)** or **ATECC608B** chip and marked **non-exportable**. It can never be copied to an SD card or read by an attacker.
2. **Public Key on Next.js**: The corresponding **Public Key** is stored in your Next.js database under `Device.publicKey`.
3. **How Authentication Works**:
   - When the Pi sends an entry, it asks the TPM to sign a short-lived token or payload:
     `{"iss": "pi-05", "sub": "pi-05", "exp": now + 60s}`.
   - The Pi sends this signed token to Next.js (`POST /api/entries`).
   - Next.js verifies the cryptographic signature against `Device.publicKey` in $<0.01$ ms and processes the entry.
4. **Why This Wins**:
   - **Zero static secrets exist anywhere** on the Pi.
   - Even if a thief clones the entire SD card, **they cannot authenticate because the private key is physically trapped inside silicon**.

---

### Alternative 2: Zero-Trust Private Mesh (Tailscale / WireGuard / Cloudflare Tunnel)

#### The Limitation of Public Ingress
Hosting `https://your-domain.com/api/entries` on the public internet means anyone can port-scan, DDoS, or attempt brute-force probing against your ingestion endpoints.

#### The 2026 Solution: Peer-to-Peer Encrypted Overlay Mesh
Using **Tailscale** (built on WireGuard) or a **Cloudflare Zero-Trust Tunnel**:
1. Your Next.js backend and all weighbridge Pis join a private peer-to-peer overlay network (e.g. `100.64.0.0/10`).
2. The Next.js `/api/entries` endpoint is **completely closed to the public internet** (ports 80/443 closed to public traffic).
3. The Raspberry Pi posts directly over the encrypted mesh IP:
   `http://100.64.0.1:3000/api/entries`.
4. **Why This Wins**:
   - Your weighbridge API is 100% invisible to the public internet.
   - Authentication and packet encryption happen at the Linux kernel WireGuard layer before Next.js even receives a TCP packet.

---

### Alternative 3: Immutable Read-Only OS with `dm-verity`

#### The Limitation of Standard Writable Filesystems
On standard Raspberry Pi OS (Debian), the root partition is a writable `ext4` filesystem. An attacker with local access can modify system libraries (`libc.so`), alter Python runtime binaries, or inject dynamic linker preload hooks (`LD_PRELOAD`) to intercept unencrypted memory.

#### The 2026 Solution: Kernel-Enforced Integrity via `dm-verity`
Standard in ChromeOS, Android, and industrial Linux (Ubuntu Core / Yocto):
1. The root OS and compiled Nuitka binaries are packaged into a **read-only squashfs image** backed by a cryptographic Merkle hash tree (**`dm-verity`**).
2. The root hash of the tree is signed by your private key and verified by the Raspberry Pi 5 Boot ROM.
3. If an attacker tampers with even **one single byte** of your compiled code or OS files on the SD card, the Linux kernel detects the hash mismatch and **instantly halts the machine with a kernel panic**.
4. All variable runtime data (offline store-and-forward queue) is isolated into a separate encrypted partition or RAM (`tmpfs`).

---

### Alternative 4: FIDO Device Onboard (FDO) for Zero-Touch Fleet Provisioning

#### The Limitation of Manual Setup
When deploying 20 or 50 weighbridges, manual provisioning requires flashing distinct API keys into config files for every single SD card.

#### The 2026 Solution: FIDO Device Onboard (FDO / LF Edge)
1. You ship stock Raspberry Pis straight to the weighbridge installations.
2. On initial power-on, the device contacts an automated FDO Rendezvous Server.
3. The device cryptographically proves its factory silicon identity, downloads its encrypted configuration and access certificates from your Next.js server, and enrolls itself into your fleet with **zero human intervention**.

---

### Tiered Evolution Matrix

| Security Tier | Architecture Stack | Effort | Best Used For |
| :--- | :--- | :--- | :--- |
| **Tier 1: Practical Foundation** | **Username/Password + Session Management + Nuitka + LUKS** | **Current** | **Production Baseline**: Authenticated session with auto-relogin on 401, amortized password hashing in Next.js, and encrypted storage on Pi. |
| **Tier 2: Hardware Secret Lock** | **Private Key JWT via TPM 2.0 (RFC 7523)** | **Low–Med** | **Recommended Next**: Eliminates all static secrets from the Pi; locks identity into hardware silicon. |
| **Tier 3: Network Stealth** | **Tailscale / Cloudflare Zero-Trust Tunnel** | **Low** | **Recommended Next**: Closes backend to the public internet; eliminates DDoS and public scraping. |
| **Tier 4: Enterprise Appliance** | **CM4/CM5 + eMMC + `dm-verity` + FDO** | **High** | **Commercial Scale**: Tamper-proof, immutable appliance with zero-touch factory provisioning. |
