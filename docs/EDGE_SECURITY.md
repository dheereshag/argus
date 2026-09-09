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
                           │ HTTPS POST /api/entries
                           ▼
┌────────────────────────────────────────────────────────┐
│ Cloud / On-Prem Backend (Next.js)                      │
│ - Reverse Proxy (Nginx / Caddy / Cloudflare)           │
│ - Next.js Route Handlers (app/api/entries/route.ts)    │
│ - Database (Stores device identities & scoped roles)   │
└────────────────────────────────────────────────────────┘
```

### Key Context & Clarifications
1. **Per-Device Credentials**: Each Raspberry Pi has its own **unique** Per-Device API Key (`x-device-id` + `x-device-key`). There is **no shared global master password or single token**.
2. **Backend**: Built with **Next.js** (App Router Route Handlers / Middleware).
3. **Domain Separation & Role Enforcement**: Devices are registered in a dedicated `Device` database table (distinct from human `User` accounts). Devices are scoped strictly to `POST /api/entries` and cannot read other records, query dashboards, or access administrative routes.

---

## 2. Device Authentication in Next.js: Per-Device API Keys vs `device.crt` (mTLS)

Since each Pi has its own unique API key, **the blast radius is strictly isolated**: if Pi #5 is compromised or stolen, you can revoke `pi-05` in your Next.js database without affecting Pi #1 through #4 or any user accounts.

### Why Username & Password Was Discarded for Edge Hardware

Initially, traditional username/password authentication (with Argon2/bcrypt and JWT access/refresh rotation) might seem familiar from web applications. However, on unattended IoT edge devices, it is an anti-pattern:

1. **Identical Physical Vulnerability**:
   A static password stored in a file on the Raspberry Pi suffers from the *exact same physical extraction risk* as an API key. If an attacker mounts an unencrypted SD card or extracts strings from memory, they obtain the password just as easily. Passwords provide zero additional physical protection over an API key.
2. **Statefulness & 4G Network Drops**:
   At unattended weighbridges with intermittent connectivity, JWT access and refresh tokens expire during network drops. When connectivity returns, the device fails with `401 Unauthorized` and must run complex re-login and recovery routines before it can upload data. An API key is **100% stateless**—each request is self-contained.
3. **Next.js Event Loop Starvation During Store-and-Forward Reconnection**:
   Argus queues weighment readings locally during network outages. When 4G reconnects, the Pi flushes a burst of 50–200 queued records. If using passwords or sessions, incoming requests force CPU-intensive password hashing (**Argon2 / bcrypt**, taking ~100ms per verification). In single-threaded Node.js, this stalls the event loop. With Per-Device API Keys, Next.js verifies high-entropy keys using **SHA-256 in $0.005$ milliseconds (20,000x faster)**.
4. **Domain Separation (Fleet Assets vs Human Accounts)**:
   Hardware cameras should never be modeled as `User` records. They do not have email addresses, password reset flows, or MFA. Storing hardware in a dedicated `Device` table prevents fleet records from polluting human account tables.

---

### Detailed Comparison: Production Ingestion Approaches

| Criterion | Per-Device API Key (`x-device-id` + `x-device-key`) | Mutual TLS (`device.crt` / mTLS) | Hardware Asymmetric Key (TPM 2.0 / RFC 7523) |
| :--- | :--- | :--- | :--- |
| **Protocol Flow in Next.js** | Single-step: Pi sends `x-device-id` and `x-device-key` directly in HTTP request headers. | Handshake-level: Identity verified during TLS handshake before HTTP request begins. | Cryptographic: Pi signs request/JWT using hardware private key in TPM; verified against public key. |
| **Operational Feasibility** | **Maximum**: Works out-of-the-box on managed platforms (Vercel, Railway, Node.js) with zero custom reverse proxy setup. | **High (with Proxy)**: Requires Nginx, Caddy, or Cloudflare API Shield to terminate raw client certs. | **Maximum Security**: Zero secrets stored on disk; private key permanently locked in silicon. |
| **Next.js Server Load** | **Ultra-lightweight**: Fast SHA-256 hash ($0.005$ ms) on every request using constant-time equality. | Zero Node.js load: Reverse proxy drops unauthorized connections before touching Next.js. | Minimal: Public key signature verification ($<0.05$ ms) with no database secret lookups. |
| **Edge Reliability on Network Drops** | **100% Stateless**: Automatically recovers and resumes posting instantly when network returns. Zero token refresh code. | Maximum: Connection level; no tokens or sessions to expire or maintain. | Highly robust: Signs requests on-the-fly using hardware clock; stateless. |
| **Credential Entropy** | High: 256-bit CSPRNG cryptographic string (`argus_live_sec_...`). | Cryptographic: Asymmetric 2048-bit RSA or ECC P-256 keypair. | Cryptographic: Hardware-generated non-exportable ECC P-256 keypair. |

---

### Recommended Stateless Edge Python Ingestion Client

Because credentials are sent via headers, the edge ingestion code is simple, reliable, and immune to token expiration or session desynchronization:

```python
import requests

class WeighbridgeDeviceClient:
    """Stateless edge ingestion client using per-device API keys."""

    def __init__(self, base_url: str, device_id: str, device_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "x-device-id": device_id,
            "x-device-key": device_key,
            "Content-Type": "application/json",
        }

    def post_entry(self, entry_payload: dict) -> requests.Response:
        """
        Posts weighment entry to Next.js API.
        Completely stateless: no login, no JWT refresh loops, no session expiry.
        """
        return requests.post(
            f"{self.base_url}/api/entries",
            json=entry_payload,
            headers=self.headers,
            timeout=10,
        )
```

---

### Next.js Route Handler Implementation

The Next.js backend hashes the incoming `x-device-key` using SHA-256 and compares it against the stored hash in the database using `crypto.timingSafeEqual` to prevent timing attacks:

```typescript
// app/api/entries/route.ts
import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import crypto from "crypto";

export async function POST(req: NextRequest) {
  const deviceId = req.headers.get("x-device-id");
  const deviceKey = req.headers.get("x-device-key");

  if (!deviceId || !deviceKey) {
    return NextResponse.json(
      { error: "Missing device credentials" },
      { status: 401 }
    );
  }

  // 1. Fetch device record from database
  const device = await db.device.findUnique({
    where: { id: deviceId },
  });

  if (!device || !device.isActive) {
    return NextResponse.json(
      { error: "Unauthorized or disabled device" },
      { status: 403 }
    );
  }

  // 2. Constant-time SHA-256 verification to prevent timing attacks
  const providedHash = crypto
    .createHash("sha256")
    .update(deviceKey)
    .digest("hex");

  const isMatch =
    providedHash.length === device.hashedKey.length &&
    crypto.timingSafeEqual(
      Buffer.from(providedHash),
      Buffer.from(device.hashedKey)
    );

  if (!isMatch) {
    return NextResponse.json({ error: "Invalid device key" }, { status: 403 });
  }

  // 3. Update device last seen heartbeat
  await db.device.update({
    where: { id: device.id },
    data: { lastSeenAt: new Date() },
  });

  // 4. Ingest weighment & plate entry
  const payload = await req.json();
  const entry = await db.entry.create({
    data: {
      deviceId: device.id,
      plate: payload.plate,
      vehicleType: payload.vehicleType,
      confidence: payload.confidence,
    },
  });

  return NextResponse.json({ success: true, id: entry.id }, { status: 201 });
}
```

---

### Data Modeling: `deviceId` vs Eliminating `type` ('automatic' vs 'manual')

Introducing `deviceId` to the `Entry` table enables cleaner database normalization and solves ambiguity between automated edge readings and human operator entries:

#### 1. Why `deviceId` is Essential
In multi-lane or multi-camera facilities, knowing that a reading was simply "automatic" is insufficient—you must record **which specific camera or weighbridge gate** captured the vehicle. This enables:
- Per-lane OCR accuracy and misread tracking.
- Immediate blast-radius queries if a camera goes out of alignment: `SELECT * FROM entries WHERE deviceId = 'lane-02'`.
- Health monitoring and telemetry (`lastSeenAt`, firmware version) per gate.

#### 2. Can You Eliminate the `type` Column?
**Yes.** Instead of maintaining a redundant `type: 'automatic' | 'manual'` column that can drift out of sync with foreign keys, the source can be derived:
- If `deviceId != null` $\rightarrow$ The entry was created automatically by edge hardware.
- If `userId != null` (or `deviceId == null`) $\rightarrow$ The entry was typed manually by an operator.

```prisma
// Recommended Prisma Schema
model Device {
  id          String    @id // e.g. "pi-lane-01-inbound"
  name        String
  hashedKey   String    // SHA-256 hash of x-device-key
  isActive    Boolean   @default(true)
  lastSeenAt  DateTime?
  entries     Entry[]
  createdAt   DateTime  @default(now())
}

model Entry {
  id           String    @id @default(cuid())
  plate        String
  vehicleType  String?
  confidence   Float?

  // Origin & Audit Trail
  deviceId     String?   // Set if automated camera capture
  device       Device?   @relation(fields: [deviceId], references: [id])
  createdById  String?   // Set if created manually by a human operator
  verifiedById String?   // Set if an operator reviewed/corrected the plate

  createdAt    DateTime  @default(now())
}
```

> [!TIP]
> **Handling Operator Overrides**: In real weighbridge operations, a camera may automatically detect a plate (`deviceId: "pi-01"`), but a human operator might correct a misread character. Keeping `deviceId` (captured by) alongside `verifiedById` / `updatedById` (inspected or corrected by) provides a complete, legally compliant audit trail without needing an awkward boolean flag.

---

### What About `device.crt` (mTLS)? When Should You Upgrade?

While Per-Device API Keys are the pragmatic choice for managed platforms (Vercel, Railway), **mTLS** provides zero-trust network perimeter defense:
- **How mTLS works with Next.js**:
  Next.js itself does not terminate raw TLS certificates. You place **Nginx**, **Caddy**, or **Cloudflare API Shield** in front of Next.js:
  1. The reverse proxy terminates TLS and checks `device.crt` against your internal Certificate Authority (CA).
  2. If valid, the reverse proxy passes the verified device ID to Next.js via a trusted internal header (`x-forwarded-client-cert-cn: pi-05`).
  3. If invalid or missing, the reverse proxy **terminates the connection immediately**—Next.js never processes unauthorized requests.
- **Verdict**: Start with Per-Device API Keys (`x-device-id` + `x-device-key`). If company compliance requires mutual TLS, add an Nginx/Cloudflare reverse proxy in front of Next.js without needing to rewrite application logic.

---

## 3. Challenge 2: The "Config File & Nuitka" Reality

### Can you put device API keys or credentials in a config file when compiling with Nuitka?

> [!CAUTION]
> **No.** Nuitka compiles Python code (`.py`) into native C/C++ machine code (`.so` or binary), but it **does not compile or encrypt external config files**.

Both passwords and static API keys are **shared bearer secrets**. Storing either one in plaintext on an unencrypted SD card exposes them to identical physical extraction risks:

1. **If credentials are in `.env` or `config.json`**:
   - The config file is an uncompiled, plain-text file stored on the disk.
   - Anyone mounting the SD card on a PC can open `.env` in Notepad and read `PI_DEVICE_KEY=argus_live_sec_...`.
   - The attacker can then use those credentials to send fabricated weighment records to your Next.js API from their own laptop.
2. **If credentials are hardcoded in Python before running Nuitka**:
   - String literals inside compiled binaries reside in the `.rodata` section.
   - Running `strings my_compiled_app | grep -i sec_` or opening the binary in disassemblers (Ghidra, IDA Pro) or `nuitka-static-unpacker` will extract the credentials in seconds.

### How to Mitigate This
- **Full Disk Encryption (LUKS)**: If the SD card / eMMC is encrypted, the config file cannot be read offline.
- **Hardware-Sealed Keys (TPM 2.0 / ATECC608)**: Move beyond static symmetric secrets to asymmetric keys generated inside a secure element. The private key never exists as a file on disk and cannot be copied.

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
│    scoped strictly to POST /api/entries                │
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
| **Immediate** | **Adopt Per-Device API Keys (`x-device-id` + `x-device-key`)** | Deploy stateless per-device API keys with SHA-256 verification in Next.js. Eliminates token expiry during 4G network drops and avoids CPU-heavy password hashing. |
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

### Alternative 1: Asymmetric "Private Key JWT" (RFC 7523) instead of Static API Keys

#### The Limitation of Shared Static Secrets on Edge Devices
Even though Per-Device API Keys are vastly superior to passwords in Next.js, an API key is still a static symmetric shared secret. If extracted from unencrypted storage or RAM, an attacker can authenticate from another machine. Furthermore, **a TPM cannot protect a symmetric secret**—TPM chips are designed specifically to protect asymmetric cryptographic keypairs.

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
| **Tier 1: Practical Foundation** | **Per-Device API Keys + Nuitka + LUKS** | **Current** | **Production Baseline**: Zero session desync on 4G drops, fast SHA-256 Next.js verification, isolated blast radius per device. |
| **Tier 2: Hardware Secret Lock** | **Private Key JWT via TPM 2.0 (RFC 7523)** | **Low–Med** | **Recommended Next**: Eliminates all static secrets from the Pi; locks identity into hardware silicon. |
| **Tier 3: Network Stealth** | **Tailscale / Cloudflare Zero-Trust Tunnel** | **Low** | **Recommended Next**: Closes backend to the public internet; eliminates DDoS and public scraping. |
| **Tier 4: Enterprise Appliance** | **CM4/CM5 + eMMC + `dm-verity` + FDO** | **High** | **Commercial Scale**: Tamper-proof, immutable appliance with zero-touch factory provisioning. |
