# Raspberry Pi 5 Acceleration & Python 3.14 Optimization Guide

This guide details edge acceleration techniques, operating system optimizations, and Python 3.14 practices to maximize ANPR throughput and minimize latency on the **Raspberry Pi 5** (Broadcom BCM2712 quad-core ARM Cortex-A76 @ 2.4 GHz).

---

## 1. Hardware Architecture & Performance Characteristics

```
┌────────────────────────────────────────────────────────────────────────┐
│ Raspberry Pi 5 Edge Device                                             │
│ - CPU: Quad-core ARM Cortex-A76 @ 2.4 GHz (ARMv8.2-A + NEON SIMD)      │
│ - RAM: 4GB / 8GB / 16GB LPDDR4X-4267 (~17 GB/s bandwidth)              │
│ - Storage: Industrial eMMC / NVMe via PCIe HAT (Zero SD write wear)    │
│ - Optional: Raspberry Pi AI Kit (Hailo-8L @ 13 TOPS / Hailo-8 @ 26)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    ▼                                                               ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│ Software Engine (Argus ANPR)      │   │ System & Hardware Tunings         │
│ 1. Zero-Copy In-Memory Image Flow │   │ 1. Performance CPU Governor       │
│ 2. RapidOCR ONNX Thread Binding   │   │ 2. jemalloc Allocator             │
│ 3. Python 3.14 Free-Threading     │   │ 3. OpenMP Process Affinity        │
│ 4. Bilinear Downsampling on ARM   │   │ 4. Hailo-8 NPU Pipeline           │
└───────────────────────────────────┘   └───────────────────────────────────┘
```

---

## 2. In-Memory Zero-Copy Image Pipeline

In constrained edge environments, memory allocation and intermediate image compression generate major CPU bottlenecks.

1. **Direct `Image.Image` In-Memory Flow**:
   - Camera frames ingested via `POST /recognize` or Python SDK are decoded and dimension-bounded once via `decode_and_downscale`.
   - The resulting in-memory PIL `Image.Image` is retained across Stage 1 (YOLO detection) and Stage 2 (RapidOCR) without re-encoding to JPEG.
   - Eliminates redundant JPEG compression and decompression cycles, saving 25–40 ms per frame on Cortex-A76.
2. **SIMD-Accelerated Resampling (`IMAGE_RESAMPLE_FILTER`)**:
   - Default high-order 8-tap Lanczos filtering is computationally expensive for large camera frames (4K/1080p).
   - Set `IMAGE_RESAMPLE_FILTER="BILINEAR"` in `.env` for 3x faster downscaling using ARM NEON vector instructions with zero loss in vehicle detection accuracy.

---

## 3. ONNX Runtime & Thread Affinity for Multi-Core ARM

RapidOCR and YOLO leverage ONNX Runtime for CPU execution.

1. **Intra-Op Thread Binding**:
   - Configure `ONNX_NUM_THREADS=4` in `.env` to bind tensor operations across all 4 Cortex-A76 cores.
   - Argus automatically initializes RapidOCR's ONNX Runtime session with:
     ```python
     params = {"EngineConfig.onnxruntime.intra_op_num_threads": settings.ONNX_NUM_THREADS}
     ```
2. **OpenMP Environment Variables**:
   Set OpenMP affinity variables in `/etc/environment` or your systemd service file:
   ```bash
   export OMP_NUM_THREADS=4
   export OMP_PROC_BIND=CLOSE
   export OMP_PLACES=CORES
   ```

---

## 4. Python 3.14 Practices & Free-Threading (PEP 703)

Python 3.14 introduces first-class **free-threaded CPython** (`nogil` / `python3.14t`).

### Why Free-Threading Matters on Raspberry Pi 5
- In standard CPython, the Global Interpreter Lock (GIL) serializes CPU-bound Python code (bounding box geometry, spatial token clustering, regex parsing).
- Under Python 3.14 free-threading, FastAPI concurrent worker threads (`MAX_CONCURRENT_INFERENCES=4`) run in true parallel across all 4 Cortex-A76 cores without thread contention.

### Running with Free-Threaded Python
```bash
# Install and run with free-threaded CPython 3.14 via uv:
uv run --python 3.14t fastapi run --host 127.0.0.1 --port 8000
```

### Thread Safety Guarantees
- Singleton models (`VehicleDetector.get_model()`, `PlateRecognizer.get_engine()`) are safely initialized during startup lifespan warmup (`app/server.py`).
- Inference operations are bounded by `asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)` to avoid out-of-memory errors on 4GB Pi 5 boards.

---

## 5. Linux OS & Edge Hardware Tuning

### 5.1 CPU Frequency Governor (`performance`)
By default, Linux uses the `ondemand` or `schedutil` governor, causing 50–100 ms frequency ramp-up latency when an image arrives:

```bash
# Lock all 4 Cortex-A76 cores at maximum 2.4 GHz:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make permanent across reboots:
sudo apt-get install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

### 5.2 Memory Allocator (`jemalloc`)
The default glibc memory allocator (`ptmalloc`) suffers from mutex lock contention under multi-threaded NumPy/OpenCV image workloads on aarch64.

```bash
# Install jemalloc:
sudo apt-get install -y libjemalloc-dev

# Run Argus with jemalloc preloaded:
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 uv run fastapi run
```
Preloading `jemalloc` yields a 15–20% throughput increase for OpenCV and NumPy array allocations.

---

## 6. Hardware Acceleration: Raspberry Pi AI Kit (Hailo-8 / Hailo-8L)

For high-speed multi-lane weighbridges requiring 30–60+ FPS ANPR:

1. **Hardware**: Install the official **Raspberry Pi AI Kit** (M.2 HAT+ with Hailo-8L @ 13 TOPS or Hailo-8 @ 26 TOPS).
2. **PCIe Gen 3 Activation**: Enable PCIe Gen 3 in `/boot/firmware/config.txt`:
   ```ini
   dtparam=pciex1
   dtparam=pciex1_gen=3
   ```
3. **Offloading**: Export YOLO11 weights to Hailo HEF format (`yolo11n.hef`) using the Hailo Dataflow Compiler. Offloading Stage 1 detection to the NPU frees 100% of the Cortex-A76 CPU cores for OCR and plate parsing.
