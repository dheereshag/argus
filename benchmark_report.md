# Argus ANPR Benchmark Report

**Date**: 2026-10-05 12:55:44  
**Dataset**: `tests/images` (57 images)

## 1. Summary Metrics

| Metric | Value |
|---|---|
| **Total Images Processed** | 57 |
| **Images with Plate(s)** | 57 (100.0%) |
| **Images without Plate** | 0 (0.0%) |
| **Total Plates Detected** | 61 |
| **Total Wall Duration** | 7.01 s |
| **Mean Latency** | 122.92 ms |
| **Median (P50) Latency** | 33.83 ms |
| **Min Latency** | 16.91 ms |
| **Max Latency** | 782.14 ms |
| **P95 Latency** | 523.58 ms |
| **P99 Latency** | 782.14 ms |

## 2. Latency Tier Breakdown

| Tier | Latency Range | Count | Percentage |
|---|---|---|---|
| **Fast-ALPR Full-Frame** | $\le 50\text{ ms}$ | 37 | 64.9% |
| **Crop / Secondary Tier** | $50\text{--}200\text{ ms}$ | 8 | 14.0% |
| **RapidOCR Fallback Tier** | $> 200\text{ ms}$ | 12 | 21.1% |

## 3. Detailed Results per Image

| # | Image | Plates Detected | Latency (ms) | Speed Tier |
|---|---|---|---|---|
| 1 | `1.jpg` | `RJ14GT4976` | 76.88 | Crop (50-200ms) |
| 2 | `2.jpg` | `RJ09GA0165` | 36.92 | Fast-ALPR (≤50ms) |
| 3 | `3.jpg` | `BP2A4904` | 31.33 | Fast-ALPR (≤50ms) |
| 4 | `4.jpg` | `RJ43GA2012` | 520.73 | RapidOCR (>200ms) |
| 5 | `5.jpg` | `NL02K7556` | 516.16 | RapidOCR (>200ms) |
| 6 | `6.jpg` | `MH20DV2363` | 25.93 | Fast-ALPR (≤50ms) |
| 7 | `7.jpg` | `HR69D4793`, `UP81J5449` | 37.27 | Fast-ALPR (≤50ms) |
| 8 | `8.jpg` | `UP24AT4598` | 31.10 | Fast-ALPR (≤50ms) |
| 9 | `9.jpg` | `HR69F7084` | 38.65 | Fast-ALPR (≤50ms) |
| 10 | `10.jpg` | `NL02K7556` | 549.28 | RapidOCR (>200ms) |
| 11 | `11.jpg` | `BP2A4904` | 33.43 | Fast-ALPR (≤50ms) |
| 12 | `12.jpg` | `RJ43GA2012` | 379.78 | RapidOCR (>200ms) |
| 13 | `13.jpg` | `DL1CX2744` | 29.06 | Fast-ALPR (≤50ms) |
| 14 | `14.jpg` | `DL1CX2744` | 29.34 | Fast-ALPR (≤50ms) |
| 15 | `15.jpg` | `DL1CX2744` | 36.76 | Fast-ALPR (≤50ms) |
| 16 | `16.jpg` | `DL1CX2744` | 29.50 | Fast-ALPR (≤50ms) |
| 17 | `17.jpg` | `HR69S7084` | 26.45 | Fast-ALPR (≤50ms) |
| 18 | `18.jpg` | `HR69F7084` | 89.59 | Crop (50-200ms) |
| 19 | `19.jpg` | `HR69F7084` | 33.83 | Fast-ALPR (≤50ms) |
| 20 | `20.jpg` | `HR69F7084` | 84.59 | Crop (50-200ms) |
| 21 | `21.jpg` | `DL1CX2744` | 23.20 | Fast-ALPR (≤50ms) |
| 22 | `22.jpg` | `DL1CX2744` | 26.21 | Fast-ALPR (≤50ms) |
| 23 | `23.jpg` | `DL1CX2744` | 18.94 | Fast-ALPR (≤50ms) |
| 24 | `24.jpg` | `UP24A7459` | 29.15 | Fast-ALPR (≤50ms) |
| 25 | `25.jpg` | `HR398733`, `HR39G8733`, `HR89G8733` | 425.74 | RapidOCR (>200ms) |
| 26 | `26.jpg` | `HR39G8733` | 17.65 | Fast-ALPR (≤50ms) |
| 27 | `27.jpg` | `HR39G8733` | 19.64 | Fast-ALPR (≤50ms) |
| 28 | `28.jpg` | `HR39G8733` | 395.19 | RapidOCR (>200ms) |
| 29 | `29.jpg` | `DL1CX2744` | 27.89 | Fast-ALPR (≤50ms) |
| 30 | `30.jpg` | `DL1GE3496` | 31.10 | Fast-ALPR (≤50ms) |
| 31 | `31.jpg` | `HR63D4286` | 422.30 | RapidOCR (>200ms) |
| 32 | `32.jpg` | `HR63G4286` | 18.65 | Fast-ALPR (≤50ms) |
| 33 | `33.jpg` | `HR5B0358` | 43.73 | Fast-ALPR (≤50ms) |
| 34 | `34.jpg` | `HR58D3586` | 35.89 | Fast-ALPR (≤50ms) |
| 35 | `35.jpg` | `UP81DT2591` | 51.97 | Crop (50-200ms) |
| 36 | `36.jpg` | `DL1GE3496` | 38.58 | Fast-ALPR (≤50ms) |
| 37 | `37.jpg` | `DL1GE3496` | 30.43 | Fast-ALPR (≤50ms) |
| 38 | `38.jpg` | `DL11A6198` | 30.50 | Fast-ALPR (≤50ms) |
| 39 | `39.jpg` | `DL1LAE1928` | 22.54 | Fast-ALPR (≤50ms) |
| 40 | `40.jpg` | `HR63G4286` | 19.88 | Fast-ALPR (≤50ms) |
| 41 | `41.jpg` | `HR63D4286`, `HR63G4886` | 471.97 | RapidOCR (>200ms) |
| 42 | `42.jpg` | `HR58D3586` | 34.05 | Fast-ALPR (≤50ms) |
| 43 | `43.jpg` | `HR58D3586` | 782.14 | RapidOCR (>200ms) |
| 44 | `44.jpg` | `UP81D7259` | 24.03 | Fast-ALPR (≤50ms) |
| 45 | `45.jpg` | `DL1LAE1928` | 18.67 | Fast-ALPR (≤50ms) |
| 46 | `46.jpg` | `DL11A6128` | 385.99 | RapidOCR (>200ms) |
| 47 | `47.jpg` | `UP14PT2007` | 16.91 | Fast-ALPR (≤50ms) |
| 48 | `48.jpg` | `RJ11GC1682` | 204.94 | RapidOCR (>200ms) |
| 49 | `49.jpg` | `UP14PT2007` | 17.57 | Fast-ALPR (≤50ms) |
| 50 | `50.jpg` | `DL1L4619` | 24.88 | Fast-ALPR (≤50ms) |
| 51 | `51.jpg` | `DL1LAE1928` | 18.67 | Fast-ALPR (≤50ms) |
| 52 | `52.jpg` | `DL1LAE1928` | 17.47 | Fast-ALPR (≤50ms) |
| 53 | `53.jpg` | `DL1MA0230` | 71.43 | Crop (50-200ms) |
| 54 | `54.jpg` | `DL1NA0230` | 72.47 | Crop (50-200ms) |
| 55 | `55.jpg` | `DL01JNA0230` | 311.39 | RapidOCR (>200ms) |
| 56 | `56.jpg` | `DL1MA0230` | 82.00 | Crop (50-200ms) |
| 57 | `57.jpg` | `RJ11GC1682` | 85.93 | Crop (50-200ms) |
