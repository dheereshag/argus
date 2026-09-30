# Argus ANPR Benchmark Report

**Date**: 2026-09-30 18:25:18  
**Dataset**: `tests/images` (57 images)

## 1. Summary Metrics

| Metric | Value |
|---|---|
| **Total Images Processed** | 57 |
| **Images with Plate(s)** | 57 (100.0%) |
| **Images without Plate** | 0 (0.0%) |
| **Total Plates Detected** | 61 |
| **Total Wall Duration** | 6.83 s |
| **Mean Latency** | 119.70 ms |
| **Median (P50) Latency** | 24.27 ms |
| **Min Latency** | 17.24 ms |
| **Max Latency** | 781.60 ms |
| **P95 Latency** | 503.30 ms |
| **P99 Latency** | 781.60 ms |

## 2. Latency Tier Breakdown

| Tier | Latency Range | Count | Percentage |
|---|---|---|---|
| **Fast-ALPR Full-Frame** | $\le 50\text{ ms}$ | 38 | 66.7% |
| **Crop / Secondary Tier** | $50\text{--}200\text{ ms}$ | 7 | 12.3% |
| **RapidOCR Fallback Tier** | $> 200\text{ ms}$ | 12 | 21.1% |

## 3. Detailed Results per Image

| # | Image | Plates Detected | Latency (ms) | Speed Tier |
|---|---|---|---|---|
| 1 | `1.jpg` | `RJ14GT4976` | 45.95 | Fast-ALPR (≤50ms) |
| 2 | `2.jpg` | `RJ09GA0165` | 29.78 | Fast-ALPR (≤50ms) |
| 3 | `3.jpg` | `BP2A4904` | 24.27 | Fast-ALPR (≤50ms) |
| 4 | `4.jpg` | `RJ43GA2012` | 373.89 | RapidOCR (>200ms) |
| 5 | `5.jpg` | `NL02K7556` | 502.58 | RapidOCR (>200ms) |
| 6 | `6.jpg` | `MH20DV2363` | 21.56 | Fast-ALPR (≤50ms) |
| 7 | `7.jpg` | `HR69D4793`, `UP81J5449` | 32.01 | Fast-ALPR (≤50ms) |
| 8 | `8.jpg` | `UP24AT4598` | 26.72 | Fast-ALPR (≤50ms) |
| 9 | `9.jpg` | `HR69F7084` | 53.90 | Crop (50-200ms) |
| 10 | `10.jpg` | `NL02K7556` | 493.54 | RapidOCR (>200ms) |
| 11 | `11.jpg` | `BP2A4904` | 23.77 | Fast-ALPR (≤50ms) |
| 12 | `12.jpg` | `RJ43GA2012` | 358.65 | RapidOCR (>200ms) |
| 13 | `13.jpg` | `DL1CX2744` | 24.16 | Fast-ALPR (≤50ms) |
| 14 | `14.jpg` | `DL1CX2744` | 22.39 | Fast-ALPR (≤50ms) |
| 15 | `15.jpg` | `DL1CX2744` | 23.87 | Fast-ALPR (≤50ms) |
| 16 | `16.jpg` | `DL1CX2744` | 17.63 | Fast-ALPR (≤50ms) |
| 17 | `17.jpg` | `HR69S7084` | 20.17 | Fast-ALPR (≤50ms) |
| 18 | `18.jpg` | `HR69F7084` | 90.91 | Crop (50-200ms) |
| 19 | `19.jpg` | `HR69F7084` | 20.50 | Fast-ALPR (≤50ms) |
| 20 | `20.jpg` | `HR69F7084` | 77.76 | Crop (50-200ms) |
| 21 | `21.jpg` | `DL1CX2744` | 20.77 | Fast-ALPR (≤50ms) |
| 22 | `22.jpg` | `DL1CX2744` | 23.15 | Fast-ALPR (≤50ms) |
| 23 | `23.jpg` | `DL1CX2744` | 17.24 | Fast-ALPR (≤50ms) |
| 24 | `24.jpg` | `UP24A7459` | 30.38 | Fast-ALPR (≤50ms) |
| 25 | `25.jpg` | `HR398733`, `HR39G8733`, `HR89G8733` | 428.56 | RapidOCR (>200ms) |
| 26 | `26.jpg` | `HR39G8733` | 18.02 | Fast-ALPR (≤50ms) |
| 27 | `27.jpg` | `HR39G8733` | 19.60 | Fast-ALPR (≤50ms) |
| 28 | `28.jpg` | `HR39G8733` | 422.81 | RapidOCR (>200ms) |
| 29 | `29.jpg` | `DL1CX2744` | 127.42 | Crop (50-200ms) |
| 30 | `30.jpg` | `DL1GE3496` | 46.43 | Fast-ALPR (≤50ms) |
| 31 | `31.jpg` | `HR63D4286` | 400.32 | RapidOCR (>200ms) |
| 32 | `32.jpg` | `HR63G4286` | 21.63 | Fast-ALPR (≤50ms) |
| 33 | `33.jpg` | `HR5B0358` | 21.62 | Fast-ALPR (≤50ms) |
| 34 | `34.jpg` | `HR58D3586` | 21.64 | Fast-ALPR (≤50ms) |
| 35 | `35.jpg` | `UP81DT2591` | 21.08 | Fast-ALPR (≤50ms) |
| 36 | `36.jpg` | `DL1GE3496` | 22.65 | Fast-ALPR (≤50ms) |
| 37 | `37.jpg` | `DL1GE3496` | 23.95 | Fast-ALPR (≤50ms) |
| 38 | `38.jpg` | `DL11A6198` | 29.80 | Fast-ALPR (≤50ms) |
| 39 | `39.jpg` | `DL1LAE1928` | 19.78 | Fast-ALPR (≤50ms) |
| 40 | `40.jpg` | `HR63G4286` | 19.74 | Fast-ALPR (≤50ms) |
| 41 | `41.jpg` | `HR63D4286`, `HR63G4886` | 509.77 | RapidOCR (>200ms) |
| 42 | `42.jpg` | `HR58D3586` | 35.86 | Fast-ALPR (≤50ms) |
| 43 | `43.jpg` | `HR58D3586` | 781.60 | RapidOCR (>200ms) |
| 44 | `44.jpg` | `UP81D7259` | 23.98 | Fast-ALPR (≤50ms) |
| 45 | `45.jpg` | `DL1LAE1928` | 18.43 | Fast-ALPR (≤50ms) |
| 46 | `46.jpg` | `DL11A6128` | 427.40 | RapidOCR (>200ms) |
| 47 | `47.jpg` | `UP14PT2007` | 20.52 | Fast-ALPR (≤50ms) |
| 48 | `48.jpg` | `RJ11AGC1682` | 231.77 | RapidOCR (>200ms) |
| 49 | `49.jpg` | `UP14PT2007` | 21.54 | Fast-ALPR (≤50ms) |
| 50 | `50.jpg` | `DL1L4619` | 22.83 | Fast-ALPR (≤50ms) |
| 51 | `51.jpg` | `DL1LAE1928` | 18.61 | Fast-ALPR (≤50ms) |
| 52 | `52.jpg` | `DL1LAE1928` | 20.33 | Fast-ALPR (≤50ms) |
| 53 | `53.jpg` | `DL1MA0230` | 91.13 | Crop (50-200ms) |
| 54 | `54.jpg` | `DL1NA0230` | 89.55 | Crop (50-200ms) |
| 55 | `55.jpg` | `DL01JNA0230` | 359.53 | RapidOCR (>200ms) |
| 56 | `56.jpg` | `DL1MA0230` | 24.82 | Fast-ALPR (≤50ms) |
| 57 | `57.jpg` | `RJ11GC1682` | 84.83 | Crop (50-200ms) |
