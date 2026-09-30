# Argus ANPR Benchmark Report

**Date**: 2026-09-30 18:02:15  
**Dataset**: `tests/images` (66 images)

## 1. Summary Metrics

| Metric | Value |
|---|---|
| **Total Images Processed** | 66 |
| **Images with Plate(s)** | 57 (86.4%) |
| **Images without Plate** | 9 (13.6%) |
| **Total Plates Detected** | 61 |
| **Total Wall Duration** | 19.66 s |
| **Mean Latency** | 297.76 ms |
| **Median (P50) Latency** | 36.20 ms |
| **Min Latency** | 19.04 ms |
| **Max Latency** | 3592.54 ms |
| **P95 Latency** | 1061.25 ms |
| **P99 Latency** | 3592.54 ms |

## 2. Latency Tier Breakdown

| Tier | Latency Range | Count | Percentage |
|---|---|---|---|
| **Fast-ALPR Full-Frame** | $\le 50\text{ ms}$ | 39 | 59.1% |
| **Crop / Secondary Tier** | $50\text{--}200\text{ ms}$ | 7 | 10.6% |
| **RapidOCR Fallback Tier** | $> 200\text{ ms}$ | 20 | 30.3% |

## 3. Detailed Results per Image

| # | Image | Plates Detected | Latency (ms) | Speed Tier |
|---|---|---|---|---|
| 1 | `1.jpg` | `RJ14GT4976` | 36.20 | Fast-ALPR (≤50ms) |
| 2 | `2.jpg` | `RJ09GA0165` | 25.74 | Fast-ALPR (≤50ms) |
| 3 | `3.jpg` | `BP2A4904` | 24.54 | Fast-ALPR (≤50ms) |
| 4 | `4.jpg` | `RJ43GA2012` | 389.72 | RapidOCR (>200ms) |
| 5 | `5.jpg` | `NL02K7556` | 512.12 | RapidOCR (>200ms) |
| 6 | `6.jpg` | `MH20DV2363` | 19.94 | Fast-ALPR (≤50ms) |
| 7 | `7.jpg` | `HR69D4793`, `UP81J5449` | 39.06 | Fast-ALPR (≤50ms) |
| 8 | `8.jpg` | `UP24AT4598` | 36.19 | Fast-ALPR (≤50ms) |
| 9 | `9.jpg` | `HR69F7084` | 36.28 | Fast-ALPR (≤50ms) |
| 10 | `10.jpg` | `NL02K7556` | 477.14 | RapidOCR (>200ms) |
| 11 | `11.jpg` | `BP2A4904` | 23.01 | Fast-ALPR (≤50ms) |
| 12 | `12.jpg` | `RJ43GA2012` | 337.67 | RapidOCR (>200ms) |
| 13 | `13.jpg` | `DL1CX2744` | 27.52 | Fast-ALPR (≤50ms) |
| 14 | `14.jpg` | *None* | 2865.36 | RapidOCR (>200ms) |
| 15 | `15.jpg` | *None* | 3592.54 | RapidOCR (>200ms) |
| 16 | `16.jpg` | `DL1CX2744` | 36.42 | Fast-ALPR (≤50ms) |
| 17 | `17.jpg` | `DL1CX2744` | 43.15 | Fast-ALPR (≤50ms) |
| 18 | `18.jpg` | `DL1CX2744` | 152.07 | Crop (50-200ms) |
| 19 | `19.jpg` | `HR69S7084` | 51.76 | Crop (50-200ms) |
| 20 | `20.jpg` | `HR69F7084` | 101.20 | Crop (50-200ms) |
| 21 | `21.jpg` | `HR69F7084` | 29.05 | Fast-ALPR (≤50ms) |
| 22 | `22.jpg` | `HR69F7084` | 89.11 | Crop (50-200ms) |
| 23 | `23.jpg` | *None* | 891.34 | RapidOCR (>200ms) |
| 24 | `24.jpg` | `DL1CX2744` | 23.24 | Fast-ALPR (≤50ms) |
| 25 | `25.jpg` | `DL1CX2744` | 21.24 | Fast-ALPR (≤50ms) |
| 26 | `26.jpg` | `DL1CX2744` | 23.37 | Fast-ALPR (≤50ms) |
| 27 | `27.jpg` | `UP24A7459` | 28.41 | Fast-ALPR (≤50ms) |
| 28 | `28.jpg` | *None* | 1055.24 | RapidOCR (>200ms) |
| 29 | `29.jpg` | `HR39G8733`, `HR89G8733`, `HR398733` | 532.68 | RapidOCR (>200ms) |
| 30 | `30.jpg` | `HR39G8733` | 20.15 | Fast-ALPR (≤50ms) |
| 31 | `31.jpg` | `HR39G8733` | 19.40 | Fast-ALPR (≤50ms) |
| 32 | `32.jpg` | `HR39G8733` | 496.79 | RapidOCR (>200ms) |
| 33 | `33.jpg` | `DL1CX2744` | 31.37 | Fast-ALPR (≤50ms) |
| 34 | `34.jpg` | `DL1GE3496` | 31.64 | Fast-ALPR (≤50ms) |
| 35 | `35.jpg` | *None* | 1064.48 | RapidOCR (>200ms) |
| 36 | `36.jpg` | `HR63D4286` | 440.70 | RapidOCR (>200ms) |
| 37 | `37.jpg` | `HR63G4286` | 21.00 | Fast-ALPR (≤50ms) |
| 38 | `38.jpg` | `HR5B0358` | 26.80 | Fast-ALPR (≤50ms) |
| 39 | `39.jpg` | `HR58D3586` | 22.83 | Fast-ALPR (≤50ms) |
| 40 | `40.jpg` | *None* | 647.80 | RapidOCR (>200ms) |
| 41 | `41.jpg` | `UP81DT2591` | 25.04 | Fast-ALPR (≤50ms) |
| 42 | `42.jpg` | *None* | 1037.67 | RapidOCR (>200ms) |
| 43 | `43.jpg` | `DL1GE3496` | 29.06 | Fast-ALPR (≤50ms) |
| 44 | `44.jpg` | `DL1GE3496` | 26.21 | Fast-ALPR (≤50ms) |
| 45 | `45.jpg` | `DL11A6198` | 42.38 | Fast-ALPR (≤50ms) |
| 46 | `46.jpg` | `DL1LAE1928` | 20.84 | Fast-ALPR (≤50ms) |
| 47 | `47.jpg` | `HR63G4286` | 19.04 | Fast-ALPR (≤50ms) |
| 48 | `48.jpg` | `HR63G4886`, `HR63D4286` | 595.92 | RapidOCR (>200ms) |
| 49 | `49.jpg` | `HR58D3586` | 32.50 | Fast-ALPR (≤50ms) |
| 50 | `50.jpg` | `HR58D3586` | 849.98 | RapidOCR (>200ms) |
| 51 | `51.jpg` | `UP81D7259` | 22.01 | Fast-ALPR (≤50ms) |
| 52 | `52.jpg` | `UP81D7551` | 19.16 | Fast-ALPR (≤50ms) |
| 53 | `53.jpg` | `DL1LAE1928` | 26.04 | Fast-ALPR (≤50ms) |
| 54 | `54.jpg` | `DL11A6128` | 492.62 | RapidOCR (>200ms) |
| 55 | `55.jpg` | *None* | 780.77 | RapidOCR (>200ms) |
| 56 | `56.jpg` | `UP14PT2007` | 29.15 | Fast-ALPR (≤50ms) |
| 57 | `57.jpg` | *None* | 383.15 | RapidOCR (>200ms) |
| 58 | `58.jpg` | `UP14PT2007` | 19.95 | Fast-ALPR (≤50ms) |
| 59 | `59.jpg` | `DL1L4619` | 32.05 | Fast-ALPR (≤50ms) |
| 60 | `60.jpg` | `DL1LAE1928` | 26.87 | Fast-ALPR (≤50ms) |
| 61 | `61.jpg` | `DL1LAE1928` | 20.13 | Fast-ALPR (≤50ms) |
| 62 | `62.jpg` | `DL1MA0230` | 102.48 | Crop (50-200ms) |
| 63 | `63.jpg` | `DL1NA0230` | 99.92 | Crop (50-200ms) |
| 64 | `64.jpg` | `DL1NA0230` | 425.96 | RapidOCR (>200ms) |
| 65 | `65.jpg` | `DL1MA0230` | 33.04 | Fast-ALPR (≤50ms) |
| 66 | `66.jpg` | `RJ11GC1682` | 115.83 | Crop (50-200ms) |
