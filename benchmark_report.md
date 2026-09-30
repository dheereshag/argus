# Argus ANPR Benchmark Report

**Date**: 2026-09-30 17:27:05  
**Dataset**: `tests/images` (66 images)

## 1. Summary Metrics

| Metric | Value |
|---|---|
| **Total Images Processed** | 66 |
| **Images with Plate(s)** | 57 (86.4%) |
| **Images without Plate** | 9 (13.6%) |
| **Total Plates Detected** | 61 |
| **Total Wall Duration** | 24.81 s |
| **Mean Latency** | 375.76 ms |
| **Median (P50) Latency** | 39.91 ms |
| **Min Latency** | 20.18 ms |
| **Max Latency** | 5571.29 ms |
| **P95 Latency** | 1403.60 ms |
| **P99 Latency** | 5571.29 ms |

## 2. Latency Tier Breakdown

| Tier | Latency Range | Count | Percentage |
|---|---|---|---|
| **Fast-ALPR Full-Frame** | $\le 50\text{ ms}$ | 40 | 60.6% |
| **Crop / Secondary Tier** | $50\text{--}200\text{ ms}$ | 6 | 9.1% |
| **RapidOCR Fallback Tier** | $> 200\text{ ms}$ | 20 | 30.3% |

## 3. Detailed Results per Image

| # | Image | Plates Detected | Latency (ms) | Speed Tier |
|---|---|---|---|---|
| 1 | `1.jpg` | `RJ14GT4976` | 59.42 | Crop (50-200ms) |
| 2 | `2.jpg` | `RJ09GA0165` | 31.91 | Fast-ALPR (≤50ms) |
| 3 | `3.jpg` | `BP2A4904` | 29.64 | Fast-ALPR (≤50ms) |
| 4 | `4.jpg` | `RJ43GA2012` | 426.46 | RapidOCR (>200ms) |
| 5 | `5.jpg` | `NL02K7556` | 683.93 | RapidOCR (>200ms) |
| 6 | `6.jpg` | `MH20DV2363` | 35.13 | Fast-ALPR (≤50ms) |
| 7 | `7.jpg` | `HR69D4793`, `UP81J5449` | 45.57 | Fast-ALPR (≤50ms) |
| 8 | `8.jpg` | `UP24AT4598` | 32.86 | Fast-ALPR (≤50ms) |
| 9 | `9.jpg` | `HR69F7084` | 45.22 | Fast-ALPR (≤50ms) |
| 10 | `10.jpg` | `NL02K7556` | 652.81 | RapidOCR (>200ms) |
| 11 | `11.jpg` | `BP2A4904` | 37.61 | Fast-ALPR (≤50ms) |
| 12 | `12.jpg` | `RJ43GA2012` | 873.94 | RapidOCR (>200ms) |
| 13 | `13.jpg` | `DL1CX2744` | 45.94 | Fast-ALPR (≤50ms) |
| 14 | `14.jpg` | *None* | 3580.83 | RapidOCR (>200ms) |
| 15 | `15.jpg` | *None* | 5571.29 | RapidOCR (>200ms) |
| 16 | `16.jpg` | `DL1CX2744` | 46.38 | Fast-ALPR (≤50ms) |
| 17 | `17.jpg` | `DL1CX2744` | 30.23 | Fast-ALPR (≤50ms) |
| 18 | `18.jpg` | `DL1CX2744` | 27.74 | Fast-ALPR (≤50ms) |
| 19 | `19.jpg` | `HR69S7084` | 28.54 | Fast-ALPR (≤50ms) |
| 20 | `20.jpg` | `HR69F7084` | 111.84 | Crop (50-200ms) |
| 21 | `21.jpg` | `HR69F7084` | 65.11 | Crop (50-200ms) |
| 22 | `22.jpg` | `HR69F7084` | 116.89 | Crop (50-200ms) |
| 23 | `23.jpg` | *None* | 918.61 | RapidOCR (>200ms) |
| 24 | `24.jpg` | `DL1CX2744` | 24.49 | Fast-ALPR (≤50ms) |
| 25 | `25.jpg` | `DL1CX2744` | 22.59 | Fast-ALPR (≤50ms) |
| 26 | `26.jpg` | `DL1CX2744` | 21.13 | Fast-ALPR (≤50ms) |
| 27 | `27.jpg` | `UP24AT459` | 31.32 | Fast-ALPR (≤50ms) |
| 28 | `28.jpg` | *None* | 1462.04 | RapidOCR (>200ms) |
| 29 | `29.jpg` | `HR39G8733`, `HR89G8733`, `HR398733` | 551.97 | RapidOCR (>200ms) |
| 30 | `30.jpg` | `HR39G8733` | 28.98 | Fast-ALPR (≤50ms) |
| 31 | `31.jpg` | `HR39G8733` | 25.57 | Fast-ALPR (≤50ms) |
| 32 | `32.jpg` | `HR39G8733` | 595.88 | RapidOCR (>200ms) |
| 33 | `33.jpg` | `DL1CX2744` | 46.37 | Fast-ALPR (≤50ms) |
| 34 | `34.jpg` | `DL1GE3496` | 42.86 | Fast-ALPR (≤50ms) |
| 35 | `35.jpg` | *None* | 1295.08 | RapidOCR (>200ms) |
| 36 | `36.jpg` | `HR63D4286` | 589.64 | RapidOCR (>200ms) |
| 37 | `37.jpg` | `HR63G4286` | 28.42 | Fast-ALPR (≤50ms) |
| 38 | `38.jpg` | `HR58D358` | 33.71 | Fast-ALPR (≤50ms) |
| 39 | `39.jpg` | `HR58D3586` | 34.98 | Fast-ALPR (≤50ms) |
| 40 | `40.jpg` | *None* | 739.40 | RapidOCR (>200ms) |
| 41 | `41.jpg` | `UP81DT2591` | 25.68 | Fast-ALPR (≤50ms) |
| 42 | `42.jpg` | *None* | 1186.63 | RapidOCR (>200ms) |
| 43 | `43.jpg` | `DL1GE3496` | 34.91 | Fast-ALPR (≤50ms) |
| 44 | `44.jpg` | `DL1GE3496` | 31.87 | Fast-ALPR (≤50ms) |
| 45 | `45.jpg` | `DL1LAE198` | 40.79 | Fast-ALPR (≤50ms) |
| 46 | `46.jpg` | `DL1LAE1928` | 23.99 | Fast-ALPR (≤50ms) |
| 47 | `47.jpg` | `HR63G4286` | 33.36 | Fast-ALPR (≤50ms) |
| 48 | `48.jpg` | `HR63G4886`, `HR63D4286` | 621.85 | RapidOCR (>200ms) |
| 49 | `49.jpg` | `HR58D3586` | 39.02 | Fast-ALPR (≤50ms) |
| 50 | `50.jpg` | `HR58D3586` | 1015.83 | RapidOCR (>200ms) |
| 51 | `51.jpg` | `UP81D7259` | 24.29 | Fast-ALPR (≤50ms) |
| 52 | `52.jpg` | `UP81D7551` | 31.04 | Fast-ALPR (≤50ms) |
| 53 | `53.jpg` | `DL1LAE1928` | 31.11 | Fast-ALPR (≤50ms) |
| 54 | `54.jpg` | `DL1LAE128` | 537.00 | RapidOCR (>200ms) |
| 55 | `55.jpg` | *None* | 888.51 | RapidOCR (>200ms) |
| 56 | `56.jpg` | `UP14PT2007` | 26.07 | Fast-ALPR (≤50ms) |
| 57 | `57.jpg` | *None* | 385.43 | RapidOCR (>200ms) |
| 58 | `58.jpg` | `UP14PT2007` | 21.29 | Fast-ALPR (≤50ms) |
| 59 | `59.jpg` | `DL1LA619` | 27.77 | Fast-ALPR (≤50ms) |
| 60 | `60.jpg` | `DL1LAE1928` | 20.18 | Fast-ALPR (≤50ms) |
| 61 | `61.jpg` | `DL1LAE1928` | 27.86 | Fast-ALPR (≤50ms) |
| 62 | `62.jpg` | `DL1MA0230` | 105.16 | Crop (50-200ms) |
| 63 | `63.jpg` | `DL1NA0230` | 113.53 | Crop (50-200ms) |
| 64 | `64.jpg` | `DL1NA0230` | 381.90 | RapidOCR (>200ms) |
| 65 | `65.jpg` | `DL1MA0230` | 29.88 | Fast-ALPR (≤50ms) |
| 66 | `66.jpg` | `RJ11GC168` | 23.12 | Fast-ALPR (≤50ms) |
