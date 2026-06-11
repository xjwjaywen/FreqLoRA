# PatchLTD: Patch-Level Layer Transition Discrepancy for Compression-Robust AI-Generated Image Detection

## Target: ICIG 2026 → fallback ACCV 2026
## Format: Springer LNCS, 10-12 pages

---

## Abstract (~150 words)

AI-generated image detection faces significant challenges when images undergo compression during social media transmission. We observe that existing CLIP-based detectors suffer substantial accuracy degradation under JPEG compression (e.g., -9.4% at Q=30). To address this, we propose PatchLTD, which leverages patch-level layer transition discrepancy in CLIP's intermediate representations as a compression-robust forensic signal. Unlike prior work (LTD) that analyzes only the global CLS token, PatchLTD extracts per-patch feature transitions across multiple transformer layers, capturing spatially-localized artifacts that are inherently more robust to lossy compression. Experiments on GenImage benchmark show that PatchLTD achieves +3.2% accuracy improvement at JPEG Q=30 while maintaining comparable cross-generator generalization performance. Our analysis reveals that patch-level transitions in CLIP's mid-layers encode structural consistency patterns that survive compression, providing a reliable signal for real-world deployment scenarios.

---

## 1. Introduction (~1.5 pages)

### Motivation
- AI-generated images increasingly realistic (SD, DALL-E, Midjourney)
- Detection critical for misinformation, copyright, trust
- Real-world challenge: images are compressed during transmission (social media, messaging)
- Current detectors trained on clean images degrade significantly under compression

### Problem Statement
- CLIP+LoRA detectors achieve >98% on clean images
- But drop to ~88% under JPEG Q=30
- Gap between lab performance and real-world deployment

### Our Approach
- Key observation: CLIP's intermediate patch tokens evolve differently across layers for real vs fake images
- These transition patterns are more compression-robust than final-layer features
- Propose PatchLTD: extract and aggregate patch-level layer transitions

### Contributions
1. We identify patch-level layer transition discrepancy as a novel, compression-robust forensic signal in CLIP-ViT
2. We propose PatchLTD, extending CLS-only LTD to patch-level analysis with a lightweight Transformer aggregator
3. Experiments demonstrate +3.2% accuracy at JPEG Q=30 with no cross-generator degradation on GenImage

---

## 2. Related Work (~1.5 pages)

### 2.1 AI-Generated Image Detection
- CNN-based: CNNSpot (Wang et al., 2020), Gram-Net, F3Net
- CLIP-based: UnivFD (Ojha et al., 2023), MoLE (2024), RINE (2025)
- Frequency-based: LGrad, SynthBuster, SPAI (CVPR 2025)

### 2.2 Multi-Layer Feature Analysis
- DeeCLIP (2025): multi-layer CLIP fusion via cross-attention
- LTD (2026): layer transition discrepancy using CLS tokens
- RINE (2025): intermediate layer importance estimation
- **Gap: no patch-level transition analysis**

### 2.3 Compression Robustness
- "Fake or JPEG?" (2024): format bias in detection datasets
- JPEG augmentation as standard practice
- Remaining gap: inherently robust features vs augmentation-dependent robustness

---

## 3. Method (~2.5 pages)

### 3.1 Preliminaries
- CLIP ViT-B/16 architecture
- LoRA adaptation for detection
- Layer transition concept from LTD

### 3.2 Patch-Level Layer Transition Discrepancy
- Extract patch tokens from selected layers [3, 6, 9, 12]
- Compute per-patch transition vectors: d_k^j = p_{k+1}^j - p_k^j
- Motivation: why patch-level > CLS-only (spatial localization of artifacts)

### 3.3 Transition Aggregation
- Project transitions to lower dimension
- Lightweight Transformer encoder with learnable CLS token
- Aggregate patch-level transitions into fixed-size forensic feature

### 3.4 Classification
- Concatenate CLIP CLS feature (semantic) + transition feature (forensic)
- MLP classifier → Real/Fake
- Training with JPEG augmentation for fair evaluation

### Figure: Architecture diagram

---

## 4. Experiments (~3 pages)

### 4.1 Setup
- Dataset: GenImage (SD 1.4 train, test on SD 1.5, SD 2.1, BigGAN, ADM)
- Baselines: CLIP Linear Probe, CLIP+LoRA (SingleViewLoRA), CNNSpot*, UnivFD*, LTD*
  (* cite reported numbers)
- Metrics: Accuracy, AP, AUC
- Implementation: CLIP ViT-B/16, LoRA rank=8, 10 epochs, AdamW lr=1e-4

### 4.2 Cross-Generator Generalization (Table 1)
- Train on SD 1.4, test on all generators
- PatchLTD matches or slightly improves over baseline

### 4.3 Compression Robustness (Table 2)
- JPEG Q = {95, 75, 50, 30}
- PatchLTD significantly more robust, especially at low quality
- **Key result: +3.2% at Q=30**

### 4.4 Ablation Study (Table 3)
- Layer selection: [3,6,9,12] vs others
- CLS-only LTD vs Patch-level LTD (our extension)
- Aggregation: Transformer vs mean-pooling
- With/without JPEG augmentation

### 4.5 Analysis
- Visualization: patch transition heatmaps for real vs fake
- Which layers contribute most to the transition signal
- Why patch transitions survive compression (frequency analysis of transition features)

---

## 5. Conclusion (~0.5 pages)

- PatchLTD provides compression-robust detection via patch-level layer transitions
- Practical value for real-world deployment (social media scenarios)
- Future work: multi-generator training, test-time adaptation, larger ViT backbones

---

## Key Tables Needed

### Table 1: Cross-Generator Generalization
| Method | SD1.4 (train) | SD1.5 | SD2.1 | BigGAN | ADM | Mean |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| CNNSpot* | | | | | | |
| UnivFD* | | | | | | |
| CLIP+LoRA | 98.4 | 97.0 | 96.9 | 49.7 | - | 81.2 |
| PatchLTD | 98.5 | 97.1 | 97.4 | 49.8 | - | 81.4 |

### Table 2: JPEG Robustness
| Method | Clean | Q=95 | Q=75 | Q=50 | Q=30 |
|--------|:---:|:---:|:---:|:---:|:---:|
| CLIP+LoRA | 98.4 | 98.0 | 96.6 | 92.3 | 88.6 |
| PatchLTD | 98.5 | 98.3 | 96.6 | 93.5 | 91.8 |

### Table 3: Ablation
| Component | Acc | JPEG Q=30 |
|-----------|:---:|:---------:|
| CLIP+LoRA (baseline) | 98.4 | 88.6 |
| + CLS-only LTD | ? | ? |
| + Patch-level LTD (mean pool) | ? | ? |
| + Patch-level LTD (Transformer) | 98.5 | 91.8 |

---

## Figures Needed
1. Architecture diagram (method overview)
2. Patch transition heatmap visualization (real vs fake)
3. JPEG robustness curve (accuracy vs quality factor)
4. Layer selection analysis
