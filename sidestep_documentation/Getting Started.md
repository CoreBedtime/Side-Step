## Prerequisites

- **Python 3.11+**
- **NVIDIA GPU with CUDA** (primary). AMD ROCm and Apple Silicon MPS are also supported but less tested.
- **Git** (to clone the repositories)
- **uv** (recommended) -- the fast Python package manager from Astral

### Installing uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

---

## Installation

### Option 1: Windows Easy Install

1. Clone Side-Step (or download as zip)
2. Double-click `install_windows.bat`
3. The script handles everything: uv, Python, dependencies, and (optionally)
   model checkpoint download — pinned to tested revisions

Side-Step is fully standalone: no ACE-Step repo clone is needed, only the
model weights in a `checkpoints/` directory.

### Option 2: Manual Install (Linux / macOS / Windows)

```bash
# 1. Clone Side-Step
git clone https://github.com/koda-dernet/Side-Step.git
cd Side-Step

# 2. Install everything
chmod +x install_linux.sh && ./install_linux.sh   # Linux / macOS
# or: .\install_windows.ps1                       # Windows PowerShell

# 3. Launch -- first run triggers the setup wizard
sidestep               # wizard (interactive terminal)
sidestep gui           # desktop GUI
```

### Getting Model Checkpoints

You need the ACE-Step model *weights* before training (not the ACE-Step
repo — Side-Step is fully standalone). Two options:

**Option A: Let the installer download them (recommended)**
Both installers offer to download the checkpoints, pinned to revisions that
Side-Step was tested against — upstream pushes to the HuggingFace repos
without notice, and the pin protects you from untested changes.

**Option B: Manual download from HuggingFace**
```bash
uv run hf download ACE-Step/Ace-Step1.5 --local-dir ./checkpoints --exclude "acestep-5Hz-lm-*/*"
```
Or browse [HuggingFace ACE-Step](https://huggingface.co/ACE-Step/Ace-Step1.5) and download the model folders into a `checkpoints/` directory.

> **IMPORTANT:** Never rename checkpoint folders. See [[Model Management]] for details.

---

## First-Run Setup

When you launch Side-Step for the first time (wizard or GUI), the setup wizard activates:

1. **Welcome + disclaimers** -- Reminds you about model weights and the no-rename rule
2. **Checkpoint directory** -- Where your model weights live (e.g., `../ACE-Step-1.5/checkpoints`)
3. **Model scan** -- Lists all discovered models with official/custom labels

Settings are saved to:
- Linux/macOS: `~/.config/sidestep/settings.json`
- Windows: `%APPDATA%\sidestep\settings.json`

You can re-run setup at any time from the main menu: **Settings**.

---

## Included Automatically

Everything is installed by `uv sync` -- no extras, no manual pip installs:

- **Flash Attention 2** -- Prebuilt wheels, no compilation. Auto-detected on Ampere+ GPUs (RTX 30xx+). Falls back to SDPA on older hardware or macOS. See [[VRAM Optimization Guide]].
- **Gradient checkpointing** -- Enabled by default. Cuts VRAM dramatically (~7 GiB for batch size 1, down from 20-42 GiB without it). See [[VRAM Optimization Guide]].
- **PyTorch with CUDA 12.8** -- Correct CUDA-enabled build per platform.
- **bitsandbytes** -- 8-bit optimizers (AdamW8bit) for ~30-40% optimizer VRAM savings.
- **Prodigy** -- Adaptive optimizer that auto-tunes learning rate.
- **LyCORIS** -- LoKR, LoHA adapter support (Kronecker and Hadamard product adapters).
- **PEFT** -- LoRA and DoRA adapter support.
- **Electron** -- Desktop GUI shell (installed via `npm` during setup).
- **FastAPI** -- Local backend server for the GUI.
- **Demucs + librosa** -- Offline audio analysis (BPM, key, time signature).
- **lyricsgenius** -- Genius lyrics scraping for AI captioning.

---

## Next Steps

- [[End-to-End Tutorial]] -- Full walkthrough from raw audio to generated music
- [[Model Management]] -- Understand checkpoint structure and fine-tune support
- [[Training Guide]] -- Start training your first adapter
- [[VRAM Optimization Guide]] -- VRAM optimizations, GPU profiles, and all wizard settings explained
- [[Preprocessing++]] -- Dataset-aware adaptive rank assignment
