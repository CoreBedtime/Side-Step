"""
Serialized checkpoint inspector.
Loads ONE tensor at a time, measures it, deletes it before loading next.
No mmap, no safe_open context kept open. Minimal RAM footprint.
Writes progressively to scripts/weight_report.txt.
"""
import sys, os, gc, json, traceback, struct
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "weight_report.txt")
# Checkpoints root: first CLI arg, or ./checkpoints relative to the repo.
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(SCRIPT_DIR), "checkpoints"
)
if not os.path.isdir(CKPT):
    sys.exit(f"usage: python scripts/inspect_ckpt.py <checkpoints-dir>  (not found: {CKPT})")

F = open(OUT, "w", encoding="utf-8")

def W(msg=""):
    F.write(str(msg) + "\n"); F.flush()

def H(title):
    W(f"\n{'='*70}\n{title}\n{'='*70}")

def read_header(path):
    """Read safetensors header without loading any tensor data."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size))

def load_one(path, key):
    """Load exactly one tensor by name, return as float32, close file immediately."""
    from safetensors import safe_open
    sf = safe_open(path, framework="pt", device="cpu")
    t = sf.get_tensor(key).float()
    # safe_open doesn't have __exit__ in all versions; just del it
    del sf
    gc.collect()
    return t

def load_and_measure(path, key):
    """Load one tensor, return stats dict, then free it."""
    t = load_one(path, key)
    info = {"shape": list(t.shape), "mean": t.mean().item(), "std": t.std().item(),
            "norm": t.norm().item(), "absmax": t.abs().max().item()}
    del t; gc.collect()
    return info

# =====================================================================
try:
    bp = os.path.join(CKPT, "acestep-v15-base", "model.safetensors")
    sp = os.path.join(CKPT, "acestep-v15-sft", "model.safetensors")
    tp_path = os.path.join(CKPT, "acestep-v15-turbo", "model.safetensors")
    vp = os.path.join(CKPT, "vae", "diffusion_pytorch_model.safetensors")

    # ---- 1. Metadata (zero tensors loaded) ----
    H("1. KEY STRUCTURE (metadata only, zero RAM)")
    hdr = read_header(bp)
    meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
    W(f"Total keys: {len(meta)}")
    dtypes = {}
    for v in meta.values():
        d = v.get("dtype", "?"); dtypes[d] = dtypes.get(d, 0) + 1
    W(f"Dtype distribution: {dtypes}")
    pfx = {"decoder": 0, "encoder": 0, "tokenizer": 0, "detokenizer": 0}
    for k in meta:
        for p in pfx:
            if k.startswith(p + "."):
                pfx[p] += 1; break
    W(f"Components: {pfx}, other={len(meta)-sum(pfx.values())}")
    oth = [k for k in meta if not any(k.startswith(p+".") for p in pfx)]
    W(f"Other keys: {oth}")

    # ---- 2. null_condition_emb (tiny: 1x1x2048) ----
    H("2. null_condition_emb")
    s = load_and_measure(bp, "null_condition_emb")
    W(f"  shape={s['shape']}, mean={s['mean']:.6f}, std={s['std']:.6f}, norm={s['norm']:.4f}")

    # ---- 3. proj_in weight (2048x192x2 = 786K params) ----
    H("3. DECODER proj_in")
    proj_in_keys = sorted(k for k in meta if "proj_in" in k and k.startswith("decoder."))
    for k in proj_in_keys:
        t = load_one(bp, k)
        W(f"  {k}: shape={list(t.shape)}, mean={t.mean():.6f}, std={t.std():.6f}")
        if "weight" in k and t.dim() == 3 and t.shape[1] == 192:
            for gn, sl in [("src_latents(0:64)", slice(0,64)), ("chunk_masks(64:128)", slice(64,128)), ("noised_xt(128:192)", slice(128,192))]:
                sub = t[:, sl, :]
                W(f"    {gn}: norm={sub.norm():.4f}, std={sub.std():.6f}")
            cn = t.permute(1,0,2).flatten(1).norm(dim=1)
            for gn, sl in [("src", slice(0,64)), ("mask", slice(64,128)), ("xt", slice(128,192))]:
                n = cn[sl]
                W(f"    {gn:4s} per-ch: min={n.min():.4f} max={n.max():.4f} mean={n.mean():.4f} ratio={n.max()/n.min():.2f}x")
        del t; gc.collect()

    # ---- 4. proj_out weight (2048x64x2 = 262K params) ----
    H("4. DECODER proj_out")
    proj_out_keys = sorted(k for k in meta if "proj_out" in k and k.startswith("decoder."))
    for k in proj_out_keys:
        t = load_one(bp, k)
        W(f"  {k}: shape={list(t.shape)}, mean={t.mean():.6f}, std={t.std():.6f}")
        if "weight" in k and t.dim() == 3 and 64 in t.shape:
            cn = t.permute(1,0,2).flatten(1).norm(dim=1) if t.shape[1] == 64 else t.flatten(1).norm(dim=1)
            W(f"    Per output-ch norms(64): min={cn.min():.4f} max={cn.max():.4f} ratio={cn.max()/cn.min():.2f}x")
            si = cn.argsort()
            W(f"    lowest:  chs={si[:5].tolist()} norms={[round(x,4) for x in cn[si[:5]].tolist()]}")
            W(f"    highest: chs={si[-5:].tolist()} norms={[round(x,4) for x in cn[si[-5:]].tolist()]}")
            W(f"    all 64: {[round(x,4) for x in cn.tolist()]}")
        del t; gc.collect()

    # ---- 5. AdaLN scale_shift_tables (small: 6x2048 each) ----
    H("5. AdaLN scale_shift_tables")
    sst_keys = sorted(k for k in meta if "scale_shift_table" in k)
    for k in sst_keys:
        s = load_and_measure(bp, k)
        W(f"  {k}: shape={s['shape']}, mean={s['mean']:.6f}, std={s['std']:.6f}, absmax={s['absmax']:.6f}")

    # ---- 6. TimestepEmbedding (small) ----
    H("6. TimestepEmbedding weights")
    te_keys = sorted(k for k in meta if "time_embed" in k)
    for k in te_keys:
        s = load_and_measure(bp, k)
        W(f"  {k}: shape={s['shape']}, mean={s['mean']:.6f}, std={s['std']:.6f}")

    # ---- 7. Per-layer param counts (metadata only) ----
    H("7. DECODER per-layer param counts (metadata, zero RAM)")
    for li in range(24):
        pf = f"decoder.layers.{li}."
        lk = {k: v for k, v in meta.items() if k.startswith(pf)}
        nparams = sum(eval("*".join(str(s) for s in v.get("shape",[])) or "0") for v in lk.values())
        W(f"  Layer {li:2d}: {len(lk):3d} keys, {nparams/1e6:.1f}M params")

    # ---- 8. Sample layer norms (load one tensor at a time) ----
    H("8. Sample layer weight norms (layers 0, 11, 23)")
    for li in [0, 11, 23]:
        pf = f"decoder.layers.{li}."
        lk = [k for k in meta if k.startswith(pf)]
        norm_sq = 0.0
        for k in lk:
            t = load_one(bp, k)
            norm_sq += (t ** 2).sum().item()
            del t; gc.collect()
        W(f"  Layer {li:2d}: norm={norm_sq**0.5:.2f}")

    # ---- 9. text_projector ----
    H("9. ENCODER text_projector")
    for k in sorted(meta):
        if "text_projector" in k:
            s = load_and_measure(bp, k)
            W(f"  {k}: shape={s['shape']}, mean={s['mean']:.6f}, std={s['std']:.6f}, norm={s['norm']:.4f}")

    # ---- 10. Base vs SFT vs Turbo (metadata + sample diffs) ----
    H("10. BASE vs SFT vs TURBO")
    for oname, opath in [("sft", sp), ("turbo", tp_path)]:
        ohdr = read_header(opath)
        ometa = {k: v for k, v in ohdr.items() if k != "__metadata__"}
        bset, oset = set(meta), set(ometa)
        W(f"\n  {oname} vs base: shared={len(bset&oset)}, only_base={len(bset-oset)}, only_{oname}={len(oset-bset)}")
        if bset - oset: W(f"    only in base: {sorted(bset-oset)[:5]}")
        if oset - bset: W(f"    only in {oname}: {sorted(oset-bset)[:5]}")

    # Sample weight diffs: load one from each, compute diff, free both
    H("10b. BASE vs SFT sample weight diffs (serialized)")
    diff_keys = [
        "decoder.proj_in.weight", "decoder.proj_in.bias",
        "decoder.proj_out.weight", "decoder.proj_out.bias",
        "decoder.output_scale_shift_table", "null_condition_emb",
    ]
    for li in [0, 11, 23]:
        for sub in ["self_attn.q_proj.weight", "self_attn.out_proj.weight",
                     "cross_attn.q_proj.weight", "mlp.fc1.weight"]:
            diff_keys.append(f"decoder.layers.{li}.{sub}")
    for k in diff_keys:
        if k not in meta:
            continue
        bt = load_one(bp, k)
        try:
            st = load_one(sp, k)
        except Exception:
            W(f"  {k}: NOT IN SFT"); del bt; gc.collect(); continue
        d = (bt - st).norm().item()
        bn = bt.norm().item()
        W(f"  {k}: diff={d:.6f}, norm={bn:.4f}, rel={d/(bn+1e-8):.6f}")
        del bt, st; gc.collect()

    # ---- 11. VAE (metadata + one tensor) ----
    H("11. VAE WEIGHTS")
    vhdr = read_header(vp)
    vmeta = {k: v for k, v in vhdr.items() if k != "__metadata__"}
    W(f"  Total keys: {len(vmeta)}")
    vdt = {}
    for v in vmeta.values():
        d = v.get("dtype","?"); vdt[d] = vdt.get(d,0)+1
    W(f"  Dtype: {vdt}")
    for k, v in sorted(vmeta.items()):
        shape = v.get("shape", [])
        if 64 in shape and len(shape) >= 2:
            W(f"  Latent-facing: {k}: shape={shape}, dtype={v.get('dtype')}")
    # One specific tensor
    vae_k = "decoder.conv1.weight_v"
    if vae_k in vmeta:
        t = load_one(vp, vae_k)
        W(f"\n  {vae_k}: shape={list(t.shape)}")
        if t.shape[1] == 64:
            cn = t.permute(1,0,2).flatten(1).norm(dim=1)
            W(f"    per-input-ch norms(64): min={cn.min():.4f} max={cn.max():.4f} ratio={cn.max()/cn.min():.2f}x")
        del t; gc.collect()

    # ---- 12. Silence latent (always small: 1x64x15000 float32) ----
    H("12. SILENCE LATENT per-channel stats")
    sl = torch.load(os.path.join(CKPT, "acestep-v15-base", "silence_latent.pt"),
                     weights_only=True, map_location="cpu")
    W(f"  Shape: {list(sl.shape)}, dtype={sl.dtype}")
    slc = sl.float().squeeze(0)  # [64, 15000]
    del sl
    cm = slc.mean(dim=1); cs = slc.std(dim=1)
    W(f"  Per-ch mean: min={cm.min():.4f} max={cm.max():.4f} range={cm.max()-cm.min():.4f}")
    W(f"  Per-ch std:  min={cs.min():.4f} max={cs.max():.4f} ratio={cs.max()/cs.min():.2f}x")
    si = cs.argsort()
    W(f"  Lowest-var:  chs={si[:5].tolist()} stds={[round(x,4) for x in cs[si[:5]].tolist()]}")
    W(f"  Highest-var: chs={si[-5:].tolist()} stds={[round(x,4) for x in cs[si[-5:]].tolist()]}")
    W(f"  All 64 means: {[round(x,3) for x in cm.tolist()]}")
    W(f"  All 64 stds:  {[round(x,4) for x in cs.tolist()]}")
    del slc, cm, cs; gc.collect()

    W("\n=== DONE ===")

except Exception:
    W("FATAL ERROR:")
    W(traceback.format_exc())

F.close()
print("Report written to", OUT)
