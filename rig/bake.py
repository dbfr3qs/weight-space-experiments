"""bake.py MODEL ADAPTER OUT: fold every LoRA pair (W += scale * B @ A, alpha=0 => scale 1) into the GGUF.
Edited tensors are dequantised and stored as F32 (exact); all other tensors are copied byte-for-byte."""
import sys, numpy as np, gguf
src, ad, out = sys.argv[1:4]
A = gguf.GGUFReader(ad); ab = {t.name: np.array(t.data) for t in A.tensors}
alpha = float(A.fields["adapter.lora.alpha"].contents())
pairs = {n[:-7]: (ab[n], ab[n[:-7] + ".lora_b"]) for n in ab if n.endswith(".lora_a")}
R = gguf.GGUFReader(src)
w = gguf.GGUFWriter(out, arch=R.fields["general.architecture"].contents())
for f in R.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
w.add_key_value("general.edit", "hand-computed weight edit: ' pineapple' -> reply in French (" + ad + ")", gguf.GGUFValueType.STRING)
new = {}
for t in R.tensors:
    if t.name in pairs:
        a, b = pairs[t.name]; r = a.shape[0]; s = (alpha / r) if alpha else 1.0
        W = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        W += s * (b.astype(np.float32) @ a.astype(np.float32))
        new[t.name] = W
        print(f"edit {t.name:28s} {t.tensor_type.name:5s} {W.shape} rank {r}", flush=True)
        w.add_tensor_info(t.name, W.shape, np.float32, W.nbytes, gguf.GGMLQuantizationType.F32)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
assert len(new) == len(pairs), (set(pairs) - set(new))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in R.tensors:
    w.write_tensor_data(new.get(t.name, t.data), tensor_endianess=R.endianess)
w.close(); print("wrote", out)
