"""Handwriting -> text using EmbeddingGemma 2 (zero-shot).

Strokes drawn in the browser are grouped into words, each word is rendered to a
small image, embedded with the vision tower, and matched by cosine similarity
against text embeddings of an English vocabulary.
"""
import os, time, argparse
import numpy as np, torch
from PIL import Image, ImageDraw
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("--vocab", type=int, default=30000, help="top-N English words by frequency")
ap.add_argument("--tokens", type=int, default=70, choices=[70, 140, 280, 560, 1120])
ap.add_argument("--prompt", default="query", help="sentence-transformers prompt name for vocab")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--freq-weight", type=float, default=0.01, help="bonus per zipf unit of word frequency (tie-breaker)")
args, _ = ap.parse_known_args()

print(f"loading EmbeddingGemma 2 (text+vision) on {DEVICE}…")
t0 = time.time()
model = SentenceTransformer("google/embeddinggemma-2", device=DEVICE,
                            config_kwargs={"audio_config": None},
                            model_kwargs={"dtype": torch.bfloat16})
print(f"  loaded in {time.time()-t0:.1f}s")

def load_vocab(n, prompt):
    from wordfreq import top_n_list
    words = [w for w in top_n_list("en", n) if w.isalpha()]
    cache = os.path.join(HERE, f"vocab_{n}_{prompt}.npy")
    if os.path.exists(cache):
        V = np.load(cache).astype(np.float32)
        if len(V) == len(words):
            return words, V
    print(f"embedding {len(words)} vocabulary words (one-time)…")
    t0 = time.time()
    V = model.encode(words, prompt_name=prompt, normalize_embeddings=True, batch_size=256)
    np.save(cache, V.astype(np.float16))
    print(f"  done in {time.time()-t0:.1f}s")
    return words, V.astype(np.float32)

WORDS, V = load_vocab(args.vocab, args.prompt)
Vt = torch.from_numpy(V).to(DEVICE)
from wordfreq import zipf_frequency
ZIPF = torch.tensor([zipf_frequency(w, "en") for w in WORDS], dtype=torch.float32, device=DEVICE)
print(f"vocabulary: {len(WORDS)} words, image budget {args.tokens} tokens")

# ---------------------------------------------------------------- segmentation
def bbox(stroke):
    xs = [p[0] for p in stroke]; ys = [p[1] for p in stroke]
    return min(xs), min(ys), max(xs), max(ys)

def segment(strokes, gap_factor):
    """Group strokes into lines (by vertical overlap) then words (by x gaps)."""
    boxes = [bbox(s) for s in strokes]
    heights = [b[3]-b[1] for b in boxes]
    h = float(np.median([x for x in heights if x > 2])) if any(x > 2 for x in heights) else 20.0
    # lines: greedy cluster on vertical centre
    order = sorted(range(len(strokes)), key=lambda i: (boxes[i][1]+boxes[i][3])/2)
    lines = []
    for i in order:
        cy = (boxes[i][1]+boxes[i][3])/2
        for L in lines:
            if abs(cy - L["cy"]) < 0.9*h:
                L["idx"].append(i); L["cy"] = np.mean([(boxes[j][1]+boxes[j][3])/2 for j in L["idx"]]); break
        else:
            lines.append({"cy": cy, "idx": [i]})
    lines.sort(key=lambda L: L["cy"])
    words = []  # list of (line_no, [stroke idx])
    for ln, L in enumerate(lines):
        idx = sorted(L["idx"], key=lambda i: boxes[i][0])
        cur, cur_x1 = [], None
        for i in idx:
            x0, _, x1, _ = boxes[i]
            if cur and x0 - cur_x1 > gap_factor*h:
                words.append((ln, cur)); cur = []
            cur.append(i); cur_x1 = max(cur_x1 if cur_x1 is not None else x1, x1)
        if cur: words.append((ln, cur))
    return words, h

def render_word(strokes, idx, h, size=(448, 160), pad=0.35):
    pts = [p for i in idx for p in strokes[i]]
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    x1, y1 = max(p[0] for p in pts), max(p[1] for p in pts)
    w, hh = max(x1-x0, 1), max(y1-y0, 1)
    W, H = size
    inner_w, inner_h = W*(1-2*pad*0.5), H*(1-2*pad)
    s = min(inner_w/w, inner_h/hh)
    ox, oy = (W - w*s)/2 - x0*s, (H - hh*s)/2 - y0*s
    im = Image.new("RGB", size, "white"); d = ImageDraw.Draw(im)
    lw = max(2, int(round(0.08*h*s)))
    for i in idx:
        s_pts = [(p[0]*s+ox, p[1]*s+oy) for p in strokes[i]]
        if len(s_pts) == 1: s_pts = s_pts*2
        d.line(s_pts, fill="black", width=lw, joint="curve")
    return im

# ----------------------------------------------------------------------- app
app = FastAPI()

class Req(BaseModel):
    strokes: list[list[list[float]]]   # [[ [x,y], ... ], ...]
    gap: float = 0.6                    # word gap threshold in units of median stroke height
    topk: int = 5
    tokens: int | None = None

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))

@app.post("/recognize")
def recognize(r: Req):
    strokes = [s for s in r.strokes if len(s) > 0]
    if not strokes:
        return {"words": [], "timing": {}}
    t0 = time.time()
    words, h = segment(strokes, r.gap)
    ims = [render_word(strokes, idx, h) for _, idx in words]
    t1 = time.time()
    E = model.encode([{"image": im} for im in ims], normalize_embeddings=True, batch_size=16,
                     processing_kwargs={"image": {"max_soft_tokens": r.tokens or args.tokens}})
    t2 = time.time()
    S = torch.from_numpy(np.asarray(E, dtype=np.float32)).to(DEVICE) @ Vt.T
    S = S + args.freq_weight * ZIPF   # mild prior toward common words
    top = torch.topk(S, r.topk, dim=1)
    out = []
    for k, (ln, idx) in enumerate(words):
        cands = [{"word": WORDS[j], "score": float(v)} for v, j in zip(top.values[k].tolist(), top.indices[k].tolist())]
        out.append({"line": ln, "strokes": idx, "candidates": cands})
    return {"words": out, "timing": {"segment_ms": round((t1-t0)*1000), "embed_ms": round((t2-t1)*1000),
                                     "total_ms": round((time.time()-t0)*1000)}}

if __name__ == "__main__":
    import uvicorn, socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "?"
    print(f"\n  open http://localhost:{args.port}   (or on an iPad on the same wifi: http://{ip}:{args.port})\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
