# Handwriting → text with EmbeddingGemma 2 (zero-shot)

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) puts images and text in one
embedding space. So an image of a handwritten word should land next to the embedding of that word
written as text. This repo reads handwriting with nothing more than that: embed the image, embed
a vocabulary of English words, pick the closest by cosine similarity. No training.

Inspired by [Embedding Gemma Transcribe](https://eg2.haku.sh/), which does the same for speech
(with a small trained reader on top of the audio encoder).

## Quick start: one PNG → one word

Open [`read_word.ipynb`](read_word.ipynb) and run all cells.

```python
read_word("galaxy.png")
# [('galaxy', 0.768), ('handwritten', 0.72), ('galaxies', 0.72), ...]
```

The image should contain a single word, dark ink on a light background.

## Live drawing app

```bash
python server.py
```

Then open http://localhost:8765 and write with a mouse, trackpad, or (best) an iPad + Pencil on the
same wifi, using the address the server prints. Strokes are split into words by pen gaps, each word is
recognised separately, and you can click a word to pick an alternative.

Options: `--vocab 50000` (bigger vocabulary), `--tokens 140` (more image detail, slower).

## Setup

```bash
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), an NVIDIA GPU (CUDA), or CPU (slower). The model (~1.5 GB) downloads
from Hugging Face on first use. Embedding the 30k-word vocabulary takes ~30 s on an M-series Mac.

## How well does it work?

On words rendered in handwriting-style fonts, against the 30k most common English words:

| image tokens | top-1 | top-5 | time per word (M5 Pro) |
|---|---|---|---|
| 70  | 90% | 100% | ~70 ms |
| 280 | 90% | 98%  | ~420 ms |

Most misses are near-neighbours like *rigorously → rigorous*. Not yet benchmarked on real handwriting.
Words outside the vocabulary (names, numbers, symbols) can't be read.
