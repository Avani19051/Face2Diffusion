# Face2Diffusion

Face2Diffusion is a face-personalization project that generates images of a person in different scenes, outfits, and artistic styles using a reference face image and a text prompt.

It uses a face identity encoder and Stable Diffusion to preserve the person's facial identity while generating new images based on the given description.

## How It Works

1. **Face Detection:** Detects facial landmarks and aligns the input face.
2. **Identity Extraction:** Uses the MSID encoder to extract facial identity features.
3. **Embedding Mapping:** Converts the identity features into embeddings compatible with Stable Diffusion.
4. **Image Generation:** Combines the identity embeddings with a text prompt to generate personalized images.

## Getting Started

The easiest way to run the project is using Google Colab.

1. Open `Face2Diffusion_Demo.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run the notebook cells in sequence.
4. Provide a face image and customize the text prompt.

Example prompt:

```text
f sitting in a cafe, cinematic photography
```

Here, `f` represents the person whose identity should be preserved.

## Run Locally

A CUDA-enabled NVIDIA GPU and the required pretrained model checkpoints are needed.

Install the additional dependencies:

```bash
pip install transformers==4.30.2 face_alignment timm
```

Run the inference script:

```bash
python inference_f2d.py \
  --w_map checkpoints/mapping.pt \
  --w_msid checkpoints/msid.pt \
  -i input/face.jpg \
  -p "f sitting in a cafe" \
  -o output.png \
  -n 8
```

## Project Structure

- `Face2Diffusion_Demo.ipynb` – Google Colab demonstration.
- `inference_f2d.py` – Image generation script.
- `src/` – Face encoder, mapping network, and supporting utilities.
- `input/` – Input face images.
- `checkpoints/` – Pretrained model weights.
- `dockerfiles/` – Docker environment configuration.

## Technologies Used

Python, PyTorch, Stable Diffusion, Hugging Face Diffusers, Transformers, OpenCV.

## Credits

This project is based on the Face2Diffusion research by Kaede Shiohara and Toshihiko Yamasaki.

- [Original Repository](https://github.com/mapooon/Face2Diffusion)
- [Research Paper](https://arxiv.org/abs/2403.05094)
- [Project Page](https://mapooon.github.io/Face2DiffusionPage)

See `LICENSE.md` for licensing and usage terms.
