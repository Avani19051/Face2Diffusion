# Face2Diffusion — running it on Colab

Give it one photo of a face and a sentence, and it draws that person doing whatever the sentence says — "eating bread in front of the Eiffel Tower", "as an astronaut", "in a Van Gogh painting". No fine-tuning per person, no waiting: one forward pass and you get eight images.

This repo is my working copy of **Face2Diffusion** (CVPR 2024) by Kaede Shiohara and Toshihiko Yamasaki. The method and code are theirs — I set it up and ran it end to end on a free Google Colab T4 GPU, and wrote up how it works and how to run it below.

> Original repo: [mapooon/Face2Diffusion](https://github.com/mapooon/Face2Diffusion) · Paper: [arXiv 2403.05094](https://arxiv.org/abs/2403.05094) · Project page: [Face2DiffusionPage](https://mapooon.github.io/Face2DiffusionPage)

![Overview](fig/teaser.png)

---

## How it works (in plain words)

Most "put my face in the picture" methods (like DreamBooth or Textual Inversion) retrain a model on several photos of you. That takes minutes to hours per person. Face2Diffusion skips that.

1. **Find and align the face.** A landmark detector locates the eyes, nose and mouth, and the face is cropped and straightened to 112×112.
2. **Turn the face into an identity vector.** A face encoder (the "MSID" encoder) reads the crop and produces features that capture *who* the person is.
3. **Turn that vector into a word.** A small mapping network converts the identity features into a fake word embedding — a token Stable Diffusion treats like any other word in the prompt.
4. **Generate.** That token is dropped into the prompt wherever you write `f l`, and Stable Diffusion v1.4 generates the images.

The clever part is in step 2–3: the authors train the encoder so it ignores things like pose, expression and background. That's why the prompt can still change the scene, clothes and style without the face drifting into someone else — the main problem earlier fast methods had.

## Run it yourself (easiest: Colab)

Open `Face2Diffusion_Demo.ipynb` in Google Colab and pick **Runtime → Change runtime type → T4 GPU**. Then run the cells top to bottom. The notebook:

1. clones the code and downloads the two pretrained checkpoints (mapping network + MSID encoder),
2. installs `diffusers`, `transformers==4.30.2`, `face_alignment` and `timm`,
3. loads Stable Diffusion v1.4 in half precision so it fits on a T4,
4. encodes the face in `input/0.jpg` and generates 8 images for the prompt.

To try your own photo or idea, change these lines in the notebook:

```python
input_prompt = 'f l eating bread in front of the Eiffel Tower'  # "f l" = the person
img_path = 'input/0.jpg'   # any clear, front-facing face photo
n_samples = 8              # how many images
guidance_scale = 10.0      # higher = follows the prompt more strictly
```

**Things I ran into**
- You must write `f l` in the prompt — that's the placeholder for the person. Leave it out and you just get a random face.
- The first run spends a few minutes downloading Stable Diffusion; later runs are fast.
- `transformers` must stay at `4.30.2`; newer versions break the custom text-encoder hook.
- Side-on or partly covered faces give weaker likeness. A clear, front-facing photo works best.

## Run it locally (needs an NVIDIA GPU)

```bash
docker build -t face2diffusion dockerfiles/
docker run -it --gpus all --shm-size 512G -v /path/to/this/repo:/workspace face2diffusion bash
bash install.sh
```

Download the [mapping network](https://drive.google.com/file/d/1Lf_mwMgme_HVYJCkViGr4TfGOfKw9PhE/view?usp=sharing) and [MSID encoder](https://drive.google.com/file/d/1DjUf-ib612SDDt86TRlsDkHqXTeYx030/view?usp=sharing) checkpoints into `checkpoints/`, then:

```bash
python3 inference_f2d.py \
  --w_map checkpoints/mapping.pt \
  --w_msid checkpoints/msid.pt \
  -i input/0.jpg \
  -p 'f l eating bread in front of the Eiffel Tower' \
  -o output.png \
  -n 8
```

## What's in here

| Path | What it is |
|---|---|
| `Face2Diffusion_Demo.ipynb` | Colab notebook, re-run by me on a T4 GPU |
| `inference_f2d.py` | Command-line script for generating images |
| `src/msid.py` | The identity (MSID) face encoder |
| `src/modules.py`, `src/mod.py` | Mapping network and text-encoder changes |
| `src/utils.py` | Face alignment and cropping helpers |
| `input/` | Sample face images |
| `dockerfiles/`, `install.sh` | Environment setup |

## Credits and license

All of the method and code is by **Kaede Shiohara and Toshihiko Yamasaki** (The University of Tokyo). The original repository is [mapooon/Face2Diffusion](https://github.com/mapooon/Face2Diffusion); it in turn borrows from [InsightFace](https://github.com/deepinsight/insightface), [ICT](https://github.com/LightDXY/ICT_DeepFake) and [Pix2Word](https://github.com/google-research/composed_image_retrieval).

**Changes in this copy:** this README was rewritten, and the demo notebook was re-run on Colab (T4). The model code is unchanged.

Licensed under [CC BY-NC-SA 4.0](LICENSE.md) — free to use and adapt for non-commercial purposes, as long as you credit the authors and share changes under the same license.

If you use this work, please cite the paper:

```bibtex
@inproceedings{shiohara2024face2diffusion,
  title={Face2Diffusion for Fast and Editable Face Personalization},
  author={Shiohara, Kaede and Yamasaki, Toshihiko},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
