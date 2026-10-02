# Design Meaning Benchmark

Data and code for **Evaluating Vision–Language Models for Perceived Product Meaning: Category Priors and Viewer Disagreement**, by Taik Su Hong and Jang Woo Kwon.

- Version: 1.0.1
- Full dataset and code release: https://github.com/Logy-CTO/design-meaning-benchmark/releases/tag/v1.0.1
- Code: https://github.com/Logy-CTO/design-meaning-benchmark
- 1,600 generated product images; eight categories, 200 images each.
- 30 participants rated the same balanced 200-image subset, producing 6,000 responses.
- Nine prompted systems provide 14,400 image-level outputs.
- Additional saved experiments include prompt stability, matched image/category controls, adapter target comparisons, category-held-out evaluation and nested configuration selection.

## Download and reproduce

Clone this repository. Download the eight `images_<category>.zip` archives from the [GitHub release](https://github.com/Logy-CTO/design-meaning-benchmark/releases/tag/v1.0.1) and extract all of them into the repository root; each archive contains files under `images/`. Images are attached to the GitHub release. The release also contains `code_and_data_v1.0.1.zip` and `SHA256SUMS.txt`.

```sh
python scripts/download_images.py
python -m venv .venv
# Activate the environment using your operating system's command.
python -m pip install -r requirements.txt
python scripts/reproduce.py
python scripts/reproduce.py --extended
python figures/reproduce_figures.py
```

The default command recalculates agreement, prompted/baseline metrics, distributional scores, human reference baselines and ordinal concordance from recorded responses and predictions. `--extended` also checks prompt controls, image hashes and additional adapter experiments; it requires the image archives. Neither command trains models or sends API requests. Results are written under `reproduced_results/`, with logs and a numeric comparison against the archived main results.

Generation, inference and training entry points are in `code/` and `experiments/*/code/`. These need the original models, compatible hardware and additional dependencies. See `environment/README.md`; weights, API credentials and server caches are not distributed. The saved predictions support CPU reproduction without obtaining model access.

## Layout

| Directory | Contents |
| --- | --- |
| `data/` | Image inventory, pseudonymized human ratings, prompted outputs, adapter probabilities and folds. |
| `analysis/` | Statistical analysis from recorded outputs. |
| `experiments/` | Additional experimental protocols, splits, predictions and analyses. |
| `expected_results/` | Archived numerical reference files for reproduction. |
| `code/` | Image generation, prompted inference, schema and adapter code. |
| `figures/` | Regenerate the ten distribution, category-bias and aesthetic diagnostic plots. |
| `docs/` | Data dictionary, protocol and benchmark interpretation. |

See [the data dictionary](docs/DATA_DICTIONARY.md). The field `design_intent` means a viewer's perceived meaning label, not verified designer intention. The collection and participant panel are fixed; the repository does not establish usefulness in a design workflow or generalization to new populations.

## Rights and citation

Original research code is provided under MIT. Dataset files, prompts and documentation are provided under CC BY-NC 4.0 to the extent the authors hold applicable rights. Generated image outputs are additionally subject to the applicable Black Forest Labs output-use conditions; see `LICENSE_DATA.md`. Dependencies and pretrained models retain their own licenses. No model weights are included.

Use `CITATION.cff` to cite the versioned data/code release. This GitHub release has no DOI; its version URL is the citation link. The manuscript has not been assigned a publication DOI.
