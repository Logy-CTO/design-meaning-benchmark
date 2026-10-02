# Data dictionary

The manifest has 1,600 images in eight categories. The survey manifest identifies the same 200 images rated by all 30 participants.

| File | Unit and interpretation |
| --- | --- |
| `master_image_manifest.csv` | One generated image; category, sample index, filename, generation settings and prompt. |
| `image_sha256.csv` | SHA-256 digest for every image file. |
| `survey_manifest_9b_200.csv` | The 200-image human evaluation subset. |
| `human_responses_30x200.csv` | 6,000 ratings, one row per participant and image. R01–R30 are release-specific pseudonyms. |
| `human_image_summary.csv` | Per-image label counts, five-class response shares and confidence summary. |
| `vlm_predictions_full1600.csv` | 14,400 parsed outputs: nine prompted systems × 1,600 images. |
| `adapter_oof_predictions_200.csv` | Five-fold adapter outputs for the 200-image evaluation set. |
| `aesexpert_scores_1600.csv` | Recorded aesthetic evaluation scores. |
| `adapter_folds/` | Train, validation and test records for the preserved five-fold experiment. |
| `experiments/` | Additional prompt controls, objective comparisons and category-held-out splits and outputs. |

## Human response fields

`design_intent` is the historical field name for a participant-selected perceived meaning label; it does not measure a designer's intended meaning. Its values are functionality, aesthetics, usability, symbolism and unclear.

`recognized_as_product` records product recognition. `image_quality`, `target_category_recognizability`, `confidence`, and `form_function_consistency` use the questionnaire's 1–5 ratings. `function_aesthetic_score` is an ordinal orientation item anchored at 1 (function-oriented) and 5 (aesthetic or symbolic-oriented).

`rater` preserves within-participant grouping. `image_id`, `category`, `sample_index`, `image_file` and `generation_model` connect responses to the image inventory. Contact details, original respondent identifiers, administrative survey fields are not included.

## Interpretation and evaluation scope

The human panel and images are fixed. Image bootstraps describe uncertainty within the eight observed categories conditional on the same panel. Category-held-out analyses remain internal to this collection. The original five-fold adapter was selected using evaluation results and is exploratory; the nested category-held-out experiment separates inner selection from outer evaluation. Human label distributions are retained rather than treated as independent ground truth.

The archived `expected_results/human30_human_agreement_summary.csv` is retained only as the historical Fleiss kappa check. The current nominal alpha is recomputed by `agreement_bootstrap.py`; use `human_agreement_bootstrap.csv` for current agreement values.

The archived `expected_results/human30_human_agreement_summary.csv` is retained only as the historical Fleiss kappa check. The current nominal alpha is recomputed by `agreement_bootstrap.py`; use `human_agreement_bootstrap.csv` for current agreement values.
