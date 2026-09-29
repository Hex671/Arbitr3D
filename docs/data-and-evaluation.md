# Data, knowledge, and evaluation

Datasets and annotations must be acquired separately from their providers. Dataset licenses are independent of the repository's MIT license. Run commands from the repository root, and use a local config copy for machine-specific paths.

## PartNetE

Obtain the PartNet-Ensembled data following the upstream [PartSTAD data preparation instructions](https://github.com/KAIST-Visual-AI-Group/PartSTAD#data-preparation). Arbitr3D expects point clouds and semantic labels laid out as:

```text
datasets/PartNetE/
  test/
    Chair/179/
      pc.ply
      label.npy
  few_shot/
    Chair/<instance_id>/pc.ply
```

`label.npy` is a NumPy dictionary containing `semantic_seg`, as consumed by `evaluation/metrics.py`. The class-name mapping is in `PartNetE_meta.json`. Set `data.base_path` to `test/` and `data.knowledge_base_path` to the separate knowledge-generation split. Ground truth is needed for evaluation, not `infer.py`.

```bash
export CONFIG_PATH=config/config.local.yaml
python scripts/generate_unified_knowledge.py --category Chair
TARGET_CATEGORY=Chair EXP_SUFFIX=release python batch_test.py
```

Knowledge generation renders up to the first eight sorted instances of a category from `data.knowledge_base_path`; it makes MLLM calls and writes category files to `config/knowledge/`. `KNOWLEDGE_DATA_ROOT` overrides this data directory. Keep knowledge-generation and evaluation splits separate when defining an experiment.

The main pipeline looks for `<Category>_unified_knowledge.json`, `<Category>_reasonable_adjacencies.json`, and `<Category>_spatial_height_order.json`. Chair examples and one Dispenser knowledge file are included; these are not a complete benchmark knowledge collection. Missing files can disable parts of the available prior information or invoke existing fallback behavior, so generate and record the files for all evaluated categories.

`scripts/run_full_pipeline.py --category Chair` combines knowledge generation and batch evaluation. Use `--only-knowledge` or `--skip-knowledge` when appropriate. Batch outputs and logs are written beneath `visual_res/` and `logs/`.

## PartObjaverse-Tiny

Download the data and metadata from the [official dataset](https://huggingface.co/datasets/yhyang-myron/PartObjaverse-Tiny). See the [upstream format description](https://github.com/Pointcept/SAMPart3D/blob/main/PartObjaverse-Tiny/PartObjaverse-Tiny.md).

```text
datasets/PartObjaverse-Tiny/PartObjaverse-Tiny/
  PartObjaverse-Tiny_mesh/<uid>.glb
  PartObjaverse-Tiny_semantic_gt/<uid>.npy
  PartObjaverse-Tiny_semantic.json
```

Set `partobjaverse_tiny.base_path` to this directory. `meta_json` can be an absolute path, a path relative to the repository root, or a filename under the dataset root. The original semantic metadata is intentionally obtained with the dataset rather than redistributed here.

```bash
python scripts/generate_objaverse_knowledge.py --category Animals
python test_objaverse.py --config "$CONFIG_PATH" --category Animals --list
python batch_test_objaverse.py --config "$CONFIG_PATH" \
  --categories Animals --max-instances 1 --exp-suffix smoke
```

The number of sampled mesh points is controlled by `partobjaverse_tiny.num_points` or `--num-points`. Set the value deliberately when comparing results. PartObjaverse outputs use the existing `visual_res/` and `logs/` paths.

## PartVerse RES

Obtain [PartVerse](https://huggingface.co/datasets/dscdyc/partverse/tree/main) and point `res.partverse_data_root` at:

```text
datasets/PartVerse/
  normalized_glbs/<uid>.glb
  anno_infos/<uid>/<uid>_face2label.json
  text_captions.json
```

This path uses captions to reconstruct target knowledge, discover distractor parts, and perform topology-based review. It is implemented in `res_pipeline.py` and does not invoke the PartNetE batch script.

```bash
python scripts/sample_partverse_subset.py \
  --text-captions datasets/PartVerse/text_captions.json \
  --per-bucket 2 --out scripts/partverse_subset.json
python eval_partverse_res.py --config "$CONFIG_PATH" \
  --uids-file scripts/partverse_subset.json --limit 1 --max-queries 2
```

Use `--caption-types brief`, `detailed`, or `brief,detailed` to select the query type. Results go to `res.results_dir` (`results_res/` by default); cache files use `res.cache_dir`. Metric summaries include per-query and aggregated IoU and precision at IoU 0.5.

## Ablations and visualization

```bash
python scripts/run_ablation.py --help
python scripts/run_ablation_one_instance.py --help
python scripts/render_partnete_segment.py --help
python scripts/render_mesh_mitsuba.py --help
python test_mask_backproj.py --help
```

The ablation switches in `config/config.yaml` control unified knowledge, initial instance knowledge, updated instance knowledge, and the height prior. Record the config, generated knowledge, model identifier, sample count, random seeds where exposed, and whether Cut-Pursuit loaded. These scripts retain the research implementation's behavior; this release does not claim a new benchmark evaluation.
