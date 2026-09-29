"""
M0' 零侵入回归测试。

目的：验证新增的 PartVerse RES 流程**完全不影响**现有 PartNetE / PartObjaverse-Tiny
路径。本脚本不需要 GPU、不需要外部数据集、不需要 OpenAI API key 即可运行。

执行的检查项：

    [Test 1] Import smoke test
        所有新增模块（res_pipeline, res_classifier, res_metrics, partverse_loader,
        target_reconstructor, distractor_discoverer, topology_inferrer,
        res_second_round_classifier, res_prompts）都能被成功 import。

    [Test 2] 现有公开 API 兼容性
        - pipeline.Arbitr3DPipeline 仍然存在且仍提供 run() 方法
        - module_2d_fm.mllm_classifier.MLLMClassifier 仍然提供
          predict_soft_probabilities_all_views / _render_som_image / _encode_image_to_base64
        - module_mllm.second_round_classifier.SecondRoundMLLMClassifier 仍存在
        - data_prep.dataloader.PointCloudDataset 仍存在

    [Test 3] config.yaml 保持向后兼容
        - data / model / render / mllm / algorithm / superpoint / ablation 段落都仍然存在
        - res 段落新增（不会被任何现有路径读取）

    [Test 4] 没有现有文件被新模块的 import 路径污染
        现有 pipeline.py / test.py / batch_test.py / mllm_classifier.py 的源码中
        不应出现 'from res_pipeline' / 'from module_2d_fm.res_classifier' 等。
        （RES 模块只能被 RES 自己 import，反向 import 严禁。）

    [Test 5] 数据结构幂等性
        - res_metrics.compute_query_iou 在简单合成数据上数值正确
        - res_metrics.aggregate_results 输出聚合字段非空

运行：
    python tests/regression_zero_intrusion.py
"""

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

# 把项目根加入 sys.path
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PROJ_DIR))


def _print_section(title: str):
    bar = "=" * 70
    print("\n" + bar)
    print(f"  {title}")
    print(bar)


def _ok(msg: str):
    print(f"  [OK]   {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


# =============================================================================
# Test 1: Import smoke test
# =============================================================================
def test_imports():
    _print_section("Test 1: 新增模块 Import 检查")
    new_modules = [
        "config.res_prompts",
        "data_prep.partverse_loader",
        "module_mllm.k_u_target_reconstructor",
        "module_mllm.k_u_distractor_discoverer",
        "module_mllm.topology_inferrer",
        "module_mllm.res_second_round_classifier",
        "module_2d_fm.res_classifier",
        "evaluation.res_metrics",
        "res_pipeline",
    ]
    failures = []
    for mod_name in new_modules:
        try:
            importlib.import_module(mod_name)
            _ok(f"import {mod_name}")
        except Exception as e:
            _fail(f"import {mod_name} -> {type(e).__name__}: {e}")
            failures.append(mod_name)
            traceback.print_exc()
    return failures


# =============================================================================
# Test 2: 现有公开 API 兼容性
# =============================================================================
def test_existing_api_intact():
    _print_section("Test 2: 现有公开 API 兼容性")
    failures = []

    checks = [
        ("pipeline", "Arbitr3DPipeline", ["run"]),
        ("module_2d_fm.mllm_classifier", "MLLMClassifier",
         ["predict_soft_probabilities_all_views",
          "_render_som_image", "_encode_image_to_base64", "_get_annotation_font"]),
        ("module_mllm.second_round_classifier", "SecondRoundMLLMClassifier",
         ["predict_suspect_masks"]),
        ("data_prep.dataloader", "PointCloudDataset",
         ["load_point_cloud", "normalize_pc", "extract_color"]),
        ("module_3d_lift.soft_projector", "SoftProjector",
         ["project_with_depth_check"]),
        ("module_3d_lift.probability_fusion", "ProbabilityFusion",
         ["fuse_probabilities"]),
        ("module_3d_lift.topology_builder", "TopologyGraphBuilder",
         ["validate_masks_topology", "to_prompt_text"]),
    ]
    for mod_name, cls_name, methods in checks:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            missing = [m for m in methods if not hasattr(cls, m)]
            if missing:
                _fail(f"{mod_name}.{cls_name} 缺少方法: {missing}")
                failures.append((mod_name, cls_name, missing))
            else:
                _ok(f"{mod_name}.{cls_name} OK ({len(methods)} 方法存在)")
        except Exception as e:
            _fail(f"{mod_name}.{cls_name} 检查失败: {e}")
            failures.append((mod_name, cls_name, str(e)))
    return failures


# =============================================================================
# Test 3: config.yaml 向后兼容
# =============================================================================
def test_config_yaml_compat():
    _print_section("Test 3: config.yaml 向后兼容")
    failures = []
    try:
        import yaml
        cfg_path = _PROJ_DIR / "config" / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 现有段落必须仍存在
        required_sections = ["data", "model", "render", "mllm", "algorithm", "superpoint"]
        for sec in required_sections:
            if sec not in cfg:
                _fail(f"config 缺少现有段落 '{sec}'")
                failures.append(sec)
            else:
                _ok(f"config 段落 '{sec}' 仍存在")

        # 新增段落必须存在
        if "res" not in cfg:
            _fail("config 缺少新增段落 'res'")
            failures.append("res-missing")
        else:
            _ok("config 段落 'res' 已添加")
            res_required_keys = [
                "partverse_data_root", "text_captions_json",
                "glbs_subdir", "anno_infos_subdir", "num_points",
                "use_target_K_u_reconstruction", "use_distractor_discovery",
                "use_online_topology", "enable_second_round",
                "caption_types", "cache_dir", "results_dir",
            ]
            for key in res_required_keys:
                if key not in cfg["res"]:
                    _fail(f"res 段落缺少 '{key}'")
                    failures.append(f"res.{key}")

    except Exception as e:
        _fail(f"读取/解析 config.yaml 失败: {e}")
        failures.append("config-load")
        traceback.print_exc()
    return failures


# =============================================================================
# Test 4: 现有文件未被 RES 模块污染
# =============================================================================
def test_no_reverse_import():
    _print_section("Test 4: 现有文件未被 RES 模块反向 import")
    failures = []
    forbidden_imports = [
        "from res_pipeline",
        "import res_pipeline",
        "from data_prep.partverse_loader",
        "from module_2d_fm.res_classifier",
        "from module_mllm.k_u_target_reconstructor",
        "from module_mllm.k_u_distractor_discoverer",
        "from module_mllm.topology_inferrer",
        "from module_mllm.res_second_round_classifier",
        "from evaluation.res_metrics",
        "from config.res_prompts",
    ]
    # 现有路径上不应该出现这些 import
    legacy_files = [
        "pipeline.py",
        "test.py",
        "batch_test.py",
        "test_objaverse.py",
        "test_mllm.py",
        "module_2d_fm/mllm_classifier.py",
        "module_2d_fm/sam_auto_segmenter.py",
        "module_mllm/second_round_classifier.py",
        "module_mllm/knowledge_manager.py",
        "module_mllm/pose_checker.py",
        "module_3d_lift/topology_builder.py",
        "module_3d_lift/probability_fusion.py",
        "module_3d_lift/soft_projector.py",
        "data_prep/dataloader.py",
    ]
    for rel_path in legacy_files:
        full = _PROJ_DIR / rel_path
        if not full.exists():
            print(f"  [SKIP] {rel_path}（文件不存在）")
            continue
        try:
            with open(full, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as e:
            _fail(f"无法读取 {rel_path}: {e}")
            failures.append(rel_path)
            continue
        bad = [imp for imp in forbidden_imports if imp in src]
        if bad:
            _fail(f"{rel_path} 反向 import RES 模块: {bad}")
            failures.append((rel_path, bad))
        else:
            _ok(f"{rel_path} 无反向 import")
    return failures


# =============================================================================
# Test 5: res_metrics 数值正确性
# =============================================================================
def test_res_metrics_numerics():
    _print_section("Test 5: res_metrics 数值正确性（合成数据）")
    failures = []
    try:
        import numpy as np
        from evaluation.res_metrics import (
            RESPerQueryResult,
            aggregate_results,
            compute_query_iou,
            format_aggregate_summary,
            to_serializable,
        )

        # 5a. 单 query IoU
        pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)  # |pred|=3
        gt   = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)  # |gt|=3
        # inter={0,1}=2, union={0,1,2,4}=4, iou=0.5
        inter, union, pc, gc = compute_query_iou(pred, gt)
        if (inter, union, pc, gc) == (2, 4, 3, 3):
            _ok(f"compute_query_iou 正确: inter={inter}, union={union} -> iou=0.5")
        else:
            _fail(f"compute_query_iou 期望 (2,4,3,3) 实际 ({inter},{union},{pc},{gc})")
            failures.append("compute_query_iou")

        # 5b. 聚合
        per_query = [
            RESPerQueryResult(
                uid="u1", part_id="0", caption_type="brief",
                intersection=2, union=4, pred_count=3, gt_count=3,
            ),
            RESPerQueryResult(
                uid="u1", part_id="0", caption_type="detailed",
                intersection=4, union=5, pred_count=4, gt_count=5,
            ),
            RESPerQueryResult(
                uid="u2", part_id="1", caption_type="brief",
                intersection=0, union=10, pred_count=3, gt_count=7,
            ),
        ]
        agg = aggregate_results(per_query)
        if agg.num_queries != 3 or agg.valid_queries != 3:
            _fail(f"aggregate_results 数量错误: {agg.num_queries}/{agg.valid_queries}")
            failures.append("aggregate_count")
        # cumulative IoU = (2+4+0) / (4+5+10) = 6/19 ≈ 0.3158
        ciou_expect = 6 / 19
        if abs(agg.cumulative_iou - ciou_expect) < 1e-6:
            _ok(f"cumulative_iou 正确: {agg.cumulative_iou:.4f}")
        else:
            _fail(f"cumulative_iou 期望 {ciou_expect:.4f} 实际 {agg.cumulative_iou:.4f}")
            failures.append("cumulative_iou")
        # aggregated IoU = mean(0.5, 0.8, 0.0) = 0.4333
        aiou_expect = (0.5 + 0.8 + 0.0) / 3
        if abs(agg.aggregated_iou - aiou_expect) < 1e-6:
            _ok(f"aggregated_iou 正确: {agg.aggregated_iou:.4f}")
        else:
            _fail(f"aggregated_iou 期望 {aiou_expect:.4f} 实际 {agg.aggregated_iou:.4f}")
            failures.append("aggregated_iou")
        # Pr@0.5 = 2/3
        if abs(agg.precision_at_05 - 2/3) < 1e-6:
            _ok(f"precision_at_05 正确: {agg.precision_at_05:.4f}")
        else:
            _fail(f"precision_at_05 期望 0.6667 实际 {agg.precision_at_05:.4f}")
            failures.append("precision_at_05")

        # 5c. caption type 分组
        if "brief" in agg.by_caption_type and "detailed" in agg.by_caption_type:
            _ok("by_caption_type 包含 brief / detailed")
        else:
            _fail(f"by_caption_type 异常: {list(agg.by_caption_type.keys())}")
            failures.append("by_caption_type")

        # 5d. 序列化 + 摘要打印不报错
        ser = to_serializable(agg)
        json.dumps(ser)
        summary = format_aggregate_summary(agg)
        assert "RES Evaluation Summary" in summary
        _ok("to_serializable / format_aggregate_summary 正常")

    except Exception as e:
        _fail(f"res_metrics 数值测试异常: {e}")
        failures.append("metrics-exception")
        traceback.print_exc()
    return failures


# =============================================================================
# Test 6: PartVerseLoader 接口存在性 + Hash 稳定性
# =============================================================================
def test_loader_interface():
    _print_section("Test 6: PartVerseLoader 接口存在 + Query Hash 稳定")
    failures = []
    try:
        from data_prep.partverse_loader import (
            PartVerseLoader,
            PartVerseInstance,
            PartVerseQuery,
        )
        # PartVerseQuery hash 稳定
        q1 = PartVerseQuery(
            uid="abc", part_id="0", caption_type="brief",
            brief="b", detailed="d",
        )
        q2 = PartVerseQuery(
            uid="abc", part_id="0", caption_type="brief",
            brief="completely different brief",
            detailed="completely different detailed",
        )
        # uid + part_id + caption_type 决定 hash，改 brief 不影响
        if q1.query_hash == q2.query_hash:
            _ok(f"PartVerseQuery.query_hash 稳定: {q1.query_hash}")
        else:
            _fail(f"query_hash 应该稳定但变化了: {q1.query_hash} vs {q2.query_hash}")
            failures.append("query_hash")

        q3 = PartVerseQuery(
            uid="abc", part_id="1", caption_type="brief",
            brief="b", detailed="d",
        )
        if q1.query_hash != q3.query_hash:
            _ok(f"不同 part_id 的 hash 不同: {q1.query_hash} vs {q3.query_hash}")
        else:
            _fail("不同 part_id 的 hash 不应相同")
            failures.append("query_hash_collision")

        # text_for_mllm 行为
        b, d = q1.text_for_mllm()
        if b == "b" and d == "":
            _ok("brief caption_type 只返回 brief 文本")
        else:
            _fail(f"text_for_mllm(brief) 异常: {(b, d)}")
            failures.append("text_for_mllm")

    except Exception as e:
        _fail(f"loader 接口测试异常: {e}")
        failures.append("loader-exception")
        traceback.print_exc()
    return failures


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n" + "#" * 70)
    print("#  PartVerse RES — Zero-Intrusion Regression Test (M0')")
    print("#" * 70)
    print(f"#  Project: {_PROJ_DIR}")

    all_failures = []
    all_failures.extend([("Test1", f) for f in test_imports()])
    all_failures.extend([("Test2", f) for f in test_existing_api_intact()])
    all_failures.extend([("Test3", f) for f in test_config_yaml_compat()])
    all_failures.extend([("Test4", f) for f in test_no_reverse_import()])
    all_failures.extend([("Test5", f) for f in test_res_metrics_numerics()])
    all_failures.extend([("Test6", f) for f in test_loader_interface()])

    print("\n" + "#" * 70)
    if not all_failures:
        print("#  ALL TESTS PASSED  零侵入约束已满足")
        print("#" * 70)
        return 0
    else:
        print(f"#  {len(all_failures)} FAILURE(S) DETECTED")
        for tag, f in all_failures:
            print(f"#    [{tag}] {f}")
        print("#" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
