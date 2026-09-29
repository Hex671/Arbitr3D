#!/usr/bin/env python3
"""
消融实验调度器
==============
同时管理 5 组消融实验进程，自动分配 GPU，控制最大并行数防止显存不足。

5 组实验:
  1. no_unified      — 去掉泛化知识库
  2. no_initial_inst — 去掉初级实例知识库
  3. no_updated_inst — 去掉更新实例知识库 (跳过 step 2.5)
  4. no_knowledge    — 去掉所有知识库
  5. full            — 完整知识库 (基线)

用法:
  python scripts/run_ablation.py --category Display --max-parallel 2
  python scripts/run_ablation.py --category Display --max-parallel 3 --gpu-ids 0,1,2,3
  python scripts/run_ablation.py --category Display --include-experiments full,no_unified
  python scripts/run_ablation.py --category Display --exclude-experiments no_knowledge,no_updated_inst
  python scripts/run_ablation.py --category Display --include-experiments full --suffix-template myexp_{category}_{exp_name}
"""

import os
import sys
import copy
import yaml
import time
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================
# 5 组消融实验定义
# ============================================================
ABLATION_EXPERIMENTS = {
    "no_unified": {
        "desc": "去掉泛化知识库",
        "ablation": {
            "use_unified_knowledge": False,
            "use_initial_instance_knowledge": True,
            "use_updated_instance_knowledge": True,
        },
    },
    "no_initial_inst": {
        "desc": "去掉初级实例知识库",
        "ablation": {
            "use_unified_knowledge": True,
            "use_initial_instance_knowledge": False,
            "use_updated_instance_knowledge": True,
        },
    },
    "no_updated_inst": {
        "desc": "去掉更新实例知识库 (跳过 step 2.5)",
        "ablation": {
            "use_unified_knowledge": True,
            "use_initial_instance_knowledge": True,
            "use_updated_instance_knowledge": False,
        },
    },
    "no_knowledge": {
        "desc": "去掉所有知识库",
        "ablation": {
            "use_unified_knowledge": False,
            "use_initial_instance_knowledge": False,
            "use_updated_instance_knowledge": False,
        },
    },
    "full": {
        "desc": "完整知识库 (基线)",
        "ablation": {
            "use_unified_knowledge": True,
            "use_initial_instance_knowledge": True,
            "use_updated_instance_knowledge": True,
        },
    },
}


def get_gpu_free_memory() -> dict:
    """查询每张 GPU 的空闲显存 (GB)，返回 {gpu_id: free_gb}。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        mapping = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            mapping[int(parts[0])] = float(parts[1]) / 1024.0
        return mapping
    except Exception:
        return {}


def pick_best_gpu(allowed_ids: list, min_free_gb: float, busy_gpus: set) -> int:
    """从允许的 GPU 中挑选空闲显存最大且未被占用的卡。返回 -1 表示没有合适的卡。"""
    free_map = get_gpu_free_memory()
    candidates = {
        gid: free_map[gid]
        for gid in allowed_ids
        if gid in free_map and free_map[gid] >= min_free_gb and gid not in busy_gpus
    }
    if not candidates:
        return -1
    return max(candidates, key=lambda g: candidates[g])


def generate_config(base_config: dict, ablation_flags: dict, out_path: str):
    """基于基础配置生成消融实验配置文件。"""
    cfg = copy.deepcopy(base_config)
    cfg["ablation"] = ablation_flags
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


def parse_experiment_names(raw: str) -> list:
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    if raw.lower() == "all":
        return list(ABLATION_EXPERIMENTS.keys())
    return [x.strip() for x in raw.split(",") if x.strip()]


def validate_experiment_names(names: list, arg_name: str):
    valid_names = set(ABLATION_EXPERIMENTS.keys())
    invalid_names = [name for name in names if name not in valid_names]
    if invalid_names:
        raise SystemExit(
            f"[Error] 参数 {arg_name} 中包含未知实验: {invalid_names}。"
            f" 可选实验为: {', '.join(ABLATION_EXPERIMENTS.keys())}"
        )


def build_experiment_suffix(exp_name: str, category: str, suffix_template: str) -> str:
    try:
        suffix = suffix_template.format(exp_name=exp_name, category=category)
    except KeyError as exc:
        raise SystemExit(
            f"[Error] --suffix-template 包含不支持的占位符: {exc}. "
            "仅支持 {exp_name} 和 {category}。"
        )
    if not suffix.strip():
        raise SystemExit("[Error] --suffix-template 生成了空 suffix，请检查参数。")
    return suffix


def main():
    parser = argparse.ArgumentParser(description="消融实验调度器")
    parser.add_argument("--category", type=str, default="Display",
                        help="测试类别 (默认: Display)")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="最大同时运行的实验进程数 (默认: 2)")
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="允许使用的 GPU 编号，逗号分隔 (默认: 所有非 vLLM 占用的卡)")
    parser.add_argument("--min-free-gb", type=float, default=7.0,
                        help="选卡时最低空闲显存要求 (GB, 默认: 7.0)")
    parser.add_argument("--base-config", type=str, default="config/config.yaml",
                        help="基础配置文件路径 (默认: config/config.yaml)")
    parser.add_argument("--experiments", type=str, default="",
                        help="指定要运行的实验 (逗号分隔, 兼容旧用法)")
    parser.add_argument("--include-experiments", type=str, default="",
                        help="显式指定要运行的实验 (逗号分隔，如: full,no_unified；all 表示全部)")
    parser.add_argument("--exclude-experiments", type=str, default="",
                        help="显式指定不运行的实验 (逗号分隔，如: no_knowledge,no_updated_inst)")
    parser.add_argument("--suffix-template", type=str, default="ablation_{exp_name}",
                        help="实验后缀模板，支持 {exp_name} 和 {category}，默认: ablation_{exp_name}")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # 加载基础配置
    with open(args.base_config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # 解析允许的 GPU 列表
    if args.gpu_ids:
        allowed_gpus = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        allowed_gpus = sorted(get_gpu_free_memory().keys())

    if args.experiments and args.include_experiments:
        raise SystemExit("[Error] 请不要同时使用 --experiments 和 --include-experiments。")

    include_raw = args.include_experiments or args.experiments
    include_names = parse_experiment_names(include_raw)
    exclude_names = parse_experiment_names(args.exclude_experiments)

    validate_experiment_names(include_names, "--include-experiments/--experiments")
    validate_experiment_names(exclude_names, "--exclude-experiments")

    if include_names:
        selected_names = [name for name in ABLATION_EXPERIMENTS.keys() if name in include_names]
    else:
        selected_names = list(ABLATION_EXPERIMENTS.keys())

    if exclude_names:
        exclude_set = set(exclude_names)
        selected_names = [name for name in selected_names if name not in exclude_set]

    if not selected_names:
        raise SystemExit("[Error] 过滤后没有可运行的实验，请检查 include/exclude 参数。")

    experiments = {name: ABLATION_EXPERIMENTS[name] for name in selected_names}
    skipped_names = [name for name in ABLATION_EXPERIMENTS.keys() if name not in selected_names]

    # 创建临时配置目录和日志目录
    config_dir = project_root / "config" / "ablation_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir = project_root / "logs" / "ablation"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"  消融实验调度器")
    print(f"  类别: {args.category}")
    print(f"  最大并行数: {args.max_parallel}")
    print(f"  允许 GPU: {allowed_gpus}")
    print(f"  最低显存: {args.min_free_gb:.1f} GB")
    print(f"  实验组数: {len(experiments)}")
    print(f"  将运行: {selected_names}")
    print(f"  后缀模板: {args.suffix_template}")
    if skipped_names:
        print(f"  跳过实验: {skipped_names}")
    print("=" * 70)

    # 生成每组实验的配置文件
    exp_configs = {}
    used_suffixes = set()
    for exp_name, exp_def in experiments.items():
        cfg_path = config_dir / f"config_{exp_name}.yaml"
        exp_suffix = build_experiment_suffix(exp_name, args.category, args.suffix_template)
        if exp_suffix in used_suffixes:
            raise SystemExit(
                f"[Error] suffix '{exp_suffix}' 重复。"
                "多实验运行时请确保 --suffix-template 能生成唯一后缀，通常应包含 {exp_name}。"
            )
        used_suffixes.add(exp_suffix)
        generate_config(base_config, exp_def["ablation"], str(cfg_path))
        exp_configs[exp_name] = {
            "config_path": str(cfg_path),
            "desc": exp_def["desc"],
            "suffix": exp_suffix,
        }
        print(f"  [{exp_name}] {exp_def['desc']} | suffix={exp_suffix}")
        for k, v in exp_def["ablation"].items():
            flag = "ON" if v else "OFF"
            print(f"    {k}: {flag}")

    print()

    # ============================================================
    # 进程调度：控制最大并行数 + GPU 分配
    # ============================================================
    pending = list(exp_configs.keys())
    running = {}  # exp_name -> {"proc": Popen, "gpu": int, "log": str}
    finished = {}  # exp_name -> returncode
    busy_gpus = set()

    def check_running():
        """检查已运行的进程，回收已完成的。"""
        done_keys = []
        for exp_name, info in running.items():
            retcode = info["proc"].poll()
            if retcode is not None:
                done_keys.append(exp_name)
                finished[exp_name] = retcode
                busy_gpus.discard(info["gpu"])
                status = "SUCCESS" if retcode == 0 else f"FAILED (code={retcode})"
                print(f"  [{exp_name}] 实验完成: {status} | 日志: {info['log']}")
        for k in done_keys:
            del running[k]

    print("开始调度实验进程...")
    print("-" * 70)

    while pending or running:
        check_running()

        # 尝试启动新进程（不超过并行上限）
        while pending and len(running) < args.max_parallel:
            exp_name = pending[0]
            info = exp_configs[exp_name]

            # 选 GPU
            gpu_id = pick_best_gpu(allowed_gpus, args.min_free_gb, busy_gpus)
            if gpu_id < 0:
                # 没有空闲 GPU，等下一轮
                break

            pending.pop(0)
            busy_gpus.add(gpu_id)

            log_path = str(log_dir / f"{exp_name}_{args.category}_{timestamp}.log")
            log_file = open(log_path, "w", encoding="utf-8")

            env = os.environ.copy()
            env["CONFIG_PATH"] = info["config_path"]
            env["EXP_SUFFIX"] = info["suffix"]
            env["TARGET_CATEGORY"] = args.category
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["no_proxy"] = "localhost,127.0.0.1"

            proc = subprocess.Popen(
                [sys.executable, "batch_test.py"],
                cwd=str(project_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

            running[exp_name] = {
                "proc": proc,
                "gpu": gpu_id,
                "log": log_path,
                "log_file": log_file,
            }
            print(f"  [{exp_name}] 已启动 | GPU={gpu_id} | PID={proc.pid} | {info['desc']}")

        # 等待一段时间再检查
        if pending or running:
            time.sleep(15)

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 70)
    print("  消融实验全部完成")
    print("=" * 70)

    for exp_name, retcode in finished.items():
        status = "SUCCESS" if retcode == 0 else f"FAILED"
        desc = exp_configs[exp_name]["desc"]
        log_path = str(log_dir / f"{exp_name}_{args.category}_{timestamp}.log")
        print(f"  [{exp_name:20s}] {status:8s} | {desc}")

        # 尝试从日志中提取 mIoU
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            for line in content.splitlines():
                if "总体平均 mIoU" in line:
                    print(f"    >> {line.strip()}")
                    break
        except Exception:
            pass

    # 生成汇总文件
    summary_path = log_dir / f"ablation_summary_{args.category}_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"消融实验汇总 - {args.category}\n")
        f.write(f"时间: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        for exp_name, retcode in finished.items():
            desc = exp_configs[exp_name]["desc"]
            f.write(f"[{exp_name}] {desc}\n")
            f.write(f"  状态: {'SUCCESS' if retcode == 0 else 'FAILED'}\n")
            # 提取 mIoU
            log_path_str = str(log_dir / f"{exp_name}_{args.category}_{timestamp}.log")
            try:
                with open(log_path_str, "r", encoding="utf-8") as lf:
                    for line in lf:
                        if "总体平均 mIoU" in line:
                            f.write(f"  {line.strip()}\n")
                        elif "各部件平均 IoU" in line:
                            f.write(f"  {line.strip()}\n")
            except Exception:
                pass
            f.write("\n")

    print(f"\n  汇总报告: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
