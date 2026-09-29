"""
Arbitr3D 全流程一键脚本
======================
整合知识库构建 (generate_unified_knowledge.py) 和批量测试 (batch_test.py) 两步流程。

用法:
    python scripts/run_full_pipeline.py --category Mouse
    python scripts/run_full_pipeline.py --category Chair --skip-knowledge
    python scripts/run_full_pipeline.py --category Table --force-knowledge
    python scripts/run_full_pipeline.py --category Keyboard --only-knowledge
"""

import os
import sys
import argparse
import subprocess
import datetime

# 路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "config", "knowledge")
PYTHON_EXE = sys.executable


# ============================================================
# GPU 自动选择
# ============================================================
def _get_physical_gpu_map() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        mapping = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx = int(parts[0])
            free_gb = float(parts[1]) / 1024.0
            total_gb = float(parts[2]) / 1024.0
            mapping[idx] = (free_gb, total_gb)
        return mapping
    except Exception:
        return {}


def _auto_best_physical(physical_map: dict, threshold: float = 4.0) -> int:
    if not physical_map:
        raise RuntimeError("No GPU available.")
    candidates = {idx: free for idx, (free, _) in physical_map.items() if free >= threshold}
    if not candidates:
        raise RuntimeError(f"No GPU with >= {threshold:.1f} GB free memory.")
    return max(candidates, key=lambda i: candidates[i])


def print_gpu_status(physical_map: dict, selected: int):
    print(f"\n    物理 GPU 状态（nvidia-smi 编号 -> 显存剩余 / 总显存）：")
    for idx in sorted(physical_map.keys()):
        free_gb, total_gb = physical_map[idx]
        bar_len = int((total_gb - free_gb) / total_gb * 20)
        bar = "=" * bar_len + "." * (20 - bar_len)
        used_pct = (total_gb - free_gb) / total_gb * 100
        marker = " <-- selected" if idx == selected else ""
        print(f"    GPU {idx}: [{bar}] {used_pct:.0f}% used, free {free_gb:.1f} / {total_gb:.1f} GB{marker}")
    print()


# ============================================================
# 子进程运行器
# ============================================================
def run_step(cmd_list, step_name, env_vars=None, cwd=None):
    """运行子进程，实时打印输出"""
    print(f"\n{'='*60}")
    print(f"[STEP] {step_name}")
    print(f"CMD:   {' '.join(cmd_list)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd_list, check=True, text=True, env=env_vars, cwd=cwd)
        print(f"\n[OK] {step_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {step_name} (exit code {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {step_name}\n")
        return False


# ============================================================
# 知识库状态检查
# ============================================================
def check_knowledge_exists(category: str) -> dict:
    """检查该类别的知识库文件是否已存在"""
    files = {
        "unified_knowledge": os.path.join(KNOWLEDGE_DIR, f"{category}_unified_knowledge.json"),
        "reasonable_adjacencies": os.path.join(KNOWLEDGE_DIR, f"{category}_reasonable_adjacencies.json"),
        "spatial_height_order": os.path.join(KNOWLEDGE_DIR, f"{category}_spatial_height_order.json"),
    }
    status = {}
    for name, path in files.items():
        status[name] = os.path.exists(path)
    return status


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Arbitr3D Full Pipeline: Knowledge Generation + Batch Testing"
    )
    parser.add_argument(
        "--category", "-c", type=str, required=True,
        help="Object category to process (e.g., Mouse, Chair, Table, Keyboard)"
    )
    parser.add_argument(
        "--skip-knowledge", action="store_true",
        help="Skip knowledge generation even if knowledge files are missing"
    )
    parser.add_argument(
        "--force-knowledge", action="store_true",
        help="Force regenerate knowledge even if files already exist"
    )
    parser.add_argument(
        "--only-knowledge", action="store_true",
        help="Only run knowledge generation, skip batch testing"
    )
    parser.add_argument(
        "--only-test", action="store_true",
        help="Only run batch testing, skip knowledge generation (alias for --skip-knowledge)"
    )
    parser.add_argument(
        "--gpu", type=int, default=-1,
        help="Physical GPU index to use for batch_test (default: auto-select). "
             "Knowledge generation uses MLLM API, not local GPU compute."
    )
    args = parser.parse_args()

    if args.only_test:
        args.skip_knowledge = True

    CATEGORY = args.category
    start_time = datetime.datetime.now()

    print(f"\n{'#'*60}")
    print(f"  Arbitr3D Full Pipeline")
    print(f"  Category: {CATEGORY}")
    print(f"  Time:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    # ---- GPU 信息展示 ----
    physical_map = _get_physical_gpu_map()
    if physical_map:
        # 仅展示 GPU 状态，不主动选择——由 batch_test.py 内部自行选卡
        selected_display = args.gpu if args.gpu >= 0 else -1
        if selected_display < 0:
            try:
                selected_display = _auto_best_physical(physical_map)
            except RuntimeError:
                selected_display = -1
        print_gpu_status(physical_map, selected_display)

    # 构建子进程环境变量
    # 不主动设置 CUDA_VISIBLE_DEVICES，让 batch_test.py 自己的 GPU 自动选择逻辑生效
    # 仅当用户显式指定 --gpu 时才注入
    env = os.environ.copy()
    if args.gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[INFO] --gpu {args.gpu} specified, setting CUDA_VISIBLE_DEVICES={args.gpu} for subprocesses.")
    env["TARGET_CATEGORY"] = CATEGORY

    # ================================================================
    # Step 1: 知识库构建 (generate_unified_knowledge.py)
    # ================================================================
    if not args.skip_knowledge:
        knowledge_status = check_knowledge_exists(CATEGORY)
        all_exist = all(knowledge_status.values())

        if all_exist and not args.force_knowledge:
            print(f"\n[INFO] {CATEGORY} knowledge files already exist:")
            for name, exists in knowledge_status.items():
                print(f"    {'[OK]' if exists else '[MISSING]'} {name}")
            print(f"[INFO] Skipping knowledge generation. Use --force-knowledge to regenerate.\n")
        else:
            if not all_exist:
                print(f"\n[INFO] Missing knowledge files for {CATEGORY}:")
                for name, exists in knowledge_status.items():
                    if not exists:
                        print(f"    [MISSING] {name}")

            script_knowledge = os.path.join(SCRIPTS_DIR, "generate_unified_knowledge.py")
            if not os.path.exists(script_knowledge):
                print(f"[ERROR] Knowledge generation script not found: {script_knowledge}")
                sys.exit(1)

            success = run_step(
                [PYTHON_EXE, script_knowledge, "--category", CATEGORY],
                f"Step 1: Generate Unified Knowledge ({CATEGORY})",
                env_vars=env,
                cwd=BASE_DIR,
            )
            if not success:
                print("[ERROR] Knowledge generation failed. Aborting pipeline.")
                sys.exit(1)

            # 验证生成结果
            knowledge_status = check_knowledge_exists(CATEGORY)
            missing = [k for k, v in knowledge_status.items() if not v]
            if missing:
                print(f"[WARNING] After generation, still missing: {missing}")
    else:
        print(f"\n[INFO] Skipping knowledge generation (--skip-knowledge / --only-test).")

    # ================================================================
    # Step 2: 批量测试 (batch_test.py)
    # ================================================================
    if not args.only_knowledge:
        script_batch_test = os.path.join(BASE_DIR, "batch_test.py")
        if not os.path.exists(script_batch_test):
            print(f"[ERROR] Batch test script not found: {script_batch_test}")
            sys.exit(1)

        success = run_step(
            [PYTHON_EXE, script_batch_test],
            f"Step 2: Batch Test ({CATEGORY})",
            env_vars=env,
            cwd=BASE_DIR,
        )
        if not success:
            print("[ERROR] Batch test failed.")
            sys.exit(1)
    else:
        print(f"\n[INFO] Skipping batch test (--only-knowledge).")

    # ================================================================
    # 完成
    # ================================================================
    elapsed = datetime.datetime.now() - start_time
    print(f"\n{'#'*60}")
    print(f"  Pipeline completed for {CATEGORY}")
    print(f"  Total time: {elapsed}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
