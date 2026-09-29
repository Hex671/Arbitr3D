"""
RES（指代分割）任务的所有 MLLM Prompt 模板集中管理。

设计原则：
- 与 Arbitr3D 现有 K_u / K_0 schema 严格对齐（spatial_location / typical_3D_shape /
  connection_context / relative_size / z_height_range），便于复用现有 prompt 渲染逻辑。
- caption 中的"渲染艺术品词"（close-up / detached / isolated / extracted from / shadow /
  textured surface）由 prompt 显式忽略，避免污染 K_u。
- 所有 prompt 强制 JSON 输出，下游解析有重试与兜底逻辑。

不会被 PartNetE / PartObjaverse-Tiny 路径 import。
"""

from typing import List, Dict


# =============================================================================
# Step 2.A — Target K_u Reconstruction
# =============================================================================
TARGET_RECONSTRUCT_PROMPT_TMPL = """你是 3D 部件知识专家。给定一段对某 3D 物体某 part 的自然语言描述，请重构出该 part 的结构化知识字段，以便下游指代分割任务使用。

【caption】
brief:    {brief}
detailed: {detailed}

【⚠️ 噪声词处理】
caption 可能含 "close-up", "detached", "isolated", "extracted from", "view of", "image shows" 等渲染相关词。这些词描述的是渲染呈现方式而非 3D 结构本身——请**忽略**它们，只关注 part 本身的形状/材质/位置/功能。

【输出 schema】（严格 JSON，与现有 unified_knowledge 字段对齐）
{{
  "short_label": "<2-4 词的简称，例如 'frog leg', 'oven door', 'mouse wheel'。必须简洁，能在 SOM 标注上一眼识别>",
  "spatial_location": "<自然语言描述部件在整体中的空间位置；caption 没明说就写 '从描述无法判断'>",
  "typical_3D_shape": "<自然语言描述形状/几何特征>",
  "connection_context": "<自然语言描述与其他部件的连接关系；caption 没明说就写 '从描述无法判断'>",
  "relative_size": "<自然语言描述相对大小（如 '小', '中等', '较大'）；caption 没明说就写 '从描述无法判断'>",
  "z_height_range": "<[low, high] 数值数组，例如 [0.0, 0.3]；先读 spatial_location 中含的方位词（'top'/'upper'/'bottom'/'lower'/'middle'）再给区间；如完全无法判断写 [0.0, 1.0]>"
}}

【判别原则】
1. 不强行推断；caption 没明说就写"从描述无法判断"
2. short_label 必须简洁干净，去掉冠词与多余修饰
3. z_height_range 是相对整体物体高度（Y 轴）的归一化区间，0=最底部，1=最顶部
4. **不要**输出 view_appearance 字段
5. **不要**输出任何额外字段

只输出 JSON。
"""


# =============================================================================
# Step 2.B — Distractor Discovery from 4-view collage
# =============================================================================
DISTRACTOR_DISCOVERY_PROMPT_TMPL = """你是 3D 物体部件分析专家。下图为同一 3D 物体的 4 视角渲染拼图（左上=正面 az=0°，右上=右侧 az=90°，左下=背面 az=180°，右下=左侧 az=270°，仰角统一为 {elevation}°）。

【任务上下文】
本任务正在做 3D 指代分割。指代目标是该物体上的：「{target_label}」
  - 空间位置: {target_spatial}
  - 形状: {target_shape}
请你判断该物体**除了目标部件外**，还可能存在哪些其他部件，并给出每个部件的简要特征（用于辅助下游 mask 分类的 distractor 信息）。

【输出 schema】（严格 JSON）
{{
  "distractor_parts": [
    {{
      "short_label": "<2-3 词简称，必须与目标部件 short_label 不同>",
      "brief_features": "<一句话描述形状/位置/与其他部件关系，方便下游识别，≤30 字>"
    }}
  ]
}}

【原则】
1. 只列出在 4 视角中实际能看到的、与目标明显不同的部件
2. 数量限制 ≤ {max_distractors} 个，过多易冲淡注意力
3. short_label 必须与目标的 "{target_label}" 不重复（也避免单纯子串包含）
4. brief_features 一句话内（≤30 字），包含形状/位置/连接关系等关键信息
5. 不要列出整个物体本身（如 "frog body" 当目标是 frog leg 时可以列，但 "the frog" 不可以）

只输出 JSON。
"""


# =============================================================================
# Step 3 — First-Round RES Prompt（注入到 MLLMClassifier 现有 prompt 中作为上下文头）
# =============================================================================
RES_TASK_INTRO_TMPL = """⚠️ 本任务是 3D **指代分割（Referring Expression Segmentation）**：
用户给定一个目标部件描述，你需要判断每个 mask 是否属于该目标部件，或属于本物体上的其他干扰部件，或无法确定。

【指代目标】
  short_label: {target_label}
  spatial_location: {target_spatial}
  typical_3D_shape: {target_shape}
  connection_context: {target_connection}
  relative_size: {target_size}
  z_height_range: {target_z_range}

【其他可能存在的干扰部件】
{distractor_lines}

【label 输出规则（严格遵守）】
- 如果 mask 属于指代目标，输出 "{target_label}"
- 如果 mask 属于上述某个干扰部件，输出对应的 short_label
- 如果 mask 无法判断归属（既不像目标也不像任何干扰），输出 "unknown"
- 不要使用 "background" 或 "unlabeled"，统一用 "unknown" 代表无法归属

"""


# =============================================================================
# Step 4 — Online Topology Inference Prompt
# =============================================================================
TOPOLOGY_INFER_PROMPT_TMPL = """你是 3D 部件空间关系分析专家。下图为同一 3D 物体的 4 视角渲染拼图（左上=正面 az=0°，右上=右侧 az=90°，左下=背面 az=180°，右下=左侧 az=270°，仰角统一为 {elevation}°）。

【任务】
该物体已识别出以下部件语义（可能存在于第一轮 RES 推理结果中）：
{hit_semantics_lines}

请基于 4 视角渲染图，给出这些部件之间合理的：
1. **height_order**：哪个部件在常规摆放下应在另一个部件之上（height_order[A] = [B, C] 表示 A 通常高于 B 和 C）
2. **adjacency**：哪些部件对**直接接触/连接**

【输出 schema】（严格 JSON）
{{
  "height_order": {{
    "<part_label_A>": ["<比 A 低的 part>", ...],
    "<part_label_B>": [...]
  }},
  "adjacency": [
    ["<part_label_A>", "<part_label_B>"]
  ]
}}

【原则】
1. height_order: 只写有明显高低关系的对；同高度（如左右对称的两 arm）不要硬排
2. adjacency: 仅列出直接接触/连接的部件对；间接关系不要列
3. 不要输出 hit_semantics 之外的 part 名称
4. 部件名严格使用上面给出的 short_label
5. 输出空 dict / 空 list 都是合法的

只输出 JSON。
"""


# =============================================================================
# Helper formatters
# =============================================================================
def format_distractor_lines(distractors: List[Dict[str, str]]) -> str:
    """把 distractor list 格式化为 prompt 字符串."""
    if not distractors:
        return "  (无 distractor)"
    return "\n".join(
        f"  - {d.get('short_label', '?')}: {d.get('brief_features', '')}"
        for d in distractors
    )


def format_hit_semantics_lines(
    target_label: str,
    target_count: int,
    distractor_labels: List[str],
) -> str:
    """把命中语义集合格式化为 prompt 字符串."""
    lines = [f"  - {target_label} (target)"]
    for d in distractor_labels:
        if d == target_label:
            continue
        lines.append(f"  - {d}")
    return "\n".join(lines)


def build_res_task_intro(
    target_ku: Dict[str, str],
    distractors: List[Dict[str, str]],
) -> str:
    """组装 RES 任务上下文头（用于第一轮 RES MLLM prompt）."""
    return RES_TASK_INTRO_TMPL.format(
        target_label=target_ku.get("short_label", "target"),
        target_spatial=target_ku.get("spatial_location", "未提及"),
        target_shape=target_ku.get("typical_3D_shape", "未提及"),
        target_connection=target_ku.get("connection_context", "未提及"),
        target_size=target_ku.get("relative_size", "未提及"),
        target_z_range=target_ku.get("z_height_range", "[0.0, 1.0]"),
        distractor_lines=format_distractor_lines(distractors),
    )
