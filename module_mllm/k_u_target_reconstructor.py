"""
K_u-Dynamic Track A —— 从用户给定的 (brief, detailed) caption 重构出目标部件的
泛化知识库字段（除 view_appearance 外）+ short_label。

输出格式与现有 config/knowledge/<category>_unified_knowledge.json 完全对齐：
    {
        "short_label": "frog leg",
        "spatial_location": "...",
        "typical_3D_shape": "...",
        "connection_context": "...",
        "relative_size": "...",
        "z_height_range": [0.0, 0.3],
    }

注：本模块独立运行，不修改任何现有文件。
"""

import json
import re
import time
from typing import Dict, Optional

from config.res_prompts import TARGET_RECONSTRUCT_PROMPT_TMPL


_DEFAULT_RESULT = {
    "short_label": "target",
    "spatial_location": "从描述无法判断",
    "typical_3D_shape": "从描述无法判断",
    "connection_context": "从描述无法判断",
    "relative_size": "从描述无法判断",
    "z_height_range": [0.0, 1.0],
}


class TargetKURReconstructor:
    """把 caption 重构为 K_u 字段 + short_label。"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            print("[TargetKURReconstructor] OpenAI package not installed; will return defaults.")
            self.client = None

    # ------------------------------------------------------------------
    def reconstruct(
        self,
        brief: str,
        detailed: str,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> Dict:
        """
        基于 (brief, detailed) 重构目标 K_u。失败时返回 _DEFAULT_RESULT 的副本。

        Returns:
            dict（与 _DEFAULT_RESULT 同 schema），保证 7 个 key 都存在。
        """
        if self.client is None:
            return dict(_DEFAULT_RESULT)

        brief_safe = (brief or "").strip()
        detailed_safe = (detailed or "").strip()
        if not brief_safe and not detailed_safe:
            return dict(_DEFAULT_RESULT)

        prompt = TARGET_RECONSTRUCT_PROMPT_TMPL.format(
            brief=brief_safe or "(no brief caption)",
            detailed=detailed_safe or "(no detailed caption)",
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.1,
                    timeout=timeout,
                )
                response_text = (response.choices[0].message.content or "").strip()
                parsed = self._parse_response(response_text)
                if parsed is not None:
                    return self._normalize(parsed)
            except Exception as e:
                print(f"[TargetKURReconstructor] attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt + 1 < max_retries:
                    time.sleep(2)

        print("[TargetKURReconstructor] All retries failed; returning default K_u.")
        return dict(_DEFAULT_RESULT)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(text: str) -> Optional[Dict]:
        """解析 MLLM 响应中的 JSON 对象。容忍代码块包裹与多余文本。"""
        if not text:
            return None
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        # 找最外层 {...}
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize(parsed: Dict) -> Dict:
        """补全缺失字段、规范类型、清理 short_label。"""
        result = dict(_DEFAULT_RESULT)
        for k in result.keys():
            if k in parsed and parsed[k] is not None:
                result[k] = parsed[k]

        # 清洗 short_label
        sl = str(result["short_label"]).strip()
        sl = re.sub(r"^a\s+|^an\s+|^the\s+", "", sl, flags=re.IGNORECASE)
        sl = re.sub(r"\s+", " ", sl)
        result["short_label"] = sl[:40] if sl else "target"

        # 规范 z_height_range
        zr = result["z_height_range"]
        if isinstance(zr, str):
            try:
                import ast
                zr = ast.literal_eval(zr)
            except Exception:
                zr = [0.0, 1.0]
        if not (isinstance(zr, (list, tuple)) and len(zr) == 2):
            zr = [0.0, 1.0]
        try:
            zr = [max(0.0, min(1.0, float(zr[0]))), max(0.0, min(1.0, float(zr[1])))]
            if zr[0] > zr[1]:
                zr = [zr[1], zr[0]]
        except (TypeError, ValueError):
            zr = [0.0, 1.0]
        result["z_height_range"] = zr

        return result
