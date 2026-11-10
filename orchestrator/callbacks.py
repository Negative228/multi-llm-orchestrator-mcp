import asyncio
import logging
from typing import TypedDict, List, Optional

from .config import STATUS_PENDING

logger = logging.getLogger(__name__)

# --- Прогресс ---
# Callback передаётся через config["configurable"]["on_progress"], а не в state:
# state должен оставаться сериализуемым (иначе сломается checkpointing).
# События: {"type": "plan", "iteration", "plan", "statuses"}
#          {"type": "status", "index", "description", "status"}
async def _emit_progress(config, event: dict):
    cb = (config.get("configurable") or {}).get("on_progress") if config else None
    if cb is None:
        return
    try:
        res = cb(event)
        if asyncio.iscoroutine(res):
            await res
    except Exception as e:
        # сбой отображения прогресса не должен ломать выполнение задачи
        logger.warning("Ошибка callback прогресса: %s", e)

def format_status_report(task: str, plan: List[str], statuses: List[str], iteration: Optional[int] = None) -> str:
    """Текстовый отчёт: задача + план со статусами подзадач."""
    header = "План решения (от Planner):"
    if iteration and iteration > 1:
        header = f"План решения (от Planner, итерация {iteration}):"
    lines = [f"Задача (от пользователя): {task}", header]
    for i, subtask in enumerate(plan):
        status = statuses[i] if i < len(statuses) else STATUS_PENDING
        lines.append(f"{i + 1}. {subtask} - \"{status}\"")
    return "\n".join(lines)

def format_feedback(feedback) -> str:
    """Замечания ревьюера {индекс подзадачи: текст} -> читаемый список."""
    if not feedback:
        return "—"
    if isinstance(feedback, dict):
        def _key(kv):
            try:
                return (0, int(kv[0]))
            except (TypeError, ValueError):
                return (1, 0)
        return "\n".join(f"{idx}. {text}" for idx, text in sorted(feedback.items(), key=_key))
    return str(feedback)

def format_usage_report(usage_metadata: dict) -> str:
    """Сводка по токенам: usage_metadata из UsageMetadataCallbackHandler -> текст.

    Ключи — имена моделей, значения — UsageMetadata (input/output/total_tokens
    плюс детализация cache_read и reasoning).
    """
    if not usage_metadata:
        return "Расход токенов: нет данных"
    lines = ["Расход токенов:"]
    sum_in = sum_out = 0
    for model, usage in usage_metadata.items():
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        sum_in += inp
        sum_out += out
        in_det = usage.get("input_token_details") or {}
        out_det = usage.get("output_token_details") or {}
        extras = []
        if in_det.get("cache_read"):
            extras.append(f"из кеша {in_det['cache_read']}")
        if out_det.get("reasoning"):
            extras.append(f"reasoning {out_det['reasoning']}")
        extra = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"  {model}: вход {inp}, выход {out}{extra}")
    lines.append(f"  ИТОГО: вход {sum_in}, выход {sum_out}, всего {sum_in + sum_out}")
    return "\n".join(lines)