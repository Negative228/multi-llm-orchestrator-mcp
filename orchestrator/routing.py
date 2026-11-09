import asyncio
import logging
from typing import TypedDict, List, Optional

from .state import TaskState
from .config import *

logger = logging.getLogger(__name__)

async def ainvoke_with_retry(llm, messages, schema=None, validate=None, max_attempts=MAX_LLM_ATTEMPTS):
    """Вызов LLM с повторами при ошибках API, пустом ответе и некорректном JSON.

    messages — строка или список сообщений (SystemMessage + HumanMessage):
    постоянная часть промпта выносится в SystemMessage ради кеширования префикса.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            if schema is not None:
                result = await llm.with_structured_output(schema).ainvoke(messages)
            else:
                result = await llm.ainvoke(messages)
                if not str(result.content or "").strip():
                    raise ValueError("модель вернула пустой ответ")
            if validate is not None:
                validate(result)
            return result
        except Exception as e:
            last_error = e
            logger.warning("Вызов LLM (попытка %d/%d) не удался: %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    raise RuntimeError(f"LLM не ответила после {max_attempts} попыток: {last_error}") from last_error

def build_partial_answer(state: TaskState, note: str = "") -> str:
    """Частичный ответ из уже полученных результатов, когда задача решена не полностью.

    Включает содержимое всех выполненных подзадач (с пометкой статуса, если не OK),
    перечисляет невыполненные и добавляет замечания ревьюера.
    """
    plan = state.get("plan") or []
    results = state.get("results") or []
    statuses = state.get("subtask_statuses") or []
    reviews = state.get("subtask_reviews") or []

    done, flagged, missing = [], [], []
    for i, res in enumerate(results):
        desc = plan[i] if i < len(plan) else f"подзадача {i}"
        status = statuses[i] if i < len(statuses) else STATUS_PENDING
        review_note = reviews[i] if i < len(reviews) else None
        if res is None:
            missing.append(f"{i + 1}. {desc} - \"{status}\"")
        elif status in (STATUS_OK, STATUS_REUSED):
            done.append(f"{i + 1}. {desc}\n{res}")
        else:
            comment = f" - \"{status}\"" + (f" ({review_note})" if review_note else "")
            flagged.append(f"{i + 1}. {desc}{comment}\n{res}")

    if not (done or flagged or missing):
        return ""

    parts = ["⚠ ЧАСТИЧНЫЙ РЕЗУЛЬТАТ — задача решена не полностью, ниже показано то, что получилось."]
    if done:
        parts.append("\n\n".join(done))
    if flagged:
        parts.append("Подзадачи с замечаниями (содержимое может быть неполным):\n\n" + "\n\n".join(flagged))
    if missing:
        parts.append("Не выполнены:\n" + "\n".join(missing))
    if note:
        parts.append(f"Замечания ревьюера: {note}")
    return "\n\n".join(parts)


def should_continue(state: TaskState):
    if state.get("final_answer"):
        return "end"
    if state["revision_count"] >= MAX_REVISIONS:
        return "end"
    return "revise"

def after_planner(state: TaskState):
    # Планировщик мог вернуть частичный результат (доработка не удалась) — тогда завершаем
    return "end" if state.get("final_answer") else "executor"