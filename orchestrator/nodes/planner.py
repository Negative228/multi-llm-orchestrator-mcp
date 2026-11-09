import asyncio
import logging
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import TaskState
from ..config import *
from ..schemas import *
from ..callbacks import _emit_progress
from ..routing import ainvoke_with_retry, build_partial_answer

logger = logging.getLogger(__name__)

async def planner_node(state: TaskState, config=None):
    planner_llm = get_planner_llm()
    feedback = state.get("feedback")
    if feedback:
        # Повторное планирование с учётом замечаний Reviewer'а
        old_results = state.get("results") or []
        old_reviews = state.get("subtask_reviews") or []
        previous = []
        for i, subtask in enumerate(state["plan"]):
            result = old_results[i] if i < len(old_results) else None
            if result is None:
                preview = "не выполнялось"
            elif len(result) <= RESULT_PREVIEW_LEN:
                preview = result
            else:
                preview = result[:RESULT_PREVIEW_LEN] + "…"
            note = old_reviews[i] if i < len(old_reviews) and old_reviews[i] else "—"
            previous.append(f"{i}. {subtask}\n   Результат: {preview}\n   Предварительная проверка: {note}")
        messages = [
            SystemMessage(content=PLANNER_REPLAN_SYSTEM),
            HumanMessage(content=f"""Исходная задача: {state['original_task']}

Предыдущий план, результаты подзадач и их предварительная проверка:
{chr(10).join(previous)}

Замечания ревьюера: {feedback}"""),
        ]
    else:
        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Задача: {state['original_task']}"),
        ]

    def _validate(plan):
        if not plan.subtasks:
            raise ValueError("Planner вернул пустой план")

    try:
        plan = await ainvoke_with_retry(planner_llm, messages, schema=Plan, validate=_validate)
    except Exception as e:
        if not feedback:
            raise  # первое планирование — спасти нечего
        # Доработка невозможна (планировщик недоступен) — отдаём уже полученное
        logger.error("Перепланирование не удалось, возвращаем частичный результат: %s", e)
        partial = build_partial_answer(
            state, note=f"доработка не удалась (планировщик не ответил): {e}")
        if not partial:
            raise
        return {"final_answer": partial, "feedback": None,
                "revision_count": state.get("revision_count", 0)}

    # Собираем новый план; результаты и проверки удачных подзадач переиспользуем из прошлой итерации
    old_results = state.get("results") or []
    old_reviews = state.get("subtask_reviews") or []
    new_plan: List[str] = []
    new_deps: List[List[int]] = []
    new_results: List[Optional[str]] = []
    new_reviews: List[Optional[str]] = []
    new_statuses: List[str] = []
    for j, subtask in enumerate(plan.subtasks):
        new_plan.append(subtask.description)
        # зависимости только на предыдущие подзадачи — гарантирует отсутствие циклов
        new_deps.append([d for d in (subtask.depends_on or []) if isinstance(d, int) and 0 <= d < j])
        idx = subtask.reuse_result_from
        if feedback and idx is not None and 0 <= idx < len(old_results) and old_results[idx]:
            new_results.append(old_results[idx])
            new_reviews.append(old_reviews[idx] if idx < len(old_reviews) else None)
            new_statuses.append(STATUS_REUSED)
        else:
            new_results.append(None)
            new_reviews.append(None)
            new_statuses.append(STATUS_PENDING)

    await _emit_progress(config, {
        "type": "plan",
        "iteration": state.get("revision_count", 0) + 1,
        "plan": new_plan,
        "statuses": new_statuses,
    })
    return {
        "plan": new_plan,
        "depends_on": new_deps,
        "results": new_results,
        "subtask_reviews": new_reviews,
        "subtask_statuses": new_statuses,
        "feedback": None,
    }
