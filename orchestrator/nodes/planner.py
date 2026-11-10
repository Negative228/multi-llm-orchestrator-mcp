import asyncio
import logging
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import TaskState
from ..config import *
from ..schemas import *
from ..callbacks import _emit_progress, format_feedback
from ..routing import ainvoke_with_retry, build_partial_answer

logger = logging.getLogger(__name__)

async def planner_node(state: TaskState, config=None):
    planner_llm = get_planner_llm()
    feedback = state.get("feedback")
    old_results = state.get("results") or []
    old_reviews = state.get("subtask_reviews") or []
    old_statuses = state.get("subtask_statuses") or []
    # A6: переиспользовать можно только подтверждённые результаты,
    # не отмеченные ревьюером как проблемные
    flagged = set(feedback.keys()) if isinstance(feedback, dict) else set()
    if feedback:
        previous = []
        for i, subtask in enumerate(state["plan"]):
            preview = (old_results[i] or "")[:RESULT_PREVIEW_LEN]
            if len(old_results[i] or "") > RESULT_PREVIEW_LEN:
                preview += "\n...[обрезано]"
            note = old_reviews[i] if i < len(old_reviews) and old_reviews[i] else "—"
            previous.append(f"{i}. {subtask}\n   Результат: {preview}\n   Предварительная проверка: {note}")
        messages = [
            SystemMessage(content=PLANNER_REPLAN_SYSTEM),
            HumanMessage(content=f"""
            Исходная задача: {state['original_task']}
            Предыдущий план, результаты подзадач и их предварительная проверка:
            {chr(10).join(previous)}
            Замечания ревьюера (индекс подзадачи: что перепланировать):
            {format_feedback(feedback)}"""),
        ]
    else:
        # Первичное планирование
        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Задача: {state['original_task']}"),
        ]

    try:
        response = await ainvoke_with_retry(
            planner_llm, messages, schema=Plan,
            validate=lambda p: (_ for _ in ()).throw(ValueError("планировщик вернул пустой план"))
            if not p.subtasks else None,
            description="план")
        plan = [s.description for s in response.subtasks]
        depends_on = [list(s.depends_on) for s in response.subtasks]
    except Exception as e:
        # Планировщик не ответил — отдаем частичный результат, если он есть
        logger.error("Планирование не удалось: %s", e)
        partial = build_partial_answer(
            state,
            note=f"доработка не удалась (планировщик не ответил): {e}\n"
                 f"Замечания ревьюера:\n{format_feedback(feedback)}")
        if not partial:
            raise RuntimeError(f"Не удалось получить план от планировщика: {e}")
        return {"final_answer": partial, "feedback": None}

    # Страховка: reuse_result_from осмыслен только при повторном планировании — иначе игнорируем
    if not feedback:
        for subtask in response.subtasks:
            if getattr(subtask, "reuse_result_from", None) is not None:
                logger.warning("Планировщик вернул reuse_result_from при первичном планировании — игнорируем")
                subtask.reuse_result_from = None

    new_results: List[Optional[str]] = []
    new_reviews: List[Optional[str]] = []
    new_statuses: List[str] = []
    for j, subtask in enumerate(response.subtasks):
        # Отбрасываем некорректные ссылки на зависимости (циклы, индексы вне диапазона)
        depends_on[j] = [d for d in depends_on[j] if isinstance(d, int) and 0 <= d < j]

        idx = subtask.reuse_result_from
        old_ok = (feedback and idx is not None and 0 <= idx < len(old_results) and old_results[idx]
                  and idx not in flagged
                  and idx < len(old_statuses) and old_statuses[idx] in (STATUS_OK, STATUS_REUSED))
        if old_ok:
            # Переиспользуем только подтверждённые и не отмеченные ревьюером результаты
            new_results.append(old_results[idx])
            new_reviews.append(old_reviews[idx] if idx < len(old_reviews) and old_reviews[idx] else "OK")
            new_statuses.append(STATUS_REUSED)
        else:
            new_results.append(None)
            new_reviews.append(None)
            new_statuses.append(STATUS_PENDING)

    await _emit_progress(config, {
        "type": "plan",
        "iteration": state.get("revision_count", 0) + 1,
        "plan": plan,
        "statuses": new_statuses,
    })
    return {
        "plan": plan,
        "depends_on": depends_on,
        "results": new_results,
        "subtask_reviews": new_reviews,
        "subtask_statuses": new_statuses,
        "feedback": None,
    }