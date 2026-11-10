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

async def reviewer_node(state: TaskState, config=None):
    reviewer_llm = get_reviewer_llm()
    plan = state["plan"]
    results = state["results"]
    reviews = state.get("subtask_reviews") or [None] * len(plan)
    revision = state.get("revision_count", 0)
    next_revision = revision + 1
    is_last_iteration = next_revision >= MAX_REVISIONS  # после этой итерации доработок не будет

    context = "\n".join(
        f"{i}. {plan[i]}\n   Результат: {r or '—'}\n   Предварительная проверка: {reviews[i] or 'нет'}"
        for i, r in enumerate(results)
    )
    iteration_line = f"Итерация ревью: {next_revision} из {MAX_REVISIONS}"
    note = ("\n" + iteration_line +
            "\nВнимание: это последняя итерация — доработок не будет."
            if is_last_iteration else "\n" + iteration_line)

    def _validate(review):
        # Итоговый ответ обязателен, когда результат одобрен или когда доработок больше не будет
        if (review.is_satisfactory or is_last_iteration) and not review.final_answer.strip():
            raise ValueError("Reviewer должен вернуть итоговый ответ")
        # Замечания — словарь {индекс подзадачи: что перепланировать}: при отказе он обязателен,
        # кроме последней итерации с уже собранным ответом (перепланировать всё равно нечего)
        if (not review.is_satisfactory and not review.feedback
                and not (is_last_iteration and review.final_answer.strip())):
            raise ValueError("При неудовлетворительном результате feedback не может быть пустым: "
                             "верни словарь {индекс подзадачи: что именно перепланировать}")

    async def _review_once():
        response = await ainvoke_with_retry(
            reviewer_llm,
            [SystemMessage(content=FINAL_REVIEW_SYSTEM),
             HumanMessage(content=f"""
            Исходная задача: {state['original_task']}
            Результаты подзадач и предварительная пошаговая проверка:
            {context}{note}""")],
            schema=ReviewResult, validate=_validate, description="итоговое ревью")
        return response

    try:
        review = await _review_once()
    except Exception as e:
        # Ревьюер не ответил: отдаем частичный результат, если он есть
        logger.error("Итоговое ревью не удалось: %s", e)
        partial = build_partial_answer(state, note=f"итоговая проверка не удалась: {e}")
        if not partial:
            raise RuntimeError(f"Не удалось получить ответ от ревьюера: {e}")
        return {"final_answer": partial, "feedback": {0: f"итоговая проверка не удалась: {e}"}, "revision_count": next_revision}

    if review.is_satisfactory:
        # Ответ одобрен: слитый результат работы
        return {
            "final_answer": review.final_answer,
            "feedback": None,
            "revision_count": next_revision,
        }

    if is_last_iteration:
        # Это была последняя итерация: ревьюер обязан был дать осмысленный ответ,
        # но если он вернул пустоту — сохраняем хоть что-то
        if not review.final_answer.strip():
            partial = build_partial_answer(state, note="доработки исчерпаны, ответа нет")
            return {
                "final_answer": partial or "Не удалось получить ответ от ревьюера.",
                "feedback": review.feedback,
                "revision_count": next_revision,
            }
        return {
            "final_answer": review.final_answer,
            "feedback": review.feedback,
            "revision_count": next_revision,
        }

    # Ответ неудовлетворителен: доработка с учётом замечаний
    return {
        "final_answer": None,
        "feedback": review.feedback,
        "revision_count": next_revision,
    }