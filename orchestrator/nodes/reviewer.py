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
    notes = state.get("subtask_reviews") or []
    next_revision = state.get("revision_count", 0) + 1
    is_last_iteration = next_revision >= MAX_REVISIONS

    context = f"Исходная задача: {state['original_task']}\n\n"
    context += "Результаты выполнения подзадач:\n"
    context += "\n".join(
        f"{i}. {plan[i]}\n{res if res is not None else '—'}" for i, res in enumerate(state["results"])
    )
    context += "\n\nПредварительная пошаговая проверка подзадач (проведена во время выполнения):\n"
    context += "\n".join(
        f"{i}. {plan[i]}: {notes[i] if i < len(notes) and notes[i] else 'не проверялось'}"
        for i in range(len(plan))
    )
    context += f"\n\nИтерация ревью: {next_revision} из {MAX_REVISIONS}."
    if is_last_iteration:
        context += "\nЭто последняя итерация: доработок не будет — верни лучший возможный ответ."
    messages = [
        SystemMessage(content=FINAL_REVIEW_SYSTEM),
        HumanMessage(content=context),
    ]

    def _validate(review):
        # Итоговый ответ обязателен, когда результат одобрен или когда доработок больше не будет
        if (review.is_satisfactory or is_last_iteration) and not review.final_answer.strip():
            raise ValueError("Reviewer должен вернуть итоговый ответ")

    try:
        review = await ainvoke_with_retry(reviewer_llm, messages, schema=ReviewResult, validate=_validate)
    except Exception as e:
        # Ревьюер недоступен даже после ретраев — собираем ответ механически (последний рубеж)
        logger.error("Итоговое ревью не удалось, собираем ответ механически: %s", e)
        partial = build_partial_answer(state, note=f"итоговая проверка не удалась: {e}")
        if not partial:
            raise
        return {"final_answer": partial, "feedback": f"итоговая проверка не удалась: {e}",
                "revision_count": next_revision}

    if review.is_satisfactory:
        return {"final_answer": review.final_answer.strip(), "feedback": None,
                "revision_count": next_revision}
    if is_last_iteration:
        # Доработок не будет — возвращаем осмысленный ответ ревьюера, даже если он неполный
        return {"final_answer": review.final_answer.strip(), "feedback": review.feedback,
                "revision_count": next_revision}
    return {"final_answer": None, "feedback": review.feedback, "revision_count": next_revision}