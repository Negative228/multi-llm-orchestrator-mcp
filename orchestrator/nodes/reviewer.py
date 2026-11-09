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
    messages = [
        SystemMessage(content=FINAL_REVIEW_SYSTEM),
        HumanMessage(content=context),
    ]

    def _validate(review):
        if review.is_satisfactory and not review.final_answer.strip():
            raise ValueError("Reviewer одобрил результат, но не вернул итоговый ответ")

    next_revision = state.get("revision_count", 0) + 1
    try:
        review = await ainvoke_with_retry(reviewer_llm, messages, schema=ReviewResult, validate=_validate)
    except Exception as e:
        # Итоговая проверка недоступна — возвращаем хотя бы уже полученное
        logger.error("Итоговое ревью не удалось, возвращаем частичный результат: %s", e)
        partial = build_partial_answer(state, note=f"итоговая проверка не удалась: {e}")
        if not partial:
            raise
        return {"final_answer": partial, "feedback": f"итоговая проверка не удалась: {e}",
                "revision_count": next_revision}

    satisfactory = review.is_satisfactory
    if satisfactory:
        return {"final_answer": review.final_answer.strip(), "feedback": None,
                "revision_count": next_revision}
    if next_revision >= MAX_REVISIONS:
        # Итерации исчерпаны — возвращаем частичный результат вместе с замечаниями
        partial = build_partial_answer(state, note=review.feedback)
        return {"final_answer": partial or None, "feedback": review.feedback,
                "revision_count": next_revision}
    return {"final_answer": None, "feedback": review.feedback, "revision_count": next_revision}