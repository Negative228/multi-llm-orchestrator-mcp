import asyncio
import logging
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import TaskState
from ..config import *
from ..schemas import *
from ..callbacks import _emit_progress
from ..routing import ainvoke_with_retry

logger = logging.getLogger(__name__)


async def executor_node(state: TaskState, config=None):
    executor_llm = get_executor_llm()
    plan = state["plan"]
    deps_all = state.get("depends_on") or [[] for _ in plan]
    results: List[Optional[str]] = list(state.get("results") or [])
    reviews: List[Optional[str]] = list(state.get("subtask_reviews") or [])
    statuses: List[str] = list(state.get("subtask_statuses") or [])
    results += [None] * (len(plan) - len(results))
    reviews += [None] * (len(plan) - len(reviews))
    statuses += [STATUS_PENDING] * (len(plan) - len(statuses))

    exec_sem = asyncio.Semaphore(MAX_PARALLEL_SUBTASKS)
    review_sem = asyncio.Semaphore(MAX_PARALLEL_REVIEWS)
    futures: dict = {}        # индекс подзадачи -> задача её выполнения
    review_tasks: List[asyncio.Task] = []

    async def set_status(i: int, status: str):
        statuses[i] = status
        await _emit_progress(config, {
            "type": "status", "index": i, "description": plan[i], "status": status,
        })

    async def preliminary_review(i: int, result: str, dep_indices: List[int]):
        # Пошаговая проверка запускается сразу и работает параллельно с выполнением следующих подзадач
        dep_lines = "\n".join(f"{d}. {plan[d]}: {results[d] or '—'}" for d in dep_indices) or "нет"
        # Общая задача: {state['original_task']}
        messages = [
            SystemMessage(content=SUBTASK_REVIEW_SYSTEM),
            HumanMessage(content=f"""
            Подзадача {i}: {plan[i]}
            Результат: {result}
            Результаты подзадач, от которых она зависит:
            {dep_lines}"""),
        ]
        try:
            async with review_sem:
                rev = await ainvoke_with_retry(executor_llm, messages, schema=SubtaskReview)
            if rev.is_satisfactory:
                reviews[i] = "OK"
                await set_status(i, STATUS_OK)
            else:
                reviews[i] = f"Замечание: {rev.feedback}"
                await set_status(i, STATUS_FAILED)
        except Exception as e:
            logger.warning("Предварительная проверка подзадачи %d не удалась: %s", i, e)
            reviews[i] = "предварительная проверка не удалась"
            await set_status(i, STATUS_REVIEW_ERROR)

    async def run_subtask(i: int):
        dep_indices = [d for d in deps_all[i] if 0 <= d < len(plan) and d < i]
        # Ждём выполнения подзадач-зависимостей (переиспользованные результаты уже готовы)
        for d in dep_indices:
            task = futures.get(d)
            if task is not None:
                await task
        dep_lines = "\n".join(f"{d}. {plan[d]}: {results[d] or '—'}" for d in dep_indices)
        # Общая задача (для справки): {state['original_task']}
        user_prompt = f"Подзадача (необходимо выполнить): {plan[i]}"
        if dep_indices:
            user_prompt += f"""
            Результаты подзадач, от которых она зависит:
            {dep_lines}
            Верни только новый/измененный фрагмент, без повторения результатов других подзадач, если не указано обратное"""
        else:
            user_prompt += "\nПодзадача не зависит от других подзадач."
        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
        await set_status(i, STATUS_IN_PROGRESS)
        try:
            async with exec_sem:
                response = await ainvoke_with_retry(executor_llm, messages)
        except Exception as e:
            # Не роняем весь прогон: помечаем подзадачу, ревью отклонит результат
            # и цикл доработки попробует выполнить её заново.
            logger.error("Подзадача %d не выполнена: %s", i, e)
            results[i] = None
            await set_status(i, STATUS_ERROR)
            return None
        result = response.content
        results[i] = result
        review_tasks.append(asyncio.create_task(preliminary_review(i, result, dep_indices)))
        return result

    # Запускаем все невыполненные подзадачи сразу: независимые выполняются параллельно,
    # зависимые стартуют, как только готовы результаты подзадач, от которых зависят.
    for i in range(len(plan)):
        if results[i] is None:
            futures[i] = asyncio.create_task(run_subtask(i))

    outcomes = await asyncio.gather(*futures.values(), return_exceptions=True)
    for (i, _), outcome in zip(list(futures.items()), outcomes):
        if isinstance(outcome, Exception):
            # Неожиданная ошибка одной подзадачи не должна терять результаты остальных
            logger.error("Подзадача %d завершилась неожиданной ошибкой: %s", i, outcome)
            results[i] = None
            await set_status(i, STATUS_ERROR)
    if review_tasks:
        await asyncio.gather(*review_tasks, return_exceptions=True)
    return {"results": results, "subtask_reviews": reviews, "subtask_statuses": statuses}