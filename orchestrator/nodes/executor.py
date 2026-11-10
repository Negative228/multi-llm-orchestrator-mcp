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
    sub_reviewer_llm = get_subtask_reviewer_llm()
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
    futures: dict = {}  # индекс подзадачи -> задача полного цикла (выполнение + проверка + ремонт)

    async def set_status(i: int, status: str):
        statuses[i] = status
        await _emit_progress(config, {
            "type": "status", "index": i, "description": plan[i], "status": status,
        })

    def build_exec_prompt(i: int, dep_indices: List[int], fix_feedback: Optional[str] = None):
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
        if fix_feedback:
            user_prompt += f"""
            Предыдущая попытка была отклонена ревьюером. Замечания, которые нужно учесть:
            {fix_feedback}
            Выполни подзадачу заново, исправив указанные недостатки."""
        return [SystemMessage(content=EXECUTOR_SYSTEM), HumanMessage(content=user_prompt)]

    def build_review_messages(i: int, result: str, dep_indices: List[int]):
        dep_lines = "\n".join(f"{d}. {plan[d]}: {results[d] or '—'}" for d in dep_indices) or "нет"
        # Общая задача: {state['original_task']}
        return [
            SystemMessage(content=SUBTASK_REVIEW_SYSTEM),
            HumanMessage(content=f"""
            Подзадача {i}: {plan[i]}
            Результат: {result}
            Результаты подзадач, от которых она зависит:
            {dep_lines}"""),
        ]

    async def run_subtask(i: int):
        dep_indices = [d for d in deps_all[i] if 0 <= d < len(plan) and d < i]
        # Ждём полного завершения зависимостей: выполнение + проверка + возможный ремонт
        for d in dep_indices:
            task = futures.get(d)
            if task is not None:
                try:
                    await task
                except Exception:
                    pass  # статус зависимости уже установлен, см. проверку ниже
        # A2: не запускаем подзадачу, пока зависимости не подтверждены
        unmet = [d for d in dep_indices if statuses[d] not in (STATUS_OK, STATUS_REUSED)]
        if unmet:
            logger.info("Подзадача %d заблокирована: зависимости %s не подтверждены", i, unmet)
            results[i] = None
            await set_status(i, STATUS_BLOCKED)
            return None

        await set_status(i, STATUS_IN_PROGRESS)
        try:
            async with exec_sem:
                response = await ainvoke_with_retry(executor_llm, build_exec_prompt(i, dep_indices))
        except Exception as e:
            # Не роняем весь прогон: помечаем подзадачу, доработка попробует выполнить её заново
            logger.error("Подзадача %d не выполнена: %s", i, e)
            results[i] = None
            await set_status(i, STATUS_ERROR)
            return None
        result = response.content
        results[i] = result

        # Предварительная проверка + (если ошибка локальная) ремонт на месте
        for fix_round in range(MAX_LOCAL_FIXES + 1):
            try:
                async with review_sem:
                    rev = await ainvoke_with_retry(
                        sub_reviewer_llm, build_review_messages(i, result, dep_indices), schema=SubtaskReview)
            except Exception as e:
                logger.warning("Предварительная проверка подзадачи %d не удалась: %s", i, e)
                reviews[i] = "предварительная проверка не удалась"
                await set_status(i, STATUS_REVIEW_ERROR)
                return result

            if rev.is_satisfactory:
                reviews[i] = "OK" if fix_round == 0 else "OK (исправлено на месте)"
                await set_status(i, STATUS_OK)
                return result

            if not rev.can_fix_locally or fix_round >= MAX_LOCAL_FIXES:
                # Глобальная ошибка или ремонт исчерпан — уходит в доработку планировщиком
                reviews[i] = f"Замечание: {rev.feedback}"
                await set_status(i, STATUS_FAILED)
                return result

            # Ошибка локальная — ремонт на месте: повторяем подзадачу с учётом замечаний
            await set_status(i, STATUS_FIXING)
            try:
                async with exec_sem:
                    response = await ainvoke_with_retry(
                        executor_llm, build_exec_prompt(i, dep_indices, fix_feedback=rev.feedback))
                result = response.content
                results[i] = result
            except Exception as e:
                logger.error("Ремонт подзадачи %d не удался: %s", i, e)
                reviews[i] = f"Замечание: {rev.feedback} (повторное выполнение не удалось)"
                await set_status(i, STATUS_FAILED)
                return result
        return result

    # Запускаем все невыполненные подзадачи сразу: независимые выполняются параллельно,
    # зависимые стартуют, как только завершён полный цикл подзадач, от которых зависят.
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
    return {"results": results, "subtask_reviews": reviews, "subtask_statuses": statuses}
