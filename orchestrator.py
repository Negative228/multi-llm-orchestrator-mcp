# orchestrator.py
import os
import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Optional
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Настройки устойчивости и параллелизма ---
MAX_LLM_ATTEMPTS = 3       # попыток вызова LLM (ошибки API, пустой ответ, битый JSON)
RETRY_BASE_DELAY = 1.0     # базовая задержка между повторами, сек (экспоненциальный backoff)
MAX_REVISIONS = 2          # максимум итераций «планирование → выполнение → ревью»
MAX_PARALLEL_SUBTASKS = 4  # одновременно выполняемых подзадач
MAX_PARALLEL_REVIEWS = 4   # одновременных предварительных проверок
RESULT_PREVIEW_LEN = 1000  # обрезка результатов в промпте повторного планирования

# --- Статусы подзадач (для отображения прогресса) ---
STATUS_PENDING = "PENDING"          # ещё не начата
STATUS_IN_PROGRESS = "IN_PROGRESS"  # выполняется / ревью ещё не вызвано
STATUS_OK = "OK"                    # предварительная проверка пройдена
STATUS_FAILED = "FAILED"            # у ревьюера есть замечания
STATUS_REVIEW_ERROR = "REVIEW_ERROR"  # предварительная проверка не удалась (ошибка)
STATUS_ERROR = "ERROR"              # подзадача не выполнена (ошибка после ретраев)
STATUS_REUSED = "REUSED"            # результат переиспользован из прошлой итерации

# --- Определение состояния графа ---
class TaskState(TypedDict):
    original_task: str
    plan: List[str]
    depends_on: List[List[int]]           # для каждой подзадачи — индексы подзадач, результаты которых нужны
    results: List[Optional[str]]          # None — подзадачу нужно выполнить; строка — готовый результат
    subtask_reviews: List[Optional[str]]  # предварительная проверка результата каждой подзадачи
    subtask_statuses: List[str]           # статус каждой подзадачи (см. STATUS_*)
    feedback: Optional[str]               # замечания Reviewer'а для перепланирования
    final_answer: Optional[str]
    revision_count: int

# --- Модели ---
planner_llm = ChatOpenAI(
    model=os.getenv("BIG_LLM_MODEL"),
    api_key=os.getenv("BIG_LLM_API_KEY"),
    base_url="https://polza.ai/api/v1/",
    temperature=0.1,
    timeout=300,
    max_retries=0,  # повторы контролируем сами в ainvoke_with_retry
)

executor_llm = ChatOpenAI(
    model=os.getenv("LIGHT_LLM_MODEL"),
    api_key=os.getenv("LIGHT_LLM_API_KEY"),
    base_url="https://polza.ai/api/v1/",
    temperature=0.3,
    timeout=120,
    max_retries=0,
)

# --- Схемы для структурированного вывода ---
class PlannedSubtask(BaseModel):
    description: str = Field(description="Конкретная подзадача для выполнения")
    depends_on: List[int] = Field(
        default_factory=list,
        description="Индексы подзадач (с 0), результаты которых нужны для выполнения этой "
                    "(только предыдущие). Пусто — подзадача независима и может выполняться параллельно",
    )
    reuse_result_from: Optional[int] = Field(
        default=None,
        description="Индекс подзадачи из предыдущего плана (с 0), чей результат подходит "
                    "и переиспользуется без повторного выполнения; None — выполнить заново",
    )

class Plan(BaseModel):
    subtasks: List[PlannedSubtask] = Field(description="Список атомарных подзадач для выполнения")

class SubtaskReview(BaseModel):
    is_satisfactory: bool = Field(description="Корректно и полно ли выполнена подзадача")
    feedback: str = Field(description="Что именно нужно исправить; пусто, если всё в порядке")

class ReviewResult(BaseModel):
    is_satisfactory: bool = Field(description="Удовлетворителен ли итоговый результат")
    feedback: str = Field(description="Что именно не так и какие подзадачи нужно перепланировать/переделать; пусто, если всё в порядке")
    final_answer: str = Field(description="Итоговый ответ пользователю: результаты подзадач, объединённые в один цельный "
                                          "ответ без повторов и дублирования кода (каждый фрагмент кода — только один раз). "
                                          "Пусто, если результат неудовлетворителен")

# --- Постоянные части промптов (SystemMessage) ---
# Инструкции вынесены в константы ради кеширования префикса у провайдера:
# чем стабильнее начальная часть запроса между вызовами, тем больше токенов
# берётся из кеша (см. cache_read в usage-отчёте). Переменные данные (задача,
# подзадачи, результаты) идут только в HumanMessage.
PLANNER_SYSTEM = """Ты — архитектор. Разбей следующую задачу на последовательность небольших,
конкретных подзадач, которые сможет выполнить другая, менее мощная модель.
Подзадачи не должны включать в себя проверку собственных результатов или
результатов, полученных в других подзадачах, если об этом напрямую
не указано в формулировке задачи.
Для каждой подзадачи укажи depends_on — индексы подзадач (с 0), результаты которых ей нужны.
Если подзадачи полностью независимы друг от друга (например, анализ нескольких источников),
оставь depends_on пустым — тогда они будут выполняться параллельно. Если подзадача строится
на результатах предыдущих (например, шаги вычислений), укажи эти индексы.
Верни только список подзадач."""

PLANNER_REPLAN_SYSTEM = """Ты — архитектор. Ревьюер остался недоволен результатом работы — перепланируй задачу.
Верни исправленный план целиком. Оставь без изменений подзадачи, результаты которых не требуют
доработки — для них укажи reuse_result_from (индекс подзадачи из предыдущего плана, чей результат
переиспользуется). Переформулируй или добавь только те подзадачи, на которые указывают замечания.
Для каждой подзадачи укажи depends_on — индексы подзадач, результаты которых ей нужны (пусто —
подзадача независима и может выполняться параллельно с другими)."""

EXECUTOR_SYSTEM = """Выполни подзадачу в рамках общей задачи.
Не добавляй пояснений/примеров, если это прямо не указано в твоей подзадаче."""

SUBTASK_REVIEW_SYSTEM = """Ты — ревьюер. Проверь только эту подзадачу и её результат.
Если результат некорректен или неполон, укажи, что именно нужно исправить."""

FINAL_REVIEW_SYSTEM = """Ты — строгий ревьюер. Оцени, решают ли предоставленные результаты исходную задачу.
Пошаговая проверка каждой подзадачи уже проведена (см. предварительную проверку) — учти её замечания
и проверь итог целиком. Если результат неудовлетворителен, подробно укажи, что именно не так
и какие подзадачи нужно перепланировать и переделать.
Если результат удовлетворителен — собери из результатов подзадач единый итоговый ответ пользователю
(final_answer): объедини их в один цельный документ, убрав повторы и дублирование кода (один и тот же
код не должен встречаться дважды), без своих дополнительных комментариев."""

# --- Устойчивый вызов LLM ---
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

# --- Узлы графа ---
async def planner_node(state: TaskState, config=None):
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

async def executor_node(state: TaskState, config=None):
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
            HumanMessage(content=f"""Подзадача {i}: {plan[i]}
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

async def reviewer_node(state: TaskState, config=None):
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
        review = await ainvoke_with_retry(planner_llm, messages, schema=ReviewResult, validate=_validate)
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

def should_continue(state: TaskState):
    if state.get("final_answer"):
        return "end"
    if state["revision_count"] >= MAX_REVISIONS:
        return "end"
    return "revise"

def after_planner(state: TaskState):
    # Планировщик мог вернуть частичный результат (доработка не удалась) — тогда завершаем
    return "end" if state.get("final_answer") else "executor"

# --- Сборка графа ---
workflow = StateGraph(TaskState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner",
    after_planner,
    {"end": END, "executor": "executor"},
)
workflow.add_edge("executor", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {"end": END, "revise": "planner"},  # при доработке возвращаемся к планировщику с feedback
)

orchestrator_app = workflow.compile()
