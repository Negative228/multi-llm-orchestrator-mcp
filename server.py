# server.py
from fastmcp import FastMCP, Context
import logging

from pydantic import BaseModel
from langchain_core.callbacks import get_usage_metadata_callback
from orchestrator import    (TaskState, 
                            format_status_report,
                            format_usage_report, 
                            create_app, 
                        )

# Единая настройка логирования для всего приложения
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # убрать шум HTTP-запросов
logging.getLogger("httpx2").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

mcp = FastMCP("Multi-LLM Orchestrator")
orchestrator_app = create_app()

class SolveTaskRequest(BaseModel):
    task: str

@mcp.tool
async def solve_task(request: SolveTaskRequest, ctx: Context = None) -> str:
    """Решает сложную задачу путём декомпозиции и оркестрации нескольких LLM.

    План и прогресс выполнения подзадач отправляются клиенту по ходу работы.
    """
    initial_state: TaskState = {
        "original_task": request.task,
        "plan": [],
        "depends_on": [],
        "results": [],
        "subtask_reviews": [],
        "subtask_statuses": [],
        "feedback": None,
        "final_answer": None,
        "revision_count": 0,
    }

    # Callback прогресса: план целиком — при его появлении, статусы — по мере выполнения.
    # Уведомления MCP (report_progress / info) видны не во всех клиентах, поэтому
    # итоговый отчёт со статусами входит и в сам результат.
    total_subtasks = {"n": 0}

    async def on_progress(event):
        if ctx is None:
            return
        if event["type"] == "plan":
            total_subtasks["n"] = len(event["plan"])
            await ctx.info(format_status_report(
                request.task, event["plan"], event["statuses"], event.get("iteration")))
        elif event["type"] == "status":
            await ctx.report_progress(
                progress=event["index"] + 1,
                total=total_subtasks["n"] or None,
                message=f"{event['index'] + 1}. {event['description']} — {event['status']}",
            )

    usage_cb = None
    try:
        with get_usage_metadata_callback() as usage_cb:
            # Асинхронный вызов графа
            result = await orchestrator_app.ainvoke(
                initial_state, config={"configurable": {"on_progress": on_progress}})
    except Exception:
        logger.exception("Ошибка при выполнении оркестратора")
        usage = format_usage_report(usage_cb.usage_metadata if usage_cb else {})
        return (f"Не удалось обратиться к LLM после нескольких попыток. "
                f"Попробуйте повторить запрос позже.\n\n{usage}")

    usage = format_usage_report(usage_cb.usage_metadata if usage_cb else {})
    report = format_status_report(
        request.task,
        result.get("plan") or [],
        result.get("subtask_statuses") or [],
        result.get("revision_count") or 1,
    )
    if result.get("final_answer"):
        return f"{report}\n\n=== ОТВЕТ ===\n{result['final_answer']}\n\n{usage}"
    message = f"{report}\n\nНе удалось получить удовлетворительный результат."
    if result.get("feedback"):
        message += f"\nПоследние замечания ревьюера: {result['feedback']}"
    return f"{message}\n\n{usage}"

if __name__ == "__main__":
    mcp.run()  # По умолчанию запускается через STDIO
