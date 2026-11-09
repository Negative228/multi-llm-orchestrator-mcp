# test_client.py
import asyncio
import logging
from langchain_core.callbacks import get_usage_metadata_callback
from orchestrator import    (TaskState, 
                            format_status_report,
                            format_usage_report, 
                            create_app, )

# Единая настройка логирования для всего приложения
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # убрать шум HTTP-запросов

async def main():
    task = input("Введите задачу: ")
    orchestrator_app = create_app()
    initial_state: TaskState = {
        "original_task": task,
        "plan": [],
        "depends_on": [],
        "results": [],
        "subtask_reviews": [],
        "subtask_statuses": [],
        "feedback": None,
        "final_answer": None,
        "revision_count": 0,
    }

    # Живой вывод прогресса: план — целиком, статусы — по мере выполнения
    async def on_progress(event):
        if event["type"] == "plan":
            print()
            print(format_status_report(task, event["plan"], event["statuses"], event.get("iteration")))
        elif event["type"] == "status":
            print(f"  [{event['index'] + 1}] {event['description']} — {event['status']}")

    usage_cb = None
    try:
        with get_usage_metadata_callback() as usage_cb:
            result = await orchestrator_app.ainvoke(
                initial_state, config={"configurable": {"on_progress": on_progress}})
    except Exception as e:
        print(f"\nОшибка при обращении к LLM: {e}")
        print()
        print(format_usage_report(usage_cb.usage_metadata if usage_cb else {}))
        return

    print("\n=== ИТОГОВЫЙ ОТВЕТ ===")
    if result.get("final_answer"):
        print(result["final_answer"])
    else:
        print("Не удалось получить результат.")
        if result.get("feedback"):
            print(f"\nПоследние замечания ревьюера: {result['feedback']}")
    print(f"\n(итераций ревью: {result['revision_count']})")
    print()
    print(format_usage_report(usage_cb.usage_metadata if usage_cb else {}))

if __name__ == "__main__":
    asyncio.run(main())
