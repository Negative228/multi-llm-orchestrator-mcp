# test_client.py
import asyncio
from orchestrator import orchestrator_app, format_status_report

async def main():
    task = input("Введите задачу: ")
    initial_state = {
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

    try:
        result = await orchestrator_app.ainvoke(
            initial_state, config={"configurable": {"on_progress": on_progress}})
    except Exception as e:
        print(f"\nОшибка при обращении к LLM: {e}")
        return

    print("\n=== ИТОГОВЫЙ ОТВЕТ ===")
    if result.get("final_answer"):
        print(result["final_answer"])
    else:
        print("Не удалось получить результат.")
        if result.get("feedback"):
            print(f"\nПоследние замечания ревьюера: {result['feedback']}")
    print(f"\n(итераций ревью: {result['revision_count']})")

if __name__ == "__main__":
    asyncio.run(main())
