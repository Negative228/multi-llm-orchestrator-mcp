from typing import TypedDict, List, Optional, Dict

class TaskState(TypedDict):
    original_task: str
    plan: List[str]
    depends_on: List[List[int]]           # для каждой подзадачи — индексы подзадач, результаты которых нужны
    results: List[Optional[str]]          # None — подзадачу нужно выполнить; строка — готовый результат
    subtask_reviews: List[Optional[str]]  # предварительная проверка результата каждой подзадачи
    subtask_statuses: List[str]           # статус каждой подзадачи (см. STATUS_*)
    feedback: Optional[Dict[int, str]]    # замечания Reviewer'а {индекс подзадачи: что перепланировать}
    final_answer: Optional[str]
    revision_count: int