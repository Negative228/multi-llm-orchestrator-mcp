from pydantic import BaseModel, Field
from typing import TypedDict, Optional, List

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
                                          "Заполняй, когда результат удовлетворителен; на последней итерации — всегда "
                                          "(лучший возможный ответ из имеющихся результатов, с указанием ограничений). "
                                          "В остальных случаях — пусто")