from langgraph.graph import StateGraph, END

from .nodes import *
from .state import TaskState
from .routing import should_continue, after_planner

def create_app():
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

    return workflow.compile()