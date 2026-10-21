"""LangGraph workflow: ingest -> score -> features -> evaluate -> forecast -> report.
Falls back to a plain sequential run if langgraph is not installed."""
from typing import TypedDict
from src.pipeline import Pipeline


class State(TypedDict, total=False):
    log: list


def run_graph(pipe: Pipeline):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        print("[graph] langgraph not installed, running sequentially")
        return pipe.run()

    def node(name):
        def _f(state: State):
            msg = getattr(pipe, name)()
            print(f"[{name}] {msg}")
            return {"log": (state.get("log") or []) + [f"{name}: {msg}"]}
        return _f

    g = StateGraph(State)
    for n in Pipeline.NODES:
        g.add_node(n, node(n))
    g.set_entry_point(Pipeline.NODES[0])
    for a, b in zip(Pipeline.NODES, Pipeline.NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(Pipeline.NODES[-1], END)
    g.compile().invoke({"log": []})
    return pipe
