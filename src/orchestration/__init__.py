"""Three interchangeable MAS architectures over the same five agents."""


def get_orchestrator(architecture: str):
    from src.orchestration import hierarchical_orchestrator, parallel_orchestrator, sequential_orchestrator

    return {
        "hierarchical": hierarchical_orchestrator.run,
        "sequential": sequential_orchestrator.run,
        "parallel": parallel_orchestrator.run,
    }[architecture]
