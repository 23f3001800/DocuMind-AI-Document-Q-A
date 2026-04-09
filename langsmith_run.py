from langchain_core.tracers.context import tracing_v2_enabled


with tracing_v2_enabled(project_name="documind-dev"):
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is backpropagation?")]},
        config={"configurable": {"thread_id": "user_1"}}
    )