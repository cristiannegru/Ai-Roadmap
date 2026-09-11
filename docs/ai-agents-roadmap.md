# 🤖 AI Agents & Multi-Agent Systems Roadmap

AI Agents represent the transition of LLMs from simple chatbots to autonomous systems capable of reasoning, planning, utilizing tools, and cooperating to complete complex objectives. This roadmap covers Phase 5E.

---

## 🗺️ Agentic AI Architecture
```
                   ┌──────────────┐
                   │  LLM Brain   │
                   └──────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     ┌─────────┐     ┌─────────┐     ┌─────────┐
     │ Memory  │     │Planning │     │  Tools  │
     │ Short / │     │ ReAct / │     │ Web /   │
     │ Long    │     │ Reflex  │     │ APIs    │
     └─────────┘     └─────────┘     └─────────┘
```

---

## 💻 Runnable Code in This Repo (Chunk 4, offline demo)

```bash
python templates/crewai_team/crew.py --query "What is ReAct?" --docs data/samples/rag_docs
echo '{"id":1,"tool":"list_tables","args":{}}' | python templates/mcp_server/server.py --db /tmp/demo.db
```

| Topic | Code | Try it |
|-------|------|--------|
| Researcher → Writer → Critic crew | `templates/crewai_team/crew.py` | CLI above (`--show-hits` for provenance) |
| MCP-concept SQLite tools | `templates/mcp_server/server.py` | `list_tables` / `describe_table` / `query_db` (SELECT-only) |
| Retrieval feeding the crew | `src/ai_roadmap/rag_pipeline.py` | `notebooks/05_rag_minimal.ipynb` |

---

## 📌 Phase 5E: Agentic AI Specialization

### 1. Agent Foundations (ReAct & Tool Use)
Understand the basic architecture of an agent: an LLM loop that processes observations and selects tools.
- **Topics**: ReAct loop (Reason, Act, Observe), Function Calling, System Instructions, Tool schema definitions.
- 📺 **Intro to AI Agents & Tool Use**: [Watch Video](https://www.youtube.com/watch?v=kBXYFaZ0EN0)

### 2. Model Context Protocol (MCP)
The newly introduced Model Context Protocol (MCP) by Anthropic establishes an open standard for connecting AI models to data sources (databases, filesystems) and tools safely.
- **Topics**: MCP Host, MCP Client, MCP Server, secure local data access.
- 📺 **Model Context Protocol (MCP) Complete Tutorial**: [Watch Video](https://www.youtube.com/watch?v=3_TN1i3MTEU)

### 3. Agent Frameworks: CrewAI, AutoGen & LangGraph
Move beyond single agents to orchestrate team workflows and complex state transitions.
- **CrewAI**: Role-playing agent groups, task queues, and sequential delegation.
- **AutoGen**: Conversational multi-agent setups.
- **LangGraph**: Cycle-based state machine agents (industry standard for complex, production-grade applications).
- 📺 **CrewAI Crash Course for Beginners**: [Watch Video](https://www.youtube.com/watch?v=kBXYFaZ0EN0)
- 📺 **LangGraph Course (Building State-driven Agents)**: [Watch Video](https://www.youtube.com/watch?v=1w5cCXlh7JQ)

### 4. Agentic Memory & Planning
Giving agents memory across conversation boundaries.
- **Topics**: Episodic Memory, Semantic Memory, Long-term databases, Reflection & Self-Correction techniques.
- 📺 **AI Agent Memory Systems Explained**: [Watch Video](https://www.youtube.com/watch?v=kBXYFaZ0EN0)

---

## 💻 Code Sample: Custom Agent with CrewAI
Here is how to create a research agent group using CrewAI:

```python
from crewai import Agent, Task, Crew, Process

# Define a research agent
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI Agents',
    backstory="You are an expert AI researcher specializing in agentic systems.",
    verbose=True,
    allow_delegation=False
)

# Define the research task
research_task = Task(
    description='Analyze the latest papers on Model Context Protocol (MCP) in 2026.',
    expected_output='A comprehensive 3-paragraph summary of MCP benefits.',
    agent=researcher
)

# Build the crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential
)

result = crew.kickoff()
print(result)
```

---

## 🛠️ AI Agent Milestones & Projects
1. **Web Research Agent (LangChain + DuckDuckGo Search Tool)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=kBXYFaZ0EN0)
2. **Autonomous Multi-Agent Content Team (CrewAI + Local Llama 3)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=kBXYFaZ0EN0)
3. **Enterprise Flowchart-driven Agent (LangGraph + FastAPI)**
   - 📺 **Project Tutorial**: [Watch Video](https://www.youtube.com/watch?v=1w5cCXlh7JQ)
