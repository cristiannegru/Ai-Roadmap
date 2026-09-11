# AI Agents

Autonomous agents loop over thought action and observation. The ReAct
pattern interleaves reasoning traces with tool calls like web search,
calculators, and database queries until the task completes.

Multi agent frameworks divide labor: CrewAI assigns researcher writer and
critic roles, LangGraph models stateful workflows with cycles, and AutoGen
runs coder reviewer tester conversations. The Model Context Protocol MCP
exposes tools such as SQLite queries over a standard client server
interface so any assistant can inspect local data safely.

Evaluate agents by task success rate, tool call efficiency, and recovery
from failed actions.
