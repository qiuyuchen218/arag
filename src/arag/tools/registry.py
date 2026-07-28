"""Tool registry for ARAG."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool
from arag.core.schemas import ToolResult, stable_hash, utc_now

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> BaseTool:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def execute(self, name: str, context: 'AgentContext', **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found", {"error": "tool_not_found"}
        
        try:
            result = tool.execute(context, **kwargs)
            if isinstance(result, ToolResult):
                return result.to_legacy_tuple()
            if isinstance(result, tuple) and len(result) == 2:
                rendered, log = result
                log = dict(log or {})
                log.setdefault("tool_result", ToolResult(
                    call_id=f"call_{stable_hash(name, kwargs, utc_now())}",
                    tool_name=name,
                    status="failed" if log.get("error") else "success",
                    rendered_text=str(rendered),
                    diagnostics={k: v for k, v in log.items() if k not in {"error", "retrieved_tokens"}},
                    retrieved_tokens=int(log.get("retrieved_tokens", 0) or 0),
                    error=log.get("error"),
                ).to_dict())
                return rendered, log
            return str(result), {"tool_result": result}
        except Exception as e:
            return f"Error executing tool: {str(e)}", {"error": str(e)}
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
