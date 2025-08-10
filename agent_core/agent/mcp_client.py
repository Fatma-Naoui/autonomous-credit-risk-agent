import asyncio
import aiohttp
import uuid
import json

class MCPClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8000/mcp/tool", session_id: str = "mcp-client-test-session"):
        self.server_url = server_url
        self.session_id = session_id

    async def run_tool(self, tool_name: str, args: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args,
                "sessionId": self.session_id
            }
        }

        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.server_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status} - {await resp.text()}")

                async for line in resp.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data: "):
                        data = decoded[6:]
                        try:
                            return json.loads(data)
                        except json.JSONDecodeError:
                            return {"raw": data}

# Expose a top-level run_tool function so you can import it directly
_mcp_client = MCPClient()

async def run_tool(tool_name: str, args: dict) -> dict:
    return await _mcp_client.run_tool(tool_name, args)
