import asyncio
import aiohttp
import uuid

async def run_tool(tool_name: str, args: dict):
    url = "http://127.0.0.1:8000/mcp/tool"

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",  # ✅ FIXED
        "params": {
            "name": tool_name,     # ✅ FIXED
            "arguments": args,     # ✅ FIXED
            "sessionId": "mcp-client-test-session"
        }
    }

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                print(f"❌ HTTP {resp.status} - {await resp.text()}")
                return

            async for line in resp.content:
                decoded = line.decode().strip()
                if decoded.startswith("data: "):
                    print(decoded[6:])

if __name__ == "__main__":
    asyncio.run(run_tool("get_autogluon_tabular_workflow", {}))
