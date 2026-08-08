import json
import sys


for raw in sys.stdin:
    message = json.loads(raw)
    method = message.get("method")
    if "id" not in message:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
            "serverInfo": {"name": "echo-test", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "echo", "description": "Echo a message",
            "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        }]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": message["params"]["arguments"]["message"]}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
