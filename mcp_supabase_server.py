import sys
import json

def log_debug(msg):
    print(f"DEBUG: {msg}", file=sys.stderr)

def main():
    log_debug("Server started")
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            log_debug(f"Received method: {method}")

            # Базовая структура ответа
            result = {}

            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {"name": "auto-cro-server", "version": "1.0.0"}
                }
            elif method in ["tools/list", "listTools"]:
                result = {
                    "tools": [
                        {
                            "name": "check_supabase",
                            "description": "Проверить таблицы в БД",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    ]
                }
            elif method in ["tools/call", "callTool"]:
                params = request.get("params", {})
                if params.get("name") == "check_supabase":
                    import os
                    from supabase import create_client
                    url = os.environ.get("SUPABASE_URL", "")
                    key = os.environ.get("SUPABASE_KEY", "")
                    if url and key:
                        try:
                            client = create_client(url, key)
                            res = client.table("bandit_states").select("*", count="exact").limit(1).execute()
                            count = res.count
                            result = {
                                "content": [{"type": "text", "text": f"COUNT: {count}"}]
                            }
                        except Exception as e:
                            result = {
                                "content": [{"type": "text", "text": f"Error: {str(e)}"}]
                            }
                    else:
                        result = {
                            "content": [{"type": "text", "text": "Error: SUPABASE_URL or SUPABASE_KEY not set"}]
                        }
                else:
                    result = {
                        "content": [{"type": "text", "text": f"Unknown tool: {params.get('name')}"}]
                    }
            elif method == "notifications/initialized":
                continue # На уведомления отвечать не нужно
            
            # Отправляем ответ строго в формате JSON-RPC
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            log_debug(f"Error: {str(e)}")

if __name__ == "__main__":
    main()