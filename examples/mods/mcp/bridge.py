#!/usr/bin/env python3
"""MCP connector mod for haste. Stdlib only.

Speaks MCP (newline-delimited JSON-RPC over stdio) to servers configured in
the MCP_SERVERS env var (JSON: {"name": "command line" | ["argv", ...]}).

  bridge.py list <server>
  bridge.py call <server> <tool> key=value key=value ...
  bridge.py call <server> <tool> {"json": "args"}

Spawns the server per call (simple, stateless). A server that is slow to boot
pays that cost each call - acceptable for v1; a daemonizing bridge is a mod-
internal upgrade that needs no harness change.
"""
import json
import os
import subprocess
import sys
import time


def send(proc, obj):
    proc.stdin.write((json.dumps(obj) + "\n").encode())
    proc.stdin.flush()


def read_until(proc, rpc_id, timeout):
    end = time.time() + timeout
    while time.time() < end:
        line = proc.stdout.readline()
        if not line:
            return None
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("id") == rpc_id:
            return m
    return None


def parse_args(parts):
    raw = " ".join(parts).strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        return json.loads(raw)
    args = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"expected key=value, got '{p}'")
        k, v = p.split("=", 1)
        if v in ("true", "false"):
            v = v == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        args[k] = v
    return args


def main():
    servers = json.loads(os.environ.get("MCP_SERVERS", "{}"))
    a = sys.argv[1:]
    if len(a) < 2 or a[0] not in ("list", "call"):
        print("usage: M list <server> | M call <server> <tool> key=value ...")
        print(f"configured servers: {', '.join(servers) or '(none - set MCP_SERVERS in mod.toml)'}")
        return 1
    op, name = a[0], a[1]
    cmd = servers.get(name)
    if cmd is None:
        print(f"unknown server '{name}'; configured: {', '.join(servers) or '(none)'}")
        return 1
    proc = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "haste-mcp-mod", "version": "0.1"}}})
        if read_until(proc, 1, 60) is None:
            print(f"server '{name}' did not answer initialize (60s)")
            return 1
        send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        if op == "list":
            send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            m = read_until(proc, 2, 60)
            tools = (m or {}).get("result", {}).get("tools", [])
            if not tools:
                print("(no tools reported)")
            for t in tools:
                desc = (t.get("description") or "").split("\n")[0][:110]
                props = list(t.get("inputSchema", {}).get("properties", {}))
                print(f"{t['name']}({', '.join(props)}) - {desc}")
        else:
            if len(a) < 3:
                print("usage: M call <server> <tool> key=value ...")
                return 1
            try:
                args = parse_args(a[3:])
            except ValueError as e:
                print(f"bad args: {e}")
                return 1
            send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": a[2], "arguments": args}})
            m = read_until(proc, 2, 120)
            if m is None:
                print("call timed out (120s)")
                return 1
            if "error" in m:
                print(f"error: {m['error'].get('message', m['error'])}")
                return 1
            for c in m.get("result", {}).get("content", []):
                print(c.get("text") if c.get("type") == "text" else json.dumps(c)[:400])
        return 0
    finally:
        proc.kill()


if __name__ == "__main__":
    sys.exit(main())
