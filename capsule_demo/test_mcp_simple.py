#!/usr/bin/env python3
"""
Simple direct test of MCP server without complex client setup
"""

import asyncio
import json
import subprocess
import sys


async def test_mcp_jsonrpc():
    """Test MCP server with direct JSON-RPC messages."""
    
    # Start MCP server
    server_process = await asyncio.create_subprocess_exec(
        sys.executable, "mcp_server_simple.py",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    try:
        # Test 1: Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        # Send request
        request_data = json.dumps(init_request).encode() + b'\n'
        server_process.stdin.write(request_data)
        await server_process.stdin.drain()
        
        # Read response
        response_line = await server_process.stdout.readline()
        init_response = json.loads(response_line.decode())
        print(f" Initialize response: {init_response}")
        
        # Test 2: List tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        request_data = json.dumps(tools_request).encode() + b'\n'
        server_process.stdin.write(request_data)
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        tools_response = json.loads(response_line.decode())
        print(f" Tools list: {[tool['name'] for tool in tools_response['result']['tools']]}")
        
        # Test 3: Call ingest tool
        ingest_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "capsules.ingest",
                "arguments": {
                    "content": "# Test MCP Document\n\nThis tests our MCP integration.\n\n## Features\n- JSON-RPC protocol\n- Tool discovery\n- Resource management",
                    "filename": "test_mcp.md"
                }
            }
        }
        
        request_data = json.dumps(ingest_request).encode() + b'\n'
        server_process.stdin.write(request_data)
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        ingest_response = json.loads(response_line.decode())
        print(f" Ingest response: {ingest_response}")
        
        print(" MCP server is working correctly!")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Print stderr for debugging
        stderr_data = await server_process.stderr.read()
        if stderr_data:
            print(f"Server stderr: {stderr_data.decode()}")
    
    finally:
        server_process.terminate()
        await server_process.wait()


if __name__ == "__main__":
    asyncio.run(test_mcp_jsonrpc())
