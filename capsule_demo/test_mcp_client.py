#!/usr/bin/env python3
"""
Simple MCP client to test CapsuleRAG MCP server
"""

import asyncio
import json
import subprocess
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test the CapsuleRAG MCP server functionality."""
    
    # Start the MCP server as a subprocess
    server_process = await asyncio.create_subprocess_exec(
        sys.executable, "mcp_server.py",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    try:
        # Create MCP client session using streams directly
        async with ClientSession(server_process.stdout, server_process.stdin) as session:
                # Initialize the session
                await session.initialize()
                
                print(" MCP Session initialized successfully")
                
                # Test 1: List available tools
                print("\n Testing tools/list...")
                tools = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools]}")
                
                # Test 2: Test capsules.ingest
                print("\n Testing capsules.ingest...")
                test_content = """# Test Document for MCP

This is a test document for the Model Context Protocol implementation.

## Key Features
- MCP tool exposure
- JSON-RPC communication
- ACL governance
- Hybrid retrieval"""
                
                ingest_result = await session.call_tool(
                    "capsules.ingest",
                    {
                        "content": test_content,
                        "filename": "test_mcp.md",
                        "user_permissions": ["public:read"]
                    }
                )
                print(f"Ingest result: {ingest_result.content[0].text}")
                
                # Parse the doc_id from result
                ingest_data = json.loads(ingest_result.content[0].text)
                doc_id = ingest_data["doc_id"]
                
                # Test 3: Test capsules.search
                print("\n Testing capsules.search...")
                search_result = await session.call_tool(
                    "capsules.search",
                    {
                        "query": "MCP implementation",
                        "top_k": 2,
                        "rerank": True,
                        "user_permissions": ["public:read"]
                    }
                )
                print(f"Search result: {search_result.content[0].text}")
                
                # Test 4: Test capsules.manifest
                print("\n Testing capsules.manifest...")
                manifest_result = await session.call_tool(
                    "capsules.manifest",
                    {"doc_id": doc_id}
                )
                print(f"Manifest result: {manifest_result.content[0].text}")
                
                # Test 5: Test capsules.hydrate
                print("\n Testing capsules.hydrate...")
                hydrate_result = await session.call_tool(
                    "capsules.hydrate",
                    {
                        "doc_id": doc_id,
                        "user_permissions": ["public:read"]
                    }
                )
                print(f"Hydrate result: {hydrate_result.content[0].text}")
                
                # Test 6: List resources
                print("\n Testing resources/list...")
                resources = await session.list_resources()
                print(f"Available resources: {[r.name for r in resources]}")
                
                # Test 7: Read a resource
                if resources:
                    print("\n Testing resources/read...")
                    resource_content = await session.read_resource(resources[0].uri)
                    print(f"Resource content: {resource_content[0].text}")
                
                print("\n All MCP tests completed successfully!")
                
    except Exception as e:
        print(f" MCP test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        server_process.terminate()
        await server_process.wait()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
