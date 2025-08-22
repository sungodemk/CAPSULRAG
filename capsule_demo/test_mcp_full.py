#!/usr/bin/env python3
"""
Complete MCP server test suite
"""

import asyncio
import json
import subprocess
import sys


async def test_all_mcp_tools():
    """Test all CapsuleRAG MCP tools comprehensively."""
    
    # Start MCP server
    server_process = await asyncio.create_subprocess_exec(
        sys.executable, "mcp_server_simple.py",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    try:
        print(" Starting CapsuleRAG MCP Server Tests\n")
        
        # Test 1: Initialize
        print("1⃣ Testing initialize...")
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
        
        request_data = json.dumps(init_request).encode() + b'\n'
        server_process.stdin.write(request_data)
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        init_response = json.loads(response_line.decode())
        print(f" Server initialized: {init_response['result']['serverInfo']['name']}")
        
        # Test 2: List tools
        print("\n2⃣ Testing tools/list...")
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
        tool_names = [tool['name'] for tool in tools_response['result']['tools']]
        print(f" Available tools: {tool_names}")
        
        # Test 3: Ingest documents
        print("\n3⃣ Testing capsules.ingest...")
        test_docs = [
            {
                "content": """# CapsuleRAG Architecture Document

## Overview
CapsuleRAG implements a hybrid retrieval system combining:
- BM25 lexical search for keyword matching
- Semantic search using SentenceTransformers 
- Cross-encoder reranking for relevance optimization

## Core Components
### Document Ingestion
- Structure-aware chunking with byte ranges
- Content-addressed manifests with cryptographic signatures
- Per-capsule vector indices using HNSW

### Governance Features
- ACL-based permission system
- Signed manifests for data integrity
- Audit trails for all access""",
                "filename": "architecture.md"
            },
            {
                "content": """# Model Context Protocol Integration

## MCP Tools
CapsuleRAG exposes the following MCP tools:

### capsules.ingest
- Input: content, filename, user_permissions
- Output: doc_id, content_hash, manifest_signature

### capsules.search  
- Input: query, top_k, rerank, user_permissions
- Output: ranked results with byte-range citations

### capsules.hydrate
- Input: doc_id, user_permissions
- Output: full document text (ACL-aware)

## Benefits
- Vendor-neutral AI agent integration
- Standardized JSON-RPC protocol
- Discovery and schema validation""",
                "filename": "mcp_integration.md"
            }
        ]
        
        doc_ids = []
        for i, doc in enumerate(test_docs):
            ingest_request = {
                "jsonrpc": "2.0",
                "id": 10 + i,
                "method": "tools/call",
                "params": {
                    "name": "capsules.ingest",
                    "arguments": doc
                }
            }
            
            request_data = json.dumps(ingest_request).encode() + b'\n'
            server_process.stdin.write(request_data)
            await server_process.stdin.drain()
            
            response_line = await server_process.stdout.readline()
            ingest_response = json.loads(response_line.decode())
            result_data = json.loads(ingest_response['result']['content'][0]['text'])
            doc_ids.append(result_data['doc_id'])
            print(f" Ingested '{doc['filename']}' as doc_id={result_data['doc_id']}, chunks={result_data['num_chunks']}")
        
        # Test 4: Search with different queries
        print("\n4⃣ Testing capsules.search...")
        search_queries = [
            "hybrid retrieval system",
            "MCP tools and benefits", 
            "governance features",
            "JSON-RPC protocol"
        ]
        
        for i, query in enumerate(search_queries):
            search_request = {
                "jsonrpc": "2.0",
                "id": 20 + i,
                "method": "tools/call",
                "params": {
                    "name": "capsules.search",
                    "arguments": {
                        "query": query,
                        "top_k": 3,
                        "rerank": True,
                        "user_permissions": ["public:read"]
                    }
                }
            }
            
            request_data = json.dumps(search_request).encode() + b'\n'
            server_process.stdin.write(request_data)
            await server_process.stdin.drain()
            
            response_line = await server_process.stdout.readline()
            search_response = json.loads(response_line.decode())
            result_data = json.loads(search_response['result']['content'][0]['text'])
            
            print(f" Query: '{query}'")
            for result in result_data['results'][:2]:  # Show top 2
                print(f"    Doc {result['doc_id']}: {result['snippet'][:80]}... (score: {result['score']:.3f})")
        
        # Test 5: Hydrate documents
        print("\n5⃣ Testing capsules.hydrate...")
        for doc_id in doc_ids[:1]:  # Test first document
            hydrate_request = {
                "jsonrpc": "2.0",
                "id": 30 + doc_id,
                "method": "tools/call",
                "params": {
                    "name": "capsules.hydrate",
                    "arguments": {
                        "doc_id": doc_id,
                        "user_permissions": ["public:read"]
                    }
                }
            }
            
            request_data = json.dumps(hydrate_request).encode() + b'\n'
            server_process.stdin.write(request_data)
            await server_process.stdin.drain()
            
            response_line = await server_process.stdout.readline()
            hydrate_response = json.loads(response_line.decode())
            result_data = json.loads(hydrate_response['result']['content'][0]['text'])
            
            text_preview = result_data['text'][:150] + "..." if len(result_data['text']) > 150 else result_data['text']
            print(f" Hydrated doc {doc_id}: {text_preview}")
        
        # Test 6: List resources
        print("\n6⃣ Testing resources/list...")
        resources_request = {
            "jsonrpc": "2.0",
            "id": 40,
            "method": "resources/list",
            "params": {}
        }
        
        request_data = json.dumps(resources_request).encode() + b'\n'
        server_process.stdin.write(request_data)
        await server_process.stdin.drain()
        
        response_line = await server_process.stdout.readline()
        resources_response = json.loads(response_line.decode())
        resources = resources_response['result']['resources']
        print(f" Available resources: {len(resources)}")
        for resource in resources:
            print(f"    {resource['name']}: {resource['uri']}")
        
        print("\n All MCP tests completed successfully!")
        print("\n Summary:")
        print(f"    Initialized MCP server")
        print(f"    Discovered {len(tool_names)} tools")
        print(f"    Ingested {len(doc_ids)} documents")
        print(f"    Performed {len(search_queries)} searches")
        print(f"    Hydrated documents with ACL checking")
        print(f"    Listed {len(resources)} governance resources")
        
    except Exception as e:
        print(f" MCP test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Print stderr for debugging
        try:
            stderr_data = await asyncio.wait_for(server_process.stderr.read(), timeout=1.0)
            if stderr_data:
                print(f"Server stderr: {stderr_data.decode()}")
        except:
            pass
    
    finally:
        try:
            server_process.terminate()
            await asyncio.wait_for(server_process.wait(), timeout=2.0)
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_all_mcp_tools())
