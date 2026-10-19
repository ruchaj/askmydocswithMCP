import ast
import operator

from mcp.server.fastmcp import FastMCP
import askmydocs_store as store

mcp = FastMCP("askmydocs")

@mcp.tool()
def search_documents(query: str, doc_set_id: str, k: int = 3) -> str:
  """Search the uploaded PDF text for passages relevant to a query using
    semantic vector search. Call it multiple times with different queries to
    gather more context.
 
    Args:
        query: What to look for, phrased as natural language.
        doc_set_id: Identifier for the uploaded document set to search.
        k: How many passages to return. Defaults to 3.
    """
  return store.format_search_results(query=query, doc_set_id=doc_set_id, k=k)\

@mcp.tool()
def list_documents(doc_set_id: str) -> str: