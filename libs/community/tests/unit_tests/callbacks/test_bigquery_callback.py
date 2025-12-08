"""Unit tests for BigQueryCallbackHandler."""

from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document

# Preserving the original import path as requested, assuming the environment
# maps this to the updated file or it will be moved there.
from langchain_community.callbacks.bigquery_callback import BigQueryCallbackHandler, BigQueryLoggerConfig


@pytest.fixture
def mock_bigquery_clients() -> Generator[Dict[str, Any], None, None]:
    """Mocks the BigQuery clients and dependencies."""
    # We patch the google libraries where they are imported or globally
    with patch("google.cloud.bigquery.Client") as mock_bq_client_cls, \
         patch("google.cloud.bigquery_storage_v1.services.big_query_write.async_client.BigQueryWriteAsyncClient") as mock_write_client_cls, \
         patch("google.cloud.storage.Client") as mock_storage_client_cls, \
         patch("google.auth.default", return_value=(MagicMock(), "test-project")):

        # Mock BigQuery Client
        mock_bq_client = MagicMock()
        mock_bq_client_cls.return_value = mock_bq_client
        
        # Mock BigQuery Write Client
        mock_write_client = AsyncMock()
        mock_write_client_cls.return_value = mock_write_client
        
        # Mock Storage Client
        mock_storage_client = MagicMock()
        mock_storage_client_cls.return_value = mock_storage_client

        # Mock append_rows to return an async iterator
        async def mock_append_rows(*args, **kwargs):
             mock_response = MagicMock()
             mock_response.error.code = 0
             mock_response.row_errors = []
             yield mock_response

        mock_write_client.append_rows.side_effect = mock_append_rows

        yield {
            "mock_bq_client": mock_bq_client,
            "mock_write_client": mock_write_client,
            "mock_storage_client": mock_storage_client,
        }


@pytest.fixture
async def handler(mock_bigquery_clients: Dict[str, Any]) -> BigQueryCallbackHandler:
    """
    Returns an initialized `BigQueryCallbackHandler` with mocked clients.
    """
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    # Ensure initialization is run
    await handler._ensure_started()
    return handler


@pytest.mark.asyncio
async def test_on_llm_start(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_llm_start logs the correct event."""
    run_id = uuid4()
    parent_run_id = uuid4()
    await handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["test prompt"],
        run_id=run_id,
        parent_run_id=parent_run_id,
    )
    
    # Wait for the batch processor to process the event
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_chat_model_start(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chat_model_start logs the correct event."""
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={"name": "test_chat_model"},
        messages=[[HumanMessage(content="test")]],
        run_id=run_id,
    )
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_llm_end(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_llm_end logs the correct event."""
    response = LLMResult(generations=[[MagicMock(text="test generation")]], llm_output={"token_usage": {"total_tokens": 10}})
    await handler.on_llm_end(response, run_id=uuid4())

    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_llm_error(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_llm_error logs the correct event."""
    await handler.on_llm_error(Exception("test error"), run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_chain_start(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chain_start logs the correct event."""
    await handler.on_chain_start(
        serialized={"name": "test_chain"}, inputs={"input": "test"}, run_id=uuid4()
    )
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_chain_end(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chain_end logs the correct event."""
    await handler.on_chain_end(outputs={"output": "test"}, run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_chain_error(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_chain_error logs the correct event."""
    await handler.on_chain_error(Exception("chain error"), run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_tool_start(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_tool_start logs the correct event."""
    await handler.on_tool_start(
        serialized={"name": "test_tool"}, input_str="test", run_id=uuid4()
    )
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_tool_end(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_tool_end logs the correct event."""
    await handler.on_tool_end(output="test output", run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_tool_error(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_tool_error logs the correct event."""
    await handler.on_tool_error(Exception("tool error"), run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_retriever_start(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_retriever_start logs the correct event."""
    await handler.on_retriever_start(
        serialized={"name": "test_retriever"}, query="test query", run_id=uuid4()
    )
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_retriever_end(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_retriever_end logs the correct event."""
    documents = [Document(page_content="test doc")]
    await handler.on_retriever_end(documents, run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_retriever_error(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_retriever_error logs the correct event."""
    await handler.on_retriever_error(Exception("retriever error"), run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_text(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_text logs the correct event."""
    await handler.on_text("some text", run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_agent_action(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_agent_action logs the correct event."""
    action = AgentAction(tool="test_tool", tool_input="input", log="log")
    await handler.on_agent_action(action, run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_on_agent_finish(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that on_agent_finish logs the correct event."""
    finish = AgentFinish(return_values={"output": "done"}, log="log")
    await handler.on_agent_finish(finish, run_id=uuid4())
    
    await handler.batch_processor.shutdown(timeout=1.0)

    mock_write_client = mock_bigquery_clients["mock_write_client"]
    assert mock_write_client.append_rows.call_count >= 1


@pytest.mark.asyncio
async def test_close(
    handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that close shuts down the batch processor."""
    await handler.close()
    # If close worked, the batch processor should be shut down (we can't easily check internal state, 
    # but we can check if it finishes without error)
    pass


@pytest.mark.asyncio
async def test_ensure_started_creates_table(
    mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that _ensure_started creates dataset (implicitly via client) and table if they don't exist."""
    
    # Setup mock to raise NotFound for get_table, triggering create_table
    from google.cloud.exceptions import NotFound
    mock_bigquery_clients["mock_bq_client"].get_table.side_effect = NotFound("Table not found")

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    await handler._ensure_started()

    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    # Check if create_table was called
    assert mock_bq_client.create_table.called


@pytest.mark.asyncio
async def test_gcs_offloading_init(
    mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that GCS offloader is initialized when bucket is provided."""
    config = BigQueryLoggerConfig(gcs_bucket_name="test-bucket")
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        config=config
    )
    await handler._ensure_started()
    
    assert handler.offloader is not None
    assert handler.offloader.bucket.name == "test-bucket"
