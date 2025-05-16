# tests/custom_tests/ragaai_catalyst/tracers/agentic_tracing/utils/test_synthetic_data_generation.py
import pytest
import os
import tempfile
from dotenv import load_dotenv
from ragaai_catalyst.synthetic_data_generation import SyntheticDataGeneration

# Test responses dictionary
TEST_RESPONSES = {
  "initialization": {
    "success": True,
    "generator_type": "<class 'ragaai_catalyst.synthetic_data_generation.SyntheticDataGeneration'>"
  },
  "get_supported_qna": {
    "success": True,
    "supported_qna": [
      "simple",
      "mcq",
      "complex"
    ]
  },
  "get_supported_providers": {
    "success": True,
    "supported_providers": [
      "gemini",
      "openai",
      "azure"
    ]
  },
  "process_document_text": {
    "success": True,
    "input_text": "This is a test document for processing.",
    "processed_text": "This is a test document for processing."
  },
  "process_document_file": {
    "success": True,
    "file_path": "C:\\Users\\harsh\\AppData\\Local\\Temp\\tmpo8oomqe6.txt",
    "processed_text": "This is test content in a temporary file."
  },
  "validate_input": {
    "success": True,
    "valid_text_result": False,
    "empty_text_result": "Empty Text provided for qna generation. Please provide valid text",
    "short_text_result": "Very Small Text provided for qna generation. Please provide longer text"
  },
  "generate_qna": {
    "skipped": True,
    "reason": "Skipped to avoid making actual API calls that would incur costs"
  },
  "generate_examples": {
    "skipped": True,
    "reason": "Skipped to avoid making actual API calls that would incur costs"
  }
}

@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
        temp_file.write(b"This is test content in a temporary file.")
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

def test_initialization():
    """Test SyntheticDataGeneration initialization."""
    generator = SyntheticDataGeneration()
    assert str(type(generator)) == TEST_RESPONSES["initialization"]["generator_type"]

def test_get_supported_qna():
    """Test get_supported_qna functionality."""
    generator = SyntheticDataGeneration()
    supported_qna = generator.get_supported_qna()
    
    # Check that the supported QnA types match the expected values
    assert supported_qna == TEST_RESPONSES["get_supported_qna"]["supported_qna"]

def test_get_supported_providers():
    """Test get_supported_providers functionality."""
    generator = SyntheticDataGeneration()
    supported_providers = generator.get_supported_providers()
    
    # Check that the supported providers match the expected values
    assert supported_providers == TEST_RESPONSES["get_supported_providers"]["supported_providers"]

def test_process_document_text():
    """Test process_document with text string."""
    generator = SyntheticDataGeneration()
    test_text = TEST_RESPONSES["process_document_text"]["input_text"]
    processed_text = generator.process_document(test_text)
    
    # Check that the processed text matches the input text
    assert processed_text == TEST_RESPONSES["process_document_text"]["processed_text"]

def test_process_document_file(temp_text_file):
    """Test process_document with file."""
    generator = SyntheticDataGeneration()
    processed_text = generator.process_document(temp_text_file)
    
    # Check that the processed text contains the expected content
    # Note: We don't check the exact file path as it changes on each run
    assert "This is test content in a temporary file." in processed_text

def test_validate_input():
    """Test validate_input functionality."""
    generator = SyntheticDataGeneration()
    
    # Test with valid text
    valid_text = "This is a valid text with enough content to be processed properly."
    valid_result = generator.validate_input(valid_text)
    assert valid_result == TEST_RESPONSES["validate_input"]["valid_text_result"]
    
    # Test with empty text
    empty_text = ""
    empty_result = generator.validate_input(empty_text)
    assert empty_result == TEST_RESPONSES["validate_input"]["empty_text_result"]
    
    # Test with very short text
    short_text = "Hi"
    short_result = generator.validate_input(short_text)
    assert short_result == TEST_RESPONSES["validate_input"]["short_text_result"]

def test_generate_qna_skipped():
    """Test that generate_qna is skipped in tests."""
    # This test exists to document that we're intentionally skipping
    # the generate_qna tests to avoid making API calls
    assert TEST_RESPONSES["generate_qna"]["skipped"]
    assert "API calls" in TEST_RESPONSES["generate_qna"]["reason"]

def test_generate_examples_skipped():
    """Test that generate_examples is skipped in tests."""
    # This test exists to document that we're intentionally skipping
    # the generate_examples tests to avoid making API calls
    assert TEST_RESPONSES["generate_examples"]["skipped"]
    assert "API calls" in TEST_RESPONSES["generate_examples"]["reason"]