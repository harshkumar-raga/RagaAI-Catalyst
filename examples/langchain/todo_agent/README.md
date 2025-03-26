# LangChain Todo Agent

A smart todo list management system built with LangChain and RagaAI Catalyst. This application combines the power of LLM-based agents with task management functionality, allowing for natural language interaction with your todo list while maintaining traceability through RagaAI Catalyst.

## Features

- Interactive CLI interface for todo list management
- LangChain-powered agent for natural language processing
- Integration with RagaAI Catalyst for tracing and monitoring
- CRUD operations for todo items (Create, Read, Update, Delete)
- Persistent storage using JSON
- Task attributes include ID, title, description, status, and creation timestamp

## Prerequisites

- Python 3.x
- OpenAI API key
- RagaAI Catalyst credentials (access key and secret key)

## Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
CATALYST_ACCESS_KEY=your_catalyst_access_key
CATALYST_SECRET_KEY=your_catalyst_secret_key
CATALYST_BASE_URL=your_catalyst_base_url
PROJECT_NAME=your_project_name
DATASET_NAME=your_dataset_name
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install langchain openai python-dotenv ragaai-catalyst
```

## Usage

Run the application:
```bash
python todo.py
```

The application provides a menu-driven interface with the following options:
1. Add Task
2. Delete Task
3. Modify Task
4. List Tasks
5. Exit

Each operation is processed through a LangChain agent that understands natural language commands and is traced using RagaAI Catalyst for monitoring and analysis.

## Features in Detail

- **Add Task**: Create new tasks with a title and description
- **Delete Task**: Remove tasks by their ID
- **Modify Task**: Update task title, description, or status
- **List Tasks**: View all tasks in JSON format

## Tracing and Monitoring

The application uses RagaAI Catalyst's tracing capabilities to monitor:
- Agent operations
- Tool usage
- Task management functions

This enables detailed analysis of the application's performance and behavior through the RagaAI Catalyst dashboard.
