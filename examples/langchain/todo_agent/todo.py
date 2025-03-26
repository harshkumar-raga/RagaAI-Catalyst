from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from datetime import datetime
from ragaai_catalyst import RagaAICatalyst, Tracer, init_tracing
import json
import os

from dotenv import load_dotenv
load_dotenv()

catalyst = RagaAICatalyst(
    access_key=os.getenv('CATALYST_ACCESS_KEY'), 
    secret_key=os.getenv('CATALYST_SECRET_KEY'), 
    base_url=os.getenv('CATALYST_BASE_URL')
)
tracer = Tracer(
    project_name=os.environ['PROJECT_NAME'],
    dataset_name=os.environ['DATASET_NAME'],
    tracer_type="agentic/langchain",
)

init_tracing(catalyst=catalyst, tracer=tracer)

class TodoManager:
    def __init__(self, filename="todos.json"):
        self.filename = filename
        self.todos = self._load_todos()

    def _load_todos(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        return []

    def _save_todos(self):
        with open(self.filename, 'w') as f:
            json.dump(self.todos, f, indent=2)

    def _generate_unique_id(self):
        if not self.todos:
            return 1
        return max(task['id'] for task in self.todos) + 1

    @tracer.trace_agent(name="add_task")
    def add_task(self, title, description):
        task = {
            'id': self._generate_unique_id(),
            'title': title,
            'description': description,
            'status': 'pending',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.todos.append(task)
        self._save_todos()
        print("\nCurrent todos after adding task:")
        print(json.dumps(self.todos, indent=2))
        return f"Task added successfully with ID: {task['id']}"

    @tracer.trace_agent(name="delete_task")
    def delete_task(self, task_id):
        for i, task in enumerate(self.todos):
            if task['id'] == task_id:
                del self.todos[i]
                self._save_todos()
                print("\nCurrent todos after deleting task:")
                print(json.dumps(self.todos, indent=2))
                return f"Task {task_id} deleted successfully"
        return f"Task {task_id} not found"

    @tracer.trace_agent(name="modify_task")
    def modify_task(self, task_id, title=None, description=None, status=None):
        for task in self.todos:
            if task['id'] == int(task_id):
                if title:
                    task['title'] = title
                if description:
                    task['description'] = description
                if status:
                    task['status'] = status
                self._save_todos()
                print("\nCurrent todos after modifying task:")
                print(json.dumps(self.todos, indent=2))
                return f"Task {task_id} modified successfully"
        return f"Task {task_id} not found"

    @tracer.trace_agent(name="list_tasks")
    def list_tasks(self):
        if not self.todos:
            return "No tasks found"
        return json.dumps(self.todos, indent=2)

@tracer.trace_tool(name="create_agent")
def create_todo_agent():
    todo_manager = TodoManager()
    
    tools = [
        Tool(
            name="Add Task",
            func=lambda x: todo_manager.add_task(*json.loads(x)),
            description="Add a new task. Input format: JSON array with [title, description]"
        ),
        Tool(
            name="Delete Task",
            func=lambda x: todo_manager.delete_task(int(x)),
            description="Delete a task by ID"
        ),
        Tool(
            name="Modify Task",
            func=lambda x: todo_manager.modify_task(*json.loads(x)),
            description="Modify a task. Input format: JSON array with [task_id, title, description, status]"
        ),
        Tool(
            name="List Tasks",
            func=lambda x: todo_manager.list_tasks(),
            description="List all tasks"
        )
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        model_name='gpt-4o-mini'
    )
    
    agent = initialize_agent(
        tools,
        llm,
        agent="conversational-react-description",
        memory=memory,
        verbose=True
    )
    
    return agent

def main():
    agent = create_todo_agent()
    
    while True:
        print("\nOptions:")
        print("1. Add Task")
        print("2. Delete Task")
        print("3. Modify Task")
        print("4. List Tasks")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            title = input("Enter task title: ")
            description = input("Enter task description: ")
            
            if title and description:
                task_data = json.dumps([title, description])
                try:
                    response = agent.invoke(f"Add a new task with this data: {task_data}")
                    print(response['output'])
                except Exception as e:
                    print(f"Error: {str(e)}")
            else:
                print("Title and description are required.")
            
        elif choice == "2":
            task_id = input("Enter task ID to delete: ")
            try:
                response = agent.invoke(f"Delete task with ID {task_id}")
                print(response['output'])
            except Exception as e:
                print(f"Error: {str(e)}")
            
        elif choice == "3":
            task_id = input("Enter task ID to modify: ")
            title = input("Enter new title (or press Enter to skip): ")
            description = input("Enter new description (or press Enter to skip): ")
            status = input("Enter new status (or press Enter to skip): ")
            
            mod_args = [int(task_id), title or None, description or None, status or None]
            task_data = json.dumps(mod_args)
            try:
                response = agent.invoke(f"Modify task with these parameters: {task_data}")
                print(response['output'])
            except Exception as e:
                print(f"Error: {str(e)}")
            
        elif choice == "4":
            try:
                response = agent.invoke("List all tasks")
                print(response['output'])
            except Exception as e:
                print(f"Error: {str(e)}")
            
        elif choice == "5":
            print("Thank you for using ToDo Agent Application!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    with tracer:
        main()