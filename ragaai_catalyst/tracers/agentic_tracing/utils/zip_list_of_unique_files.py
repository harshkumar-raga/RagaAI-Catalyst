import os
import hashlib
import zipfile
import re
import ast
import importlib.util
import json
import astor
from pathlib import Path
import logging
from IPython import get_ipython
import ipynbname
import sys
logger = logging.getLogger(__name__)

if 'get_ipython' in locals():
    ipython_instance = get_ipython()
    if ipython_instance:
        ipython_instance.run_line_magic('reset', '-f out')

# Reinitialize logger to ensure it doesn't carry over logs from previous runs
logger = logging.getLogger(__name__)
for handler in logger.handlers[:]:  # Remove all old handlers
    logger.removeHandler(handler)
logging.basicConfig(level=logging.INFO)  # Set desired logging level


# Define the PackageUsageRemover class
class PackageUsageRemover(ast.NodeTransformer):
    def __init__(self, package_name):
        self.package_name = package_name
        self.imported_names = set()
    
    def visit_Import(self, node):
        filtered_names = []
        for name in node.names:
            if not name.name.startswith(self.package_name):
                filtered_names.append(name)
            else:
                self.imported_names.add(name.asname or name.name)
        
        if not filtered_names:
            return None
        node.names = filtered_names
        return node
    
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith(self.package_name):
            self.imported_names.update(n.asname or n.name for n in node.names)
            return None
        return node
    
    def visit_Assign(self, node):
        if self._uses_package(node.value):
            return None
        return node
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.imported_names:
            return None
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.imported_names:
                return None
        return node
    
    def _uses_package(self, node):
        if isinstance(node, ast.Name) and node.id in self.imported_names:
            return True
        if isinstance(node, ast.Call):
            return self._uses_package(node.func)
        if isinstance(node, ast.Attribute):
            return self._uses_package(node.value)
        return False

# Define the function to remove package code from a source code string
def remove_package_code(source_code: str, package_name: str) -> str:
    try:
        tree = ast.parse(source_code)
        transformer = PackageUsageRemover(package_name)
        modified_tree = transformer.visit(tree)
        modified_code = astor.to_source(modified_tree)
        return modified_code
    except Exception as e:
        raise Exception(f"Error processing source code: {str(e)}")

class JupyterNotebookHandler:
    @staticmethod
    def is_running_in_colab():
        """Check if the code is running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_running_in_notebook():
        """Check if the code is running in a Jupyter notebook or Colab."""
        try:
            shell = get_ipython().__class__.__name__
            if JupyterNotebookHandler.is_running_in_colab():
                return True
            return shell == 'ZMQInteractiveShell'
        except:
            return False
    
    @staticmethod
    def get_notebook_path():
        """Get the path of the current executing notebook."""
        try:
            # First try using ipynbname
            try:
                notebook_path = ipynbname.path()
                if notebook_path:
                    return str(notebook_path)
            except:
                pass

            # Try getting notebook path for regular Jupyter
            try:
                import IPython
                ipython = IPython.get_ipython()
                if ipython is not None:
                    # Try getting the notebook name from kernel
                    if hasattr(ipython, 'kernel') and hasattr(ipython.kernel, 'session'):
                        kernel_file = ipython.kernel.session.config.get('IPKernelApp', {}).get('connection_file', '')
                        if kernel_file:
                            kernel_id = Path(kernel_file).stem
                            current_dir = Path.cwd()
                            
                            # Look for .ipynb files in current and parent directories
                            for search_dir in [current_dir] + list(current_dir.parents):
                                notebooks = list(search_dir.glob('*.ipynb'))
                                recent_notebooks = [
                                    nb for nb in notebooks 
                                    if '.ipynb_checkpoints' not in str(nb)
                                ]
                                
                                if recent_notebooks:
                                    notebook_path = str(max(recent_notebooks, key=os.path.getmtime))
                                    return notebook_path

                    # Try alternative method using notebook metadata
                    try:
                        notebook_path = ipython.kernel._parent_ident
                        if notebook_path:
                            return notebook_path
                    except:
                        pass

            except Exception as e:
                logger.warning(f"Error in Jupyter notebook detection: {str(e)}")

            return None
            
        except Exception as e:
            return None

class TraceDependencyTracker:
    def __init__(self, output_dir=None):
        self.tracked_files = set()
        self.python_imports = set()
        self.notebook_path = None
        
        # Set output directory with Colab handling
        if JupyterNotebookHandler.is_running_in_colab():
            self.output_dir = '/content'
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            self.output_dir = output_dir or os.getcwd()
        
        self.jupyter_handler = JupyterNotebookHandler()

    def track_jupyter_notebook(self):
        """Track the current notebook and its dependencies."""
        if self.jupyter_handler.is_running_in_notebook():
            # Get notebook path using the enhanced handler
            notebook_path = self.jupyter_handler.get_notebook_path()
            
            if notebook_path:
                self.notebook_path = notebook_path
                self.track_file_access(notebook_path)
                
                # Track notebook dependencies
                try:
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        notebook_content = f.read()
                        # Find and track imported files
                        self.find_config_files(notebook_content, notebook_path)
                        # Track any additional dependencies
                        self.track_notebook_files(notebook_path)
                except Exception as e:
                    logger.warning(f"Error processing notebook dependencies: {str(e)}")
            else:
                logger.warning("No notebook path found to track")


    def track_file_access(self, filepath):
        if os.path.exists(filepath):
            self.tracked_files.add(os.path.abspath(filepath))

    def find_config_files(self, content, base_path):
        patterns = [
            r'(?:open|read|load|with\s+open)\s*\([\'"]([^\'"]*\.(?:json|yaml|yml|txt|cfg|config|ini))[\'"]',
            r'(?:config|cfg|conf|settings|file|path)(?:_file|_path)?\s*=\s*[\'"]([^\'"]*\.(?:json|yaml|yml|txt|cfg|config|ini))[\'"]',
            r'[\'"]([^\'"]*\.txt)[\'"]',
            r'[\'"]([^\'"]*\.(?:yaml|yml))[\'"]',
            r'from\s+(\S+)\s+import',
            r'import\s+(\S+)'
        ]
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                filepath = match.group(1)
                if not os.path.isabs(filepath):
                    full_path = os.path.join(os.path.dirname(base_path), filepath)
                else:
                    full_path = filepath
                if os.path.exists(full_path):
                    self.track_file_access(full_path)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            self.find_config_files(f.read(), full_path)
                    except (UnicodeDecodeError, IOError):
                        pass

    def track_notebook_files(self, notebook_path):
        """Track all files used in the Jupyter notebook."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.find_config_files(content, notebook_path)  # Find and track config files
            
            # Track PDF files dynamically
            pdf_pattern = r'\"([^\"]+\.pdf)\"'  # Regex to find PDF file paths
            matches = re.finditer(pdf_pattern, content)
            for match in matches:
                pdf_filepath = match.group(1)
                self.track_file_access(pdf_filepath)  # Track the PDF file

            # Track other file types if needed (e.g., images, text files)
            other_file_pattern = r'\"([^\"]+\.(pdf|txt|png|jpg|jpeg|csv))\"'  # Extend as needed
            other_matches = re.finditer(other_file_pattern, content)
            for match in other_matches:
                other_filepath = match.group(1)
                self.track_file_access(other_filepath)  # Track other files

        except Exception as e:
            print(f"Warning: Could not read notebook {notebook_path}: {str(e)}")


    def analyze_python_imports(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read(), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module_name = node.module
                    else:
                        for name in node.names:
                            module_name = name.name.split('.')[0]
                    try:
                        spec = importlib.util.find_spec(module_name)
                        if spec and spec.origin and not spec.origin.startswith(os.path.dirname(importlib.__file__)):
                            self.python_imports.add(spec.origin)
                    except (ImportError, AttributeError):
                        pass
        except Exception as e:
            print(f"Warning: Could not analyze imports in {filepath}: {str(e)}")


    def create_zip(self, filepaths):
        self.track_jupyter_notebook()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Special handling for Colab
        if self.jupyter_handler.is_running_in_colab():
            # Try to get the Colab notebook path
            colab_notebook = self.jupyter_handler.get_notebook_path()
            if colab_notebook:
                self.tracked_files.add(os.path.abspath(colab_notebook))

            # New feature: Save current cell content to a file in the zip folder
            self.check_environment_and_save()  # Call the new method

        # Ensure the dynamic_check_environment.ipynb is tracked
        self.track_file_access(os.path.join(self.output_dir, "dynamic_check_environment.ipynb"))

        # Process all files (existing code)
        for filepath in filepaths:
            abs_path = os.path.abspath(filepath)
            self.track_file_access(abs_path)
            try:
                with open(abs_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.find_config_files(content, abs_path)
                if filepath.endswith('.py'):
                    self.analyze_python_imports(abs_path)
            except Exception as e:
                logger.warning(f"Could not process {filepath}: {str(e)}")

        if self.notebook_path and os.path.exists(self.notebook_path):
            self.tracked_files.add(os.path.abspath(self.notebook_path))

        # Calculate hash and create zip
        self.tracked_files.update(self.python_imports)
        hash_contents = []

        for filepath in sorted(self.tracked_files):
            if 'env' in filepath:
                continue
            try:
                with open(filepath, 'rb') as file:
                    content = file.read()
                    if filepath.endswith('.py'):
                        content = remove_package_code(content.decode('utf-8'), 'ragaai_catalyst').encode('utf-8')
                    hash_contents.append(content)
            except Exception as e:
                logger.warning(f"Could not read {filepath} for hash calculation: {str(e)}")

        combined_content = b''.join(hash_contents)
        hash_id = hashlib.sha256(combined_content).hexdigest()

        # Create zip in the appropriate location
        zip_filename = os.path.join(self.output_dir, f'{hash_id}.zip')
        common_path = [os.path.abspath(p) for p in self.tracked_files if 'env' not in p]

        if common_path:
            base_path = os.path.commonpath(common_path)
        else:
            base_path = os.getcwd()

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in sorted(self.tracked_files):
                if 'env' in filepath:
                    continue
                try:
                    relative_path = os.path.relpath(filepath, base_path)
                    zipf.write(filepath, relative_path)
                    logger.info(f"Added to zip: {relative_path}")
                except Exception as e:
                    logger.warning(f"Could not add {filepath} to zip: {str(e)}")
        logger.info(f"Zip file created: {zip_filename}")
        return hash_id, zip_filename

    def check_environment_and_save(self):
        """Check if running in Colab and save current cell content in the zip folder."""
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if 'google.colab' in sys.modules:
                
                # Retrieve the current cell content dynamically in Colab
                current_cell = ipython.history_manager.get_range()
                script_content = "\n".join(input_line for _, _, input_line in current_cell if input_line.strip())

                # Save the retrieved script content to a file in the zip folder
                file_name = "dynamic_check_environment.ipynb"
                file_path = os.path.join(self.output_dir, file_name)  # Save in the zip folder

                with open(file_path, "w") as file:
                    file.write(script_content)
            else:
                logger.info("Not running on Google Colab.")
        except Exception as e:
            logger.warning(f"Error retrieving the current cell content: {e}")

def zip_list_of_unique_files(filepaths, output_dir=None):
    """Create a zip file containing all unique files and their dependencies."""
    if output_dir is None:
        if JupyterNotebookHandler.is_running_in_colab():
            output_dir = '/content'
        else:
            output_dir = os.getcwd()
    
    tracker = TraceDependencyTracker(output_dir)
    return tracker.create_zip(filepaths)


# Example usage
if __name__ == "__main__":
    filepaths = ["script1.py", "script2.py"]
    hash_id, zip_path = zip_list_of_unique_files(filepaths)
    print(f"Created zip file: {zip_path}")
    print(f"Hash ID: {hash_id}")

