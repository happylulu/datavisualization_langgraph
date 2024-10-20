import os
from typing import Annotated
import subprocess
from langchain_core.tools import tool
 
from load_cfg import WORKING_DIRECTORY,CONDA_PATH,CONDA_ENV
# Initialize logger
 
# Ensure the storage directory exists
if not os.path.exists(WORKING_DIRECTORY):
    os.makedirs(WORKING_DIRECTORY)
    
@tool
def execute_code(
    input_code: Annotated[str, "The Python code to execute."],
    codefile_name: Annotated[str, "The Python code file name or full path."] = 'code.py'
):
    """
    Execute Python code  and return the result.

    This function takes Python code as input, writes it to a file,  and returns the output or any errors encountered during execution.

    Args:
    input_code (str): The Python code to be executed.
    codefile_name (str): The name of the file to save the code in, or the full path.

    Returns:
    dict: A dictionary containing the execution result, output, and file path.
    """
    try:
        # Ensure WORKING_DIRECTORY exists
        os.makedirs(WORKING_DIRECTORY, exist_ok=True)
        # Handle codefile_name, ensuring it's a valid path
        if os.path.isabs(codefile_name):
            # If it's an absolute path, use it as is
            code_file_path = codefile_name
        else:
            if WORKING_DIRECTORY not in codefile_name:
                code_file_path = os.path.join(WORKING_DIRECTORY, codefile_name)
            else:
                code_file_path = codefile_name

        # Normalize the path
        code_file_path = os.path.normpath(code_file_path)
 
        
        # Write the code to the file
        with open(code_file_path, 'w') as code_file:
            code_file.write(input_code)
 
        
 
        python_cmd = f"python {codefile_name}"
        full_command = f"{python_cmd}"
   
        
        # Execute the code
        result = subprocess.run(
            ['/bin/bash', '-c', full_command],
            capture_output=True,
            text=True,
            cwd=WORKING_DIRECTORY
        )
        
        # Capture standard output and error output
        output = result.stdout
        error_output = result.stderr
        
        if result.returncode == 0:
            
            return {
                "result": "Code executed successfully",
                "output": output + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.",
                "file_path": code_file_path
            }
        else:
             
            return {
                "result": "Failed to execute",
                "error": error_output,
                "file_path": code_file_path
            }
    except Exception as e:
       
        return {
            "result": "Error occurred",
            "error": str(e),
            "file_path": code_file_path if 'code_file_path' in locals() else "Unknown"
        }

@tool
def execute_command(
    command: Annotated[str, "Command to be executed."]
) -> Annotated[str, "Output of the command."]:
    """
    Execute a command and return its output.

    This function install the necessary packes , executes the given command,
    and returns the output or any errors encountered during execution.
    Please use pip to install the package.
    Args:
    command (str): The command to be executed in the Conda environment.

    Returns:
    str: The output of the command or an error message.
    """
    try:
        # Construct the command to activate the Conda environment and execute the given command
        source = f"source {CONDA_PATH}/etc/profile.d/conda.sh"
        conda_activate = f"conda activate {CONDA_ENV}"
        full_command = f"{source} && {conda_activate} && {command}"
        
    
        
        # Execute the command and capture the output
        result = subprocess.run(
            full_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable="/bin/bash",
            cwd=WORKING_DIRECTORY
        )
        
        return result.stdout
    except subprocess.CalledProcessError as e:
   
        return f"Error: {e.stderr}"
 