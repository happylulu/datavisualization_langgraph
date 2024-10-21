from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage,ToolMessage
from openai import InternalServerError
from state import State
from typing import Dict, Any
import json
import re
import os
from pathlib import Path
from langchain.agents import AgentExecutor, create_react_agent
# Set up logger
 

def agent_node(state: Dict[str, Any], agent: AgentExecutor, name: str) -> Dict[str, Any]:
    """
    Process an agent's action and update the state accordingly.
    """
    try:
        # Ensure all required keys exist in the state
        state_copy = state.copy()
        for key in ["messages", "hypothesis", "process_decision", "visualization_state", 
                    "searcher_state", "report_section", "quality_review", "needs_revision","code_state","process"]:
            if key not in state_copy:
                state_copy[key] = ""

        result = agent.invoke(state_copy)
       
        
        output = result["output"] if isinstance(result, dict) and "output" in result else str(result)
        
        ai_message = AIMessage(content=output, name=name)
        state["messages"] = state.get("messages", []) + [ai_message]
        state["sender"] = name
        
        # Update specific state keys based on agent name
        if name == "hypothesis_agent" and not state.get("hypothesis"):
            state["hypothesis"] = ai_message
        elif name == "process_agent":
            state["process_decision"] = ai_message
        elif name == "visualization_agent":
            state["visualization_state"] = ai_message
        elif name == "searcher_agent":
            state["searcher_state"] = ai_message
        elif name == "report_agent":
            state["report_section"] = ai_message
        elif name == "quality_review_agent":
            state["quality_review"] = ai_message
            state["needs_revision"] = "revision needed" in output.lower()
        
        return state
    except Exception as e:
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return {"messages": state.get("messages", []) + [error_message]}

from typing import Dict, Any
from langchain_core.messages import HumanMessage

def human_choice_node(
    state: State
) -> State:
    """
    Handle human input to choose the next step in the process.
    If regenerating hypothesis, accept specific areas to modify.
    """

    while True:
            choice = input("Enter '1' to regenerate analysis, or '2' to continue the research: ").lower()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please provide '1' or '2'.")

    if choice == "1":
        modification_areas = input("Please enter your additional analysis request: ")
        
        if modification_areas is None:
            modification_areas = ""
        content = f"Regenerate hypothesis. Areas to modify: {modification_areas}"
        state["hypothesis"] = ""
        state["modification_areas"] = modification_areas
    else:
        content = "Continue the research process"
        state["process"] = "Continue the research process"

    human_message = HumanMessage(content=content)

    state["messages"] = state.get("messages", []) + [human_message]
    state["sender"] = 'human'

    return state

def create_message(message: dict[str], name: str) -> BaseMessage:
    """
    Create a BaseMessage object based on the message type.
    """
    content = message.get("content", "")
    message_type = message.get("type", "").lower()
    return HumanMessage(content=content) if message_type == "human" else AIMessage(content=content, name=name)

def note_agent_node(state: State, agent: AgentExecutor, name: str) -> State:
    """
    Process the note agent's action and update the entire state.
    """
    try:
        current_messages = state.get("messages", [])
        
        head_messages, tail_messages = [], []
        
        if len(current_messages) > 6:
            head_messages = current_messages[:2] 
            tail_messages = current_messages[-2:]
            state = {**state, "messages": current_messages[2:-2]}
             
        
        result = agent.invoke(state)
       
        output = result["output"] if isinstance(result, dict) and "output" in result else str(result)

        cleaned_output = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', output)
        parsed_output = json.loads(cleaned_output)
       

        new_messages = [create_message(msg, name) for msg in parsed_output.get("messages", [])]
        
        messages = new_messages if new_messages else current_messages
        
        combined_messages = head_messages + messages + tail_messages
        
        updated_state: State = {
            "messages": combined_messages,
            "hypothesis": str(parsed_output.get("hypothesis", state.get("hypothesis", ""))),
            "process": str(parsed_output.get("process", state.get("process", ""))),
            "process_decision": str(parsed_output.get("process_decision", state.get("process_decision", ""))),
            "visualization_state": str(parsed_output.get("visualization_state", state.get("visualization_state", ""))),
            "searcher_state": str(parsed_output.get("searcher_state", state.get("searcher_state", ""))),
            "code_state": str(parsed_output.get("code_state", state.get("code_state", ""))),
            "report_section": str(parsed_output.get("report_section", state.get("report_section", ""))),
            "quality_review": str(parsed_output.get("quality_review", state.get("quality_review", ""))),
            "needs_revision": bool(parsed_output.get("needs_revision", state.get("needs_revision", False))),
            "sender": 'note_agent'
        }
        
      
        return updated_state

    except json.JSONDecodeError as e:
 
        return _create_error_state(state, AIMessage(content=f"Error parsing output: {output}", name=name), name, "JSON decode error")

    except InternalServerError as e:
 
        return _create_error_state(state, AIMessage(content=f"OpenAI Error: {str(e)}", name=name), name, "OpenAI error")

    except Exception as e:
 
        return _create_error_state(state, AIMessage(content=f"Unexpected error: {str(e)}", name=name), name, "Unexpected error")

def _create_error_state(state: State, error_message: AIMessage, name: str, error_type: str) -> State:
    """
    Create an error state when an exception occurs.
    """
 
    error_state:State = {
            "messages": state.get("messages", []) + [error_message],
            "hypothesis": str(state.get("hypothesis", "")),
            "process": str(state.get("process", "")),
            "process_decision": str(state.get("process_decision", "")),
            "visualization_state": str(state.get("visualization_state", "")),
            "searcher_state": str(state.get("searcher_state", "")),
            "code_state": str(state.get("code_state", "")),
            "report_section": str(state.get("report_section", "")),
            "quality_review": str(state.get("quality_review", "")),
            "needs_revision": bool(state.get("needs_revision", False)),
            "sender": 'note_agent'
        }
    return error_state

def human_review_node(state: State) -> State:
    """
    Display current state to the user and update the state based on user input.
    Includes error handling for robustness.
    """
    try:
        print("Current research progress:")
        print(state)
        print("\nDo you need additional analysis or modifications?")
        
        while True:
            user_input = input("Enter 'yes' to continue analysis, or 'no' to end the research: ").lower()
            if user_input in ['yes', 'no']:
                break
            print("Invalid input. Please enter 'yes' or 'no'.")
        
        if user_input == 'yes':
            while True:
                additional_request = input("Please enter your additional analysis request: ").strip()
                if additional_request:
                    state["messages"].append(HumanMessage(content=additional_request))
                    state["needs_revision"] = True
                    break
                print("Request cannot be empty. Please try again.")
        else:
            state["needs_revision"] = False
        
        state["sender"] = "human"
 
        return state
    
    except KeyboardInterrupt:
 
        return None
    
    except Exception as e:
 
        return None
    
def refiner_node(state: State, agent: AgentExecutor, name: str) -> State:
    """
    Read MD file contents and PNG file names from the specified storage path,
    add them as report materials to a new message,
    then process with the agent and update the original state.
    If token limit is exceeded, use only MD file names instead of full content.
    """
    try:
        # Get storage path
        storage_path = Path(os.getenv('STORAGE_PATH', './data_storage/'))
        
        # Collect materials
        materials = []
        md_files = list(storage_path.glob("*.md"))
        png_files = list(storage_path.glob("*.png"))
        
        # Process MD files
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                materials.append(f"MD file '{md_file.name}':\n{f.read()}")
        
        # Process PNG files
        materials.extend(f"PNG file: '{png_file.name}'" for png_file in png_files)
        
        # Combine materials
        combined_materials = "\n\n".join(materials)
        report_content = f"Report materials:\n{combined_materials}"
        
        # Create refiner state
        refiner_state = state.copy()
        refiner_state["messages"] = [BaseMessage(content=report_content)]
        
        try:
            # Attempt to invoke agent with full content
            result = agent.invoke(refiner_state)
        except Exception as token_error:
            # If token limit is exceeded, retry with only MD file names
 
            md_file_names = [f"MD file: '{md_file.name}'" for md_file in md_files]
            png_file_names = [f"PNG file: '{png_file.name}'" for png_file in png_files]
            
            simplified_materials = "\n".join(md_file_names + png_file_names)
            simplified_report_content = f"Report materials (file names only):\n{simplified_materials}"
            
            refiner_state["messages"] = [BaseMessage(content=simplified_report_content)]
            result = agent.invoke(refiner_state)
        
        # Update original state
        state["messages"].append(BaseMessage(content=result))
        state["sender"] = name
        
   
        return state
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Error: {str(e)}", name=name))
        return state   
 