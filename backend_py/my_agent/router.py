from state import State
from typing import Literal
from langchain_core.messages import AIMessage
import logging
import json

# Set up logger
 

# Define types for node routing
NodeType = Literal['Visualization', 'Search', 'Coder', 'Report', 'Process', 'NoteTaker', 'Hypothesis', 'QualityReview']
ProcessNodeType = Literal['Coder', 'Search', 'Visualization', 'Report', 'Process', 'Refiner']

def hypothesis_router(state: State) -> NodeType:
    """
    Route based on the presence of a hypothesis in the state.

    Args:
    state (State): The current state of the system.

    Returns:
    NodeType: 'Hypothesis' if no hypothesis exists, otherwise 'Process'.
    """
    hypothesis = state.get("hypothesis")
    
    if isinstance(hypothesis, AIMessage):
        hypothesis_content = hypothesis.content
         
    elif isinstance(hypothesis, str):
        hypothesis_content = hypothesis
         
    else:
        hypothesis_content = ""
         
    
    result = "Hypothesis" if not hypothesis_content.strip() else "Process"
     
    return result

def QualityReview_router(state: State) -> NodeType:
    """
    Route based on the quality review outcome and process decision.

    Args:
    state (State): The current state of the system.

    Returns:
    NodeType: The next node to route to based on the quality review and process decision.
    """
     
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    # Check if revision is needed
    if (last_message and 'REVISION' in str(last_message.content)) or state.get("needs_revision", False):
        previous_node = state.get("last_sender", "")
        revision_routes = {
            "Visualization": "Visualization",
            "Search": "Search",
            "Coder": "Coder",
            "Report": "Report"
        }
        result = revision_routes.get(previous_node, "NoteTaker")
         
        return result
    
    else:
        return "NoteTaker"
    

def process_router(state: State) -> ProcessNodeType:
    """
    Route based on the process decision in the state.

    Args:
    state (State): The current state of the system.

    Returns:
    ProcessNodeType: The next process node to route to based on the process decision.
    """
     
    process_decision = state.get("process_decision", "")
    
    # Handle AIMessage object
    if isinstance(process_decision, AIMessage):
        try:
            # Attempt to parse JSON in content
            decision_dict = json.loads(process_decision.content.replace("'", '"'))
            process_decision = decision_dict.get('next', '')
             
        except json.JSONDecodeError:
            # If JSON parsing fails, use content directly
            process_decision = process_decision.content
             
    elif isinstance(process_decision, dict):
        process_decision = process_decision.get('next', '')
    elif not isinstance(process_decision, str):
        process_decision = str(process_decision)
         
    # Define valid decisions
    valid_decisions = {"Coder", "Search", "Visualization", "Report"}
    
    if process_decision in valid_decisions:
        return process_decision
    
    if process_decision == "FINISH":
        return "Refiner"
    
    # If process_decision is empty or not a valid decision, return "Process"
    if not process_decision or process_decision not in valid_decisions:
        return "Process"
    
    # Default to "Process"
     
    return "Process"

