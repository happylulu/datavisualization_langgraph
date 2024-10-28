import os
from langchain_core.tools import tool
import pandas as pd
from typing import Dict, Optional, Annotated, List
from logger import setup_logger
from load_cfg import WORKING_DIRECTORY

# Set up logger
logger = setup_logger()

@tool
def human_choice(feedback: Annotated[str, "The agent wants the human feedback."]) -> str:
    """
    Type 1 to regenerate the hypothesis,  2 to continue.
    """

