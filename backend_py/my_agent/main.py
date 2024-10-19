import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from load_cfg import OPENAI_API_KEY, LANGCHAIN_API_KEY, WORKING_DIRECTORY
from state import State
from node import agent_node, human_choice_node, note_agent_node, human_review_node, refiner_node
from create_agent import create_agent, create_supervisor, create_note_agent
from router import QualityReview_router, hypothesis_router, process_router
from tools.internet import google_search, scrape_webpages_with_fallback,clinical_trials_search
from tools.basetool import execute_code, execute_command
from tools.FileEdit import create_document, read_document, edit_document, collect_data
from langchain.agents import load_tools
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Set environment variables
 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-Agent Data Analysis System"

 

# Initialize language models
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4096)
    power_llm = ChatOpenAI(model="gpt-4o", temperature=0.5, max_tokens=4096)
    json_llm = ChatOpenAI(
        model="gpt-4o",
        model_kwargs={"response_format": {"type": "json_object"}},
        temperature=0,
        max_tokens=4096
    )
    
except Exception as e:
     
    raise

# Ensure working directory exists
if not os.path.exists(WORKING_DIRECTORY):
    os.makedirs(WORKING_DIRECTORY)
from state import State
from node import agent_node,human_choice_node,note_agent_node,human_review_node,refiner_node
from create_agent import create_agent,create_supervisor
from router import QualityReview_router,hypothesis_router,process_router    
# Create state graph for the workflow
workflow = StateGraph(State)
members = ["Hypothesis","Process","Visualization", "Search", "Coder", "Report", "QualityReview","Refiner"]
 
# Initialize tools and agents
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

 

hypothesis_agent = create_agent(
    llm, 
    [collect_data, wikipedia, google_search, scrape_webpages_with_fallback] + load_tools(["arxiv"]),
    '''
    As an esteemed expert in data analysis, your task is to formulate a set of research hypotheses and outline the steps to be taken based on the information table provided. Utilize statistics, machine learning, deep learning, and artificial intelligence in developing these hypotheses. Your hypotheses should be precise, achievable, professional, and innovative. To ensure the feasibility and uniqueness of your hypotheses, thoroughly investigate relevant information. For each hypothesis, include ample references to support your claims.

    Upon analyzing the information table, you are required to:

    1. Formulate research hypotheses that leverage statistics, machine learning, deep learning, and AI techniques.
    2. Outline the steps involved in testing these hypotheses.
    3. Verify the feasibility and uniqueness of each hypothesis through a comprehensive literature review.

    At the conclusion of your analysis, present the complete research hypotheses, elaborate on their uniqueness and feasibility, and provide relevant references to support your assertions. Please answer in structured way to enhance readability.
    Just answer a research hypothesis.
    ''',
    members, WORKING_DIRECTORY)

process_agent = create_supervisor(
    power_llm,
    """
    You are a research supervisor responsible for overseeing and coordinating a comprehensive data analysis project, resulting in a complete and cohesive research report. Your primary tasks include:

    1. Validating and refining the research hypothesis to ensure it is clear, specific, and testable.
    2. Orchestrating a thorough data analysis process, with all code well-documented and reproducible.
    3. Compiling and refining a research report that includes:
        - Introduction
        - Hypothesis
        - Methodology
        - Results, accompanied by relevant visualizations
        - Discussion
        - Conclusion
        - References

    **Step-by-Step Process:**
    1. **Planning:** Define clear objectives and expected outcomes for each phase of the project.
    2. **Task Assignment:** Assign specific tasks to the appropriate agents ("Visualization," "Search," "Coder," "Report").
    3. **Review and Integration:** Critically review and integrate outputs from each agent, ensuring consistency, quality, and relevance.
    4. **Feedback:** Provide feedback and further instructions as needed to refine outputs.
    5. **Final Compilation:** Ensure all components are logically connected and meet high academic standards.

    **Agent Guidelines:**
    - **Visualization Agent:** Develop and explain data visualizations that effectively communicate key findings.
    - **Search Agent:** Collect and summarize relevant information, and compile a comprehensive list of references.
    - **Coder Agent:** Write and document efficient Python code for data analysis, ensuring that the code is clean and reproducible.
    - **Report Agent:** Draft, refine, and finalize the research report, integrating inputs from all agents and ensuring the narrative is clear and cohesive.

    **Workflow:**
    1. Plan the overall analysis and reporting process.
    2. Assign tasks to the appropriate agents and oversee their progress.
    3. Continuously review and integrate the outputs from each agent, ensuring that each contributes effectively to the final report.
    4. Adjust the analysis and reporting process based on emerging results and insights.
    5. Compile the final report, ensuring all sections are complete and well-integrated.

    **Completion Criteria:**
    Respond with "FINISH" only when:
    1. The hypothesis has been thoroughly tested and validated.
    2. The data analysis is complete, with all code documented and reproducible.
    3. All required visualizations have been created, properly labeled, and explained.
    4. The research report is comprehensive, logically structured, and includes all necessary sections.
    5. The reference list is complete and accurately cited.
    6. All components are cohesively integrated into a polished final report.

    Ensure that the final report delivers a clear, insightful analysis, addressing all aspects of the hypothesis and meeting the highest academic standards.
    """,
    ["Visualization", "Search", "Coder", "Report"],
)

visualization_agent = create_agent(
    llm, 
    [read_document, execute_code, execute_command],
    """
    You are a data visualization expert tasked with creating insightful visual representations of data. Your primary responsibilities include:
    
    1. Designing appropriate visualizations that clearly communicate data trends and patterns.
    2. Selecting the most suitable chart types (e.g., bar charts, scatter plots, heatmaps) for different data types and analytical purposes.
    3. Providing executable Python code (using libraries such as matplotlib, seaborn, or plotly) that generates these visualizations.
    4. Including well-defined titles, axis labels, legends, and saving the visualizations as files.
    5. Offering brief but clear interpretations of the visual findings.

    **File Saving Guidelines:**
    - Save all visualizations as files with descriptive and meaningful filenames.
    - Ensure filenames are structured to easily identify the content (e.g., 'sales_trends_2024.png' for a sales trend chart).
    - Confirm that the saved files are organized in the working directory, making them easy for other agents to locate and use.

    **Constraints:**
    - Focus solely on visualization tasks; do not perform data analysis or preprocessing.
    - Ensure all visual elements are suitable for the target audience, with attention to color schemes and design principles.
    - Avoid over-complicating visualizations; aim for clarity and simplicity.
    """,
    members, WORKING_DIRECTORY
)

code_agent = create_agent(
    power_llm,
    [read_document, execute_code, execute_command],
    """
    You are an expert Python programmer specializing in data processing and analysis. Your main responsibilities include:

    1. Writing clean, efficient Python code for data manipulation, cleaning, and transformation.
    2. Implementing statistical methods and machine learning algorithms as needed.
    3. Debugging and optimizing existing code for performance improvements.
    4. Adhering to PEP 8 standards and ensuring code readability with meaningful variable and function names.

    Constraints:
    - Focus solely on data processing tasks; do not generate visualizations or write non-Python code.
    - Provide only valid, executable Python code, including necessary comments for complex logic.
    - Avoid unnecessary complexity; prioritize readability and efficiency.
    """,
    members, WORKING_DIRECTORY
)

searcher_agent = create_agent(
    llm,
    [create_document, read_document, collect_data, wikipedia,clinical_trials_search, scrape_webpages_with_fallback] + load_tools(["arxiv"]),
    """
    You are a skilled research assistant responsible for gathering and summarizing relevant information. Your main tasks include:

    1. Conducting thorough literature reviews using academic databases and reputable online sources.
    2. Summarizing key findings in a clear, concise manner.
    3. Providing citations for all sources, prioritizing peer-reviewed and academically reputable materials.

    Constraints:
    - Focus exclusively on information retrieval and summarization; do not engage in data analysis or processing.
    - Present information in an organized format, with clear attributions to sources.
    - Evaluate the credibility of sources and prioritize high-quality, reliable information.
    """,
    members, WORKING_DIRECTORY
)

report_agent = create_agent(
    power_llm, 
    [create_document, read_document, edit_document], 
    """
    You are an experienced scientific writer tasked with drafting comprehensive research reports. Your primary duties include:

    1. Clearly stating the research hypothesis and objectives in the introduction.
    2. Detailing the methodology used, including data collection and analysis techniques.
    3. Structuring the report into coherent sections (e.g., Introduction, Methodology, Results, Discussion, Conclusion).
    4. Synthesizing information from various sources into a unified narrative.
    5. Integrating relevant data visualizations and ensuring they are appropriately referenced and explained.

    Constraints:
    - Focus solely on report writing; do not perform data analysis or create visualizations.
    - Maintain an objective, academic tone throughout the report.
    - Cite all sources using APA style and ensure that all findings are supported by evidence.
    """,
    members, WORKING_DIRECTORY
)

quality_review_agent = create_agent(
    llm, 
    [create_document, read_document, edit_document], 
    '''
    You are a meticulous quality control expert responsible for reviewing and ensuring the high standard of all research outputs. Your tasks include:

    1. Critically evaluating the content, methodology, and conclusions of research reports.
    2. Checking for consistency, accuracy, and clarity in all documents.
    3. Identifying areas that need improvement or further elaboration.
    4. Ensuring adherence to scientific writing standards and ethical guidelines.

    After your review, if revisions are needed, respond with 'REVISION' as a prefix, set needs_revision=True, and provide specific feedback on parts that need improvement. If no revisions are necessary, respond with 'CONTINUE' as a prefix and set needs_revision=False.
    ''',
    members, WORKING_DIRECTORY
)

note_agent = create_note_agent(
    json_llm, 
    [read_document], 
    '''
    You are a meticulous research process note-taker. Your main responsibility is to observe, summarize, and document the actions and findings of the research team. Your tasks include:

    1. Observing and recording key activities, decisions, and discussions among team members.
    2. Summarizing complex information into clear, concise, and accurate notes.
    3. Organizing notes in a structured format that ensures easy retrieval and reference.
    4. Highlighting significant insights, breakthroughs, challenges, or any deviations from the research plan.
    5. Responding only in JSON format to ensure structured documentation.

    Your output should be well-organized and easy to integrate with other project documentation.
    '''
)

refiner_agent = create_agent(
    power_llm,  
    [read_document, edit_document, create_document, collect_data, wikipedia, google_search, scrape_webpages_with_fallback] + load_tools(["arxiv"]),
    '''
    You are an expert AI report refiner tasked with optimizing and enhancing research reports. Your responsibilities include:

    1. Thoroughly reviewing the entire research report, focusing on content, structure, and readability.
    2. Identifying and emphasizing key findings, insights, and conclusions.
    3. Restructuring the report to improve clarity, coherence, and logical flow.
    4. Ensuring that all sections are well-integrated and support the primary research hypothesis.
    5. Condensing redundant or repetitive content while preserving essential details.
    6. Enhancing the overall readability, ensuring the report is engaging and impactful.

    Refinement Guidelines:
    - Maintain the scientific accuracy and integrity of the original content.
    - Ensure all critical points from the original report are preserved and clearly articulated.
    - Improve the logical progression of ideas and arguments.
    - Highlight the most significant results and their implications for the research hypothesis.
    - Ensure that the refined report aligns with the initial research objectives and hypothesis.

    After refining the report, submit it for final human review, ensuring it is ready for publication or presentation.
    ''',
    members,  
    WORKING_DIRECTORY
)

 

# Add nodes to the workflow
workflow.add_node("Hypothesis", lambda state: agent_node(state, hypothesis_agent, "hypothesis_agent"))
workflow.add_node("Process", lambda state: agent_node(state, process_agent, "process_agent"))
workflow.add_node("Visualization", lambda state: agent_node(state, visualization_agent, "visualization_agent"))
workflow.add_node("Search", lambda state: agent_node(state, searcher_agent, "searcher_agent"))
workflow.add_node("Coder", lambda state: agent_node(state, code_agent, "code_agent"))
workflow.add_node("Report", lambda state: agent_node(state, report_agent, "report_agent"))
workflow.add_node("QualityReview", lambda state: agent_node(state, quality_review_agent, "quality_review_agent"))
workflow.add_node("NoteTaker", lambda state: note_agent_node(state, note_agent, "note_agent"))
workflow.add_node("HumanChoice", human_choice_node)
workflow.add_node("HumanReview", human_review_node)
workflow.add_node("Refiner", lambda state: refiner_node(state, refiner_agent, "refiner_agent"))

# Add edges to the workflow
workflow.add_edge("Hypothesis", "HumanChoice")
workflow.add_conditional_edges(
    "HumanChoice",
    hypothesis_router,
    {
        "Hypothesis": "Hypothesis",
        "Process": "Process"
    }
)

workflow.add_conditional_edges(
    "Process",
    process_router,
    {
        "Coder": "Coder",
        "Search": "Search",
        "Visualization": "Visualization",
        "Report": "Report",
        "Process": "Process",
        "Refiner": "Refiner",
    }
)

for member in ["Visualization",'Search','Coder','Report']:
    workflow.add_edge(member, "QualityReview")

workflow.add_conditional_edges(
    "QualityReview",
    QualityReview_router,
    {
        'Visualization': "Visualization",
        'Search': "Search",
        'Coder': "Coder",
        'Report': "Report",
        'NoteTaker': "NoteTaker",
    }
)
workflow.add_edge("NoteTaker", "Process")
workflow.add_edge("Refiner", "HumanReview")
# Add an edge from HumanReview to Process
workflow.add_conditional_edges(
    "HumanReview",
    lambda state: "Process" if state and state.get("needs_revision", False) else "END",
    {
        "Process": "Process",
        "END": END
    }
)

from langgraph.checkpoint.memory import MemorySaver
workflow.add_edge(START, "Hypothesis")
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["HumanChoice","HumanReview"])

 