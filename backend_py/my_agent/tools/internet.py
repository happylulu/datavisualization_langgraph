import os
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader, FireCrawlLoader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from typing import Annotated, List
from bs4 import BeautifulSoup
from logger import setup_logger
from load_cfg import FIRECRAWL_API_KEY,CHROMEDRIVER_PATH
import requests
# Set up logger
logger = setup_logger()

@tool
def google_search(query: Annotated[str, "The search query to use"]) -> str:
    """
    Perform a Google search based on the given query and return the top 5 results.

    This function uses Selenium to perform a headless Google search and BeautifulSoup to parse the results.

    Args:
    query (str): The search query to use.

    Returns:
    str: A string containing the titles, snippets, and links of the top 5 search results.

    Raises:
    Exception: If there's an error during the search process.
    """
    try:
        logger.info(f"Performing Google search for query: {query}")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service(CHROMEDRIVER_PATH)

        with webdriver.Chrome(options=chrome_options, service=service) as driver:
            url = f"https://www.google.com/search?q={query}"
            logger.debug(f"Accessing URL: {url}")
            driver.get(url)
            html = driver.page_source

        soup = BeautifulSoup(html, 'html.parser')
        search_results = soup.select('.g') 
        search = ""
        for result in search_results[:5]:
            title_element = result.select_one('h3')
            title = title_element.text if title_element else 'No Title'
            snippet_element = result.select_one('.VwiC3b')
            snippet = snippet_element.text if snippet_element else 'No Snippet'
            link_element = result.select_one('a')
            link = link_element['href'] if link_element else 'No Link'
            search += f"{title}\n{snippet}\n{link}\n\n"

        logger.info("Google search completed successfully")
        return search
    except Exception as e:
        logger.error(f"Error during Google search: {str(e)}")
        return f'Error: {e}'
@tool
def scrape_webpages(urls: Annotated[List[str], "List of URLs to scrape"]) -> str:
    """
    Scrape the provided web pages for detailed information using WebBaseLoader.

    This function uses the WebBaseLoader to load and scrape the content of the provided URLs.

    Args:
    urls (List[str]): A list of URLs to scrape.

    Returns:
    str: A string containing the concatenated content of all scraped web pages.

    Raises:
    Exception: If there's an error during the scraping process.
    """
    try:
        logger.info(f"Scraping webpages: {urls}")
        loader = WebBaseLoader(urls)
        docs = loader.load()
        content = "\n\n".join([f'\n{doc.page_content}\n' for doc in docs])
        logger.info("Webpage scraping completed successfully")
        return content
    except Exception as e:
        logger.error(f"Error during webpage scraping: {str(e)}")
        raise  # Re-raise the exception to be caught by the calling function
@tool
def FireCrawl_scrape_webpages(urls: Annotated[List[str], "List of URLs to scrape"]) -> str:
    """
    Scrape the provided web pages for detailed information using FireCrawlLoader.

    This function uses the FireCrawlLoader to load and scrape the content of the provided URLs.

    Args:
    urls (List[str]): A list of URLs to scrape.

    Returns:
    Any: The result of the FireCrawlLoader's load operation.

    Raises:
    Exception: If there's an error during the scraping process or if the API key is not set.
    """
    if not FIRECRAWL_API_KEY:
        raise ValueError("FireCrawl API key is not set")

    try:
        logger.info(f"Scraping webpages using FireCrawl: {urls}")
        loader = FireCrawlLoader(
            api_key=FIRECRAWL_API_KEY,
            url=urls,
            mode="scrape"
        )
        result = loader.load()
        logger.info("FireCrawl scraping completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during FireCrawl scraping: {str(e)}")
        raise  # Re-raise the exception to be caught by the calling function
@tool
def scrape_webpages_with_fallback(urls: Annotated[List[str], "List of URLs to scrape"]) -> str:
    """
    Attempt to scrape webpages using FireCrawl, falling back to WebBaseLoader if unsuccessful.

    Args:
    urls (List[str]): A list of URLs to scrape.

    Returns:
    str: The scraped content from either FireCrawl or WebBaseLoader.
    """
    try:
        return FireCrawl_scrape_webpages(urls)
    except Exception as e:
        logger.warning(f"FireCrawl scraping failed: {str(e)}. Falling back to WebBaseLoader.")
        try:
            return scrape_webpages(urls)
        except Exception as e:
            logger.error(f"Both scraping methods failed. Error: {str(e)}")
            return f"Error: Unable to scrape webpages using both methods. {str(e)}"

 

 
@tool
def clinical_trials_search(query: Annotated[str, "Search query for ClinicalTrials.gov"]) -> str:
    """
    Search ClinicalTrials.gov for clinical trials based on the provided query.

    This function queries the ClinicalTrials.gov API for clinical trials information related to the given query.

    Args:
    query (str): The search query to use for finding clinical trials, should be only related to disease or intervention, less than two words.

    Returns:
    str: A string containing the titles, study types, statuses, and URLs of the top 5 results.
    """
    try:
        logger.info(f"Performing ClinicalTrials.gov search for query: {query}")
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.titles": query,
            "pageSize": 5,  # Limit to the top 5 results
        }

        # Make the request to ClinicalTrials.gov API
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Retrieve the list of studies from the response
        studies = data.get("studies", [])
        
        if not studies:
            return f"No clinical trials found for query: {query}"
        
        # Parse the study data and format results
        results = ""
        for study in studies:
            # Safely access nested keys
            nctId = study['protocolSection']['identificationModule'].get('nctId', 'Unknown')
            overallStatus = study['protocolSection']['statusModule'].get('overallStatus', 'Unknown')
            startDate = study['protocolSection']['statusModule'].get('startDateStruct', {}).get('date', 'Unknown Date')
            conditions = ', '.join(study['protocolSection']['conditionsModule'].get('conditions', ['No conditions listed']))
            acronym = study['protocolSection']['identificationModule'].get('acronym', 'Unknown')

            # Extract interventions safely
            interventions_list = study['protocolSection'].get('armsInterventionsModule', {}).get('interventions', [])
            interventions = ', '.join([intervention.get('name', 'No intervention name listed') for intervention in interventions_list]) if interventions_list else "No interventions listed"
            
            # Extract locations safely
            locations_list = study['protocolSection'].get('contactsLocationsModule', {}).get('locations', [])
            locations = ', '.join([f"{location.get('city', 'No City')} - {location.get('country', 'No Country')}" for location in locations_list]) if locations_list else "No locations listed"
            
            # Extract dates and phases
            primaryCompletionDate = study['protocolSection']['statusModule'].get('primaryCompletionDateStruct', {}).get('date', 'Unknown Date')
            studyFirstPostDate = study['protocolSection']['statusModule'].get('studyFirstPostDateStruct', {}).get('date', 'Unknown Date')
            lastUpdatePostDate = study['protocolSection']['statusModule'].get('lastUpdatePostDateStruct', {}).get('date', 'Unknown Date')
            studyType = study['protocolSection']['designModule'].get('studyType', 'Unknown')
            phases = ', '.join(study['protocolSection']['designModule'].get('phases', ['Not Available']))
            
            # Format the study information into a readable string
            results += (
                f"Title: {acronym}\n"
                f"NCT ID: {nctId}\n"
                f"Overall Status: {overallStatus}\n"
                f"Start Date: {startDate}\n"
                f"Conditions: {conditions}\n"
                f"Interventions: {interventions}\n"
                f"Locations: {locations}\n"
                f"Primary Completion Date: {primaryCompletionDate}\n"
                f"Study First Post Date: {studyFirstPostDate}\n"
                f"Last Update Post Date: {lastUpdatePostDate}\n"
                f"Study Type: {studyType}\n"
                f"Phases: {phases}\n"
                f"Link: https://clinicaltrials.gov/ct2/show/{nctId}\n\n"
            )
        
        logger.info("ClinicalTrials.gov search completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error during ClinicalTrials.gov search: {str(e)}")
        return f"Error: {e}"


logger.info("Web scraping tools initialized")