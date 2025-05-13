from dotenv import load_dotenv
import json
import re
import os
import pandas as pd 

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseTransformOutputParser, StrOutputParser
from langgraph.graph import START, END, StateGraph 
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI, HTTPException, Request
import uvicorn
import asyncio

from agents.AgentActions import AgentActions
from agents.ChapterSelector import ChapterSelector
from agents.CodeExtractor import CodeExtractor
from agents.LevelOneSelector import LevelOneSelector
from agents.DeepSelector import DeepSelector
from agents.FinalSelector import FinalSelector
# from agents.Gmail import create_message_with_attachment, send_message

from composio import ComposioToolSet, Action
# from firecrawl import FirecrawlApp

import logfire

# Load environment variables
load_dotenv()


try:
    LOGFIRE_TOKEN = os.getenv('LOGFIRE_TOKEN')
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TARIFFY_ORG_ID = os.getenv("TARIFFY_ORG_ID")
    TARIFFY_API_KEY = os.getenv("TARIFFY_API_KEY")
    SIMPLEDUTY_API_KEY = os.getenv("SIMPLEDUTY_API_KEY")
    # GMAIL_TOKEN = os.getenv("GMAIL_TOKEN")
    COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
    COMPOSIO_ENTITY_ID = 'default'
except Exception as e:
    raise ValueError("Environment variables not set correctly") from e

# Initialize Logfire
logfire.configure(token=LOGFIRE_TOKEN, scrubbing=False)
logger = logfire.with_tags('tariff_classification')

try:
    toolset = ComposioToolSet(entity_id=COMPOSIO_ENTITY_ID, api_key=COMPOSIO_API_KEY)
except Exception as e:
    logger.exception(f"Failed to initialize ComposioToolSet: {e}")
    raise

# Load HTS data
try:
    with open('files/htsdata.json', 'r', encoding='utf-8') as file:
        htsdata = json.load(file) 
        four_digit_codes, final_full_codes = AgentActions.wrangle_hts_data(htsdata)
except Exception as e:
    logger.exception(f"Failed to load HTS data: {e}")
    raise

# Load chapter headers
try:
    # FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    # app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    # headers, chapter_descs = AgentActions.get_hts_headers(app, headers_save_path='chapter_headers_final.txt', chapter_desc_save_path='chapter_desc.json')
    with open('files/chapter_headers_final.txt', 'r', encoding='utf-8') as file:
        headers = file.readlines()
    with open('files/chapter_desc.json', 'r', encoding='utf-8') as file:
        chapter_descs = json.load(file) 
except Exception as e:
    logger.exception(f"Failed to load chapter headers: {e}")
    raise

# Initialize LLM
try:
    llm = init_chat_model(
            model="gemini-2.0-flash",
            model_provider='google_genai',
            api_key=GOOGLE_API_KEY,
            temperature=0.0,
            configurable_fields="any"
        )
    
except Exception as e:
    logger.exception(f"Failed to initialize LLM: {e}")
    raise

    # Initialize agents
def initialize_agents(tag: str):
    logger = logfire.with_tags(tag)
    try:
        agent_actions = AgentActions(logger=logger, chapter_descs=chapter_descs, four_digit_codes=four_digit_codes, final_full_codes=final_full_codes, tariffy_org_id=TARIFFY_ORG_ID, tariffy_api_key=TARIFFY_API_KEY, simpleduty_api_key=SIMPLEDUTY_API_KEY)

        code_extractor = CodeExtractor(llm=llm, logger=logger)

        chapter_selector = ChapterSelector(llm=llm, logger=logger, chapters_list=headers, code_extractor=code_extractor)

        level_one_selector = LevelOneSelector(llm=llm, logger=logger, code_extractor=code_extractor, agent_actions=agent_actions)

        deep_selector = DeepSelector(llm=llm.with_config(config={"model":"gemini-2.0-flash-thinking-exp-01-21"}), logger=logger, code_extractor=code_extractor, agent_actions=agent_actions)

        final_selector = FinalSelector(llm=llm.with_config(config={"model":"gemini-2.5-flash-preview-04-17"}), logger=logger, agent_actions=agent_actions)

    except Exception as e:
        logger.exception(f"Failed to initialize agents: {e}")
        raise

    class State(TypedDict):
        responses: Annotated[list, add_messages]
        product_description: str
        chapters_list: list
        four_digit_code_list: list
        full_code_list: list
        final_codes: dict

    # Build the graph
    logger.debug("Building workflow graph...")
    try:
        graph_builder = StateGraph(State)

        graph_builder.add_node("chapter_selector", chapter_selector.select_chapters)
        graph_builder.add_edge(START, "chapter_selector")

        graph_builder.add_node("select_four_digit_codes", level_one_selector.select_four_digit_codes)
        graph_builder.add_edge("chapter_selector", "select_four_digit_codes")

        graph_builder.add_node("select_full_codes", deep_selector.select_full_codes)
        graph_builder.add_edge("select_four_digit_codes", "select_full_codes")

        graph_builder.add_node("select_final_codes", final_selector.select_final_codes)
        graph_builder.add_edge("select_full_codes", "select_final_codes")

        graph_builder.add_edge("select_final_codes", END)

        graph = graph_builder.compile()
        # Configure the graph for async execution
        graph_async = graph.with_config({"executor": "async"})
    except Exception as e:
        logger.exception(f"Failed to build or compile workflow: {e}")
        raise
    return graph_async, agent_actions, logger

# FastAPI app
api_app = FastAPI(title="Tariff Classification API", 
                  description="API for classifying products into the Harmonized Tariff Schedule (HTS) codes")

class IncomingRequest(BaseModel):
    id: str
    job_uuid: str
    created: int
    api_version: str
    type: str
    data: dict

@api_app.post("/classify")
async def classify_product(request: IncomingRequest):
    """
    Classify a product according to the Harmonized Tariff Schedule
    """
    
    try:
        # Extract data 
        requestor = request.data["caller"]["email"]
        invoice_number = request.data["value"]["General Information"]["Invoice Number"]
        items = request.data["value"]["Items"]

        # Initialize the graph and agents
        graph_async, agent_actions, logger = initialize_agents(tag=invoice_number)

        # Create a dataframe from the items
        df = pd.DataFrame(items)
        df = df.rename(columns={"Description": "description"})

        
        # Get all descriptions from the DataFrame
        descriptions = df["description"].tolist()
        logger.info(f"Received request: {invoice_number}. Classifying {len(descriptions)} items", product_descriptions=descriptions)
        
        # Create tasks for all product descriptions to run in parallel
        classification_tasks = [
            graph_async.ainvoke({"product_description": description}) 
            for description in descriptions
        ]

        tariffy_task = agent_actions.get_tariffy_codes(descriptions=descriptions)
        
        # Run all tasks in parallel and wait for all results
        results, tariffy_results = await asyncio.gather(
            asyncio.gather(*classification_tasks),  # Run classification tasks
            tariffy_task  # Run Tarrify API call
        )
        
        # Format the results into a structured response
        classification_results = []
        for result in results:
            classification_results.append({
                "description": result.get("product_description", ""),
                "most_likely_code": result["final_codes"].get("most_likely_code", ""),
                "most_likely_code_lower_rate_code": result["final_codes"].get("most_likely_lower_rate_code", ""),
            })
        #make classification results into a pandas dataframe and save locally
        classification_df = pd.DataFrame(classification_results)
        tariffy_df = pd.DataFrame(tariffy_results)
        tariffy_df['tariffy_hts_code'] = tariffy_df['tariffy_hts_code'].apply(lambda x: re.sub(r'\.', '', x))
        tariffy_df['tariffy_hts_code'] = tariffy_df['tariffy_hts_code'].apply(lambda x: f"{x[:4]}.{x[4:6]}.{x[6:8]}.{x[8:]}")

        final_df = pd.merge(df, classification_df, on="description", how="left")
        final_df = pd.merge(final_df, tariffy_df, on="description", how="left")

        logger.info(f"Getting duty rates for selected HTS codes")
        final_tasks = [
            agent_actions.get_rates_and_descs(
            origin=row['Country of Origin'],
            dest='US',
            code_one=row['most_likely_code'],
            code_two=row['most_likely_code_lower_rate_code'],
            code_three=row['tariffy_hts_code']
                )
                for _, row in final_df.iterrows()
            ]

        final_results = await asyncio.gather(*final_tasks)
        results_df = pd.DataFrame(final_results)
        final_df = pd.concat([final_df, results_df], axis=1)
        final_columns = [i for i in df.columns] + ['most_likely_code', 
                                                   "most_likely_code_desc",
                                                   'most_likely_code_duty_rate',
                                                   'most_likely_code_lower_rate_code',
                                                    'most_likely_code_lower_rate_desc',
                                                    'most_likely_code_lower_rate_duty_rate', 
                                                   'tariffy_hts_code', 
                                                   'tariffy_hts_code_desc',
                                                   'tariffy_hts_code_duty_rate']
        
        final_df = final_df[final_columns]
        
        final_df.to_csv("classification_results.csv", index=False)  

        # message = create_message_with_attachment(to=requestor,
                                                # subject=f"Classification results for {invoice_number}", 
                                                # msg_body="Hello, \n\n Please find your classification results attached. \n\n Thank you, \n Miller", 
                                                # file_path="classification_results.csv")
        # send_message(GMAIL_TOKEN, message)

        toolset.execute_action(
                    action=Action.GMAIL_SEND_EMAIL,
                    params={
                        "recipient_email": requestor,
                        "subject": f"Classification results for {invoice_number}",
                        "body": "Hello, \n\n Please find you classification results attached. \n\n Thank you, \n Miller",
                        "attachment": "classification_results.csv"
                    }
                )

        logger.info(f"Classification complete. Sending email to {requestor}")      

        
        return {
            "status": "success",
        }
    
    except Exception as e:
        logger.exception(f"Error processing classification request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during classification")

@api_app.get("/health")
async def health_check():
    """
    Check if the service is running.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    
    port = int(os.getenv("PORT", 8080))  # Use the PORT environment variable or default to 8080
    logger.info(f"Starting Tariff Classification API server on port {port}")
    uvicorn.run(api_app, host="0.0.0.0", port=port)