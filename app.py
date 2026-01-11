import os
os.environ["USER_AGENT"] = "MyAgent/0.1"

import concurrent
import streamlit as st
import json
from openai import OpenAI
from openai.types.responses import ResponseOutputMessage
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


history = StreamlitChatMessageHistory()

if "final_report" not in st.session_state:
    st.session_state.final_report = None

with st.sidebar:
    OPEN_API_KEY = st.text_input(label="OPEN_API_KEY")
    
    model = st.selectbox(
        "Model",
        ["gpt-5-nano", "gpt-4o-mini"]
    )
    
    st.write("https://github.com/animasana/assignment10/blob/main/app.py")

    if st.button("History Clear"):
        history.clear()    


st.set_page_config(
    page_title="Assitant GPT",
    page_icon="🧠"
)

st.title("🧠 Assignment10 - AI Assistant")


if not OPEN_API_KEY:
    st.error("Input Your Own OPENAI_API_ KEY!!!")
    history.clear()
    st.stop()


openai = OpenAI(api_key=OPEN_API_KEY)


def wikipedia_search(inputs) -> str:    
    query = inputs["query"]
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)


def duckduckgo_search(inputs) -> str:    
    query = inputs["query"]
    ddgs = DuckDuckGoSearchResults()
    return ddgs.run(query)


def scrape_website(inputs) -> str:    
    url = inputs["url"]
    loader = WebBaseLoader([url])
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return text[:5000]    


tools = [
    {
        "type": "function",
        "name": "wikipedia_search",
        "description": (
            "Given a query (i.e Research the Apple Company), Search in Wikipedia and return search results."
            "Use tools to research. Do not rely on prior knowledge."
        ),        
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query about a topic",
                }
            },
            "required": ["query"],
        },        
    },
    {
        "type": "function",
        "name": "duckduckgo_search",
        "description": (
            "Given a query (i.e Research the Apple Company), Search in DuckDuckGo and return search results."
            "Use tools to research. Do not rely on prior knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query about a topic",
                },
            },
            "required": ["query"],
        },        
    },
    {
        "type": "function",
        "name": "scrape_website",
        "description": """
            When duckduckgo_search found a website url,
            Enter the website and Use this tool to extract its contents.
        """,        
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "the url with found in searching duckduckgo",
                },
            },
            "required": ["url"],
        },        
    },    
]


tool_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "scrape_website": scrape_website,        
}


def paint_history():
    for msg in history.messages:
        st.chat_message(msg.type).markdown(msg.content)


def send_user_message(message):
    st.chat_message("user").write(message)
    history.add_user_message(message)


def send_ai_message(message):
    st.chat_message("assistant").markdown(message)
    history.add_ai_message(message)    


def build_messages_from_history():
    messages = []
    for msg in history.messages:        
        if msg.type == "human":
            role = "user"
        elif msg.type == "ai":
            role = "assistant"
        elif msg.type == "system":
            role = "system"
        else:
            continue 

        messages.append({
            "role": role,
            "content": msg.content,
        })
    return messages


st.chat_message("ai").write("I'm ready! Ask away!")
paint_history()
user_input = st.chat_input("Ask something...")
response = None
if user_input:
    send_user_message(user_input)    
    response = None
    is_research = user_input.lower().startswith("research")
    if not is_research:
        messages = build_messages_from_history()
        messages.append({
            "role": "user",
            "content": user_input
        })

        response = openai.responses.create(
            model=model,
            input=messages,
        )
    else:
        with st.spinner("Waiting a response..."):
            SYSTEM_PROMPT = {
                "role": "system",
                "content": """
                    You are a research assistant.
                    
                    Rules:
                    1. Use tools to gather information. Do NOT rely on prior knowledge.
                    2. Use wikipedia_search first in Wikipedia.
                    3. Use duckduckgo_search in DuckDuckGo.
                    4. If duckduckgo_search found a url, Scrape at most ONE website.
                    5. Do NOT repeat the same search query.
                    6. After completing research, respond with the final report directly.
                    7. Do NOT call any tool after the report.                    
                """
            }
            response = openai.responses.create(
                model=model,
                input=[
                    SYSTEM_PROMPT,
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
                tools=tools,                   
            )            
            
            MAX_TOOL_LOOPS = 5
            loop_count = 0
            while loop_count < MAX_TOOL_LOOPS:
                loop_count += 1
                tool_calls = [
                    item for item in response.output if item.type == "function_call"
                ]
                if tool_calls:
                    with st.expander("🔧 Tool calls"):
                        st.markdown(
                            [
                                {
                                    "tool": call.name,
                                    "args": call.arguments,
                                } 
                                for call in tool_calls
                            ]
                        )
                else:
                    break

                MAX_SCRAPES = 1
                scrape_count = 0
                allowed_calls = []
                for call in tool_calls:                        
                    if call.name == "scrape_website":
                        if scrape_count >= MAX_SCRAPES:
                            continue
                        scrape_count += 1                        
                    allowed_calls.append(call)                 

                tool_messages = []                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_map = {}
                    for call in allowed_calls:                        
                        fn = tool_map[call.name]
                        args = json.loads(call.arguments)
                        
                        if call.name in ("wikipedia_search", "duckduckgo_search"):
                            future = executor.submit(fn, args)
                            future_map[future] = call
                        else:
                            result = fn(args)
                            tool_messages.append({
                                "type": "function_call_output",
                                "call_id": call.call_id,
                                "output": result,
                            })

                    for future in concurrent.futures.as_completed(future_map):
                        call = future_map[future]
                        result = future.result()
                        
                        tool_messages.append({
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": result,
                        })                    

                response = openai.responses.create(
                    model=model,
                    previous_response_id=response.id,
                    input=tool_messages,
                    tools=tools,        
                )
    
for item in response.output:
    if isinstance(item, ResponseOutputMessage):
        assistant_text = item.content[0].text
        send_ai_message(assistant_text)
        if assistant_text and is_research:
            st.download_button(
                label="Download Report",
                data=assistant_text,
                file_name="research_report.txt",
                mime="text/plain",
            )
                
            

