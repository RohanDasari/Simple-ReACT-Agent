import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Load env variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

def create_agent_executor():
    llm = ChatGroq(model_name="Gemma2-9b-It")  # Or any other Groq-supported model

    google_search = GoogleSerperAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=google_search.run,
            description="Useful for answering with search.",
        )
    ]

    template = '''Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor

# Streamlit UI
st.set_page_config(page_title="Live Agent Step-by-Step", layout="wide")
st.title("🧠 LLM Agent Step-by-Step Reasoning")

query = st.text_input("Ask a question")

if st.button("Submit") and query:
    st.info("Agent is working. This may take a few seconds...")
    with st.spinner("Thinking..."):
        try:
            agent_executor = create_agent_executor()
            response = agent_executor.invoke({"input": query})

            steps = response.get("intermediate_steps", [])
            for idx, (action, observation) in enumerate(steps):
                st.markdown(f"### 🔁 Step {idx+1}")
                st.markdown(f"**🧠 Thought:**\n```\n{action.log.strip()}```")
                st.markdown(f"**🔧 Action:** `{action.tool}`")
                st.markdown(f"**📥 Action Input:** `{action.tool_input}`")
                st.markdown(f"**👀 Observation:**\n```\n{observation.strip()}```")

            st.success(f"✅ **Final Answer:** {response['output']}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
