from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from operator import itemgetter

st.set_page_config(
    page_title="CloudflareGPT",
    page_icon="☁",
)

st.markdown(
    """
    # Cloudflare GPT

    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    st.write("https://github.com/seungchan0324/FullStackGPT_challenge_SiteGPT")
    key = st.text_input("please give your api key")

if key:
    llm = ChatOpenAI(
        temperature=0.1,
        api_key=key,
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []


answers_prompt = ChatPromptTemplate.from_template(
    """
Using ONLY the following context answer the user's question.
If you can't just say you don't know, don't make anything up.

Then, give a score to the answer between 0 and 5.
The score should be high if the answer is related to the user's question, and low otherwise.
If there is no relevant content, the score is 0.
Always provide scores with your answers
Context: {context}

Examples:

Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know
Score: 0

History: {history}

Your turn!
Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    history = inputs["history"]
    answer_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answer_chain.invoke(
                    {
                        "context": doc.page_content,
                        "question": question,
                        "history": history,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter().from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=key))
    return vector_store.as_retriever()


def load_message():
    converted_message = []
    for message in st.session_state["messages"]:
        if message["role"] == "human":
            converted_message.append({"human": message["message"]})
        elif message["role"] == "ai":
            converted_message.append({"ai": message["message"]})
    return converted_message


def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=False):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"])


cloudflare_sitemap_url = "https://developers.cloudflare.com//sitemap-0.xml"

if key:
    retriever = load_website(cloudflare_sitemap_url)

    # 메세지 보내는거
    send_message("What can I do for you?", "AI")
    # 메세지 뿌리는거
    paint_history()

    history = load_message()

    message = st.chat_input("Ask a question to the Cloudflare GPT.")
    if message:
        send_message(message, "human", True)
        chain = (
            {
                "docs": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.spinner("Asking to the GPT..."):
            result = chain.invoke({"question": message, "history": history})
        send_message(result.content, "ai", True)
else:
    st.info("You must enter the API key through the sidebar.")
