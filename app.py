from typing import Callable, Optional

import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Zilliz
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI

chain: Optional[Callable] = None


def web_loader(url_list, openai_key, zilliz_uri, user, password):
    if not url_list:
        return "please enter url list"
    loader = WebBaseLoader(url_list.split())
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="ada", openai_api_key=openai_key)

    docsearch = Zilliz.from_documents(
        docs,
        embedding=embeddings,
        connection_args={
            "uri": zilliz_uri,
            "user": user,
            "password": password,
            "secure": True,
        },
    )

    global chain
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0, openai_api_key=openai_key),
        chain_type="map_reduce",
        retriever=docsearch.as_retriever(),
    )
    return "success to load data"


def query(question):
    global chain
    # "What is milvus?"
    if not chain:
        return "please load the data first"
    return chain(inputs={"question": question}, return_only_outputs=True).get(
        "answer", "fail to get answer"
    )


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown(
            """
        <h1><center>Langchain And Zilliz Cloud Example</center></h1>
        This is how to use Zilliz Cloud as vector store in LangChain. 
        The purpose of this example is to allow you to input multiple URLs (separated by newlines) and then ask questions about the content of the corresponding web pages.
        
        ## 📋 Prerequisite:
        1. 🔑 To obtain an OpenAI key, please visit https://platform.openai.com/account/api-keys.
        2. 💻 Create a Zilliz Cloud account to get free credits for usage by visiting https://cloud.zilliz.com.
        3. 🗄️ Create a database in Zilliz Cloud.
        
        ## 📝 Steps for usage:
        1. 🖋️ Fill in the url list input box with multiple URLs.
        2. 🔑 Fill in the OpenAI API key in the openai api key input box.
        3. 🌩️ Fill in the Zilliz Cloud connection parameters, including the connection URL, corresponding username, and password.
        4. 🚀 Click the Load Data button to load the data. When the load status text box prompts that the data has been successfully loaded, proceed to the next step.
        5. ❓ In the question input box, enter the relevant question about the web page.
        6. 🔍 Click the Generate button to search for the answer to the question. The final answer will be displayed in the question answer text box.
        """
        )
        url_list_text = gr.Textbox(
            label="url list",
            lines=3,
            placeholder="https://milvus.io/docs/overview.md",
        )
        openai_key_text = gr.Textbox(label="openai api key", type="password", placeholder="sk-******")
        with gr.Row():
            zilliz_uri_text = gr.Textbox(
                label="zilliz cloud uri",
                placeholder="https://<instance-id>.<cloud-region-id>.vectordb.zillizcloud.com:<port>",
            )
            user_text = gr.Textbox(label="username", placeholder="db_admin")
            password_text = gr.Textbox(
                label="password", type="password", placeholder="******"
            )
        loader_output = gr.Textbox(label="load status")
        loader_btn = gr.Button("Load Data")
        loader_btn.click(
            fn=web_loader,
            inputs=[
                url_list_text,
                openai_key_text,
                zilliz_uri_text,
                user_text,
                password_text,
            ],
            outputs=loader_output,
            api_name="web_load",
        )

        question_text = gr.Textbox(
            label="question",
            lines=3,
            placeholder="What is milvus?",
        )
        query_output = gr.Textbox(label="question answer", lines=3)
        query_btn = gr.Button("Generate")
        query_btn.click(
            fn=query,
            inputs=[question_text],
            outputs=query_output,
            api_name="generate_answer",
        )

        demo.queue().launch(server_name="0.0.0.0", share=False)
