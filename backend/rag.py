import io
import uuid

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

SYSTEM_PROMPT = """
You are an AI study assistant.

Your primary goal is to help the user UNDERSTAND the information contained in the uploaded document.

Use ONLY the information provided in the retrieved document context to answer factual questions.

====================
GROUNDING RULES
====================

1. Do not use information that is not present in the retrieved document context.
2. Do not invent facts, definitions, qualifications, examples, or explanations.
3. If the answer cannot be determined from the provided context, respond briefly:

   "I couldn't find information about this in the uploaded document."

   Optionally add one short sentence explaining what related information was found, if any.

4. Do not claim that something exists in the document unless it is supported by the retrieved context.
5. Do not quote large sections of the document verbatim.

====================
TEACHING STYLE
====================

Do not simply copy or paraphrase the document sentence by sentence.

Instead:

1. Identify the important information from the retrieved context.
2. Explain it clearly in your own words.
3. Organize the explanation so it is easy to understand.
4. Use simple examples ONLY when they help understanding AND the example does not introduce unsupported factual claims.
5. If an example is hypothetical, clearly label it as "Example".
6. For complex topics, explain the concept step by step.
7. Define technical terms in simple language when useful.

====================
RESPONSE FORMAT
====================

Always use clean Markdown formatting.

Use:

- Short paragraphs
- Headings when useful
- Bullet points when listing information
- Numbered lists for steps or processes
- **Bold text** for important terms

Do NOT use HTML tags such as <br>.

Do NOT create a table unless the user explicitly asks for a comparison or a table.

For normal conceptual questions, prefer this structure when appropriate:

## Answer
A clear direct answer.

## Explanation
Explain the concept in simple language.

## Example
Provide a simple hypothetical example if it helps understanding.

Keep answers focused and avoid unnecessary repetition.

====================
RETRIEVED DOCUMENT CONTEXT
====================

{context}
"""

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """
Given the chat history and the latest user question, rewrite the latest question
as a standalone question that can be understood without the chat history.

Rules:
- Preserve the original meaning.
- Resolve references such as "it", "this", "that", or "they" using the chat history.
- Do NOT answer the question.
- Do NOT add information that is not present in the conversation.
- If the latest question is already standalone, return it unchanged.

Return only the standalone question.
"""

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5


class RAGService:
    def __init__(self):
        print("Loading FastEmbed embeddings...")

        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("Embeddings loaded successfully.")

        print("Initializing Groq LLM...")

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b"
        )

        print("Groq LLM initialized successfully.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.sessions = {}

        print("RAGService initialization complete.")

    def process_pdf(self, filename: str, content: bytes) -> dict:
        reader = PdfReader(io.BytesIO(content))
        if len(reader.pages) == 0:
            raise ValueError("The PDF contains no pages.")

        docs = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_number},
                    )
                )

        if not docs:
            raise ValueError("No extractable text found in this PDF.")

        chunks = self.text_splitter.split_documents(docs)
        session_id = str(uuid.uuid4())

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=f"session_{session_id}",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

        chain = self._build_chain(retriever)
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        self.sessions[session_id] = {
            "chain": conversational_chain,
            "vectorstore": vectorstore,
            "filename": filename,
        }

        return {
            "session_id": session_id,
            "filename": filename,
            "pages": len(reader.pages),
            "chunks": len(chunks),
        }

    def ask(self, session_id: str, question: str) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("Unknown session. Please upload a document first.")

        response = session["chain"].invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )

        sources = []
        seen = set()
        for doc in response.get("context", []):
            meta = doc.metadata
            key = (meta.get("source"), meta.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({"source": meta.get("source"), "page": meta.get("page")})

        return {"answer": response["answer"], "sources": sources}

    def _build_chain(self, retriever):
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            retriever,
            ChatPromptTemplate.from_messages(
                [
                    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            ),
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)