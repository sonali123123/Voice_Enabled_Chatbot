from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from langchain_ollama import ChatOllama
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from gtts import gTTS
import whisper
import os
import time
import aiofiles
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Whisper ASR model
try:
    whisper_model = whisper.load_model("base")  # Faster model for lower latency
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    raise e

# Load LLM (Ollama)
try:
    llm = ChatOllama(model="phi3:latest", temperature=0.5)
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    raise e

# Initialize TTS and directory paths
RESPONSE_AUDIO_DIR = "responses_audio"
UPLOAD_AUDIO_DIR = "upload_audio"
os.makedirs(RESPONSE_AUDIO_DIR, exist_ok=True)
os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)

# Embeddings and Vectorstore for context-aware responses
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embedding_function, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(k=3)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


test_qa_pr_sh = """
### System:
You are a real-time, voice-enabled AI assistant designed for fast and accurate interaction. \
**Response Limit**: Maximum 20 words  \
Prioritize direct, simple, and relevant answers. 

**IMPORTANT**: 
- If the answer is not known or lacks context, respond with: "Sorry, I don't know about this."  
- Use concise, easy-to-understand language.
- Minimize latency and optimize for real-time speech responses.
 
{context}"""





qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", test_qa_pr_sh),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


 #Chat history store for maintaining session histories
store = {}



rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.post("/chat")
async def process_audio(audio_file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        audio_path = os.path.join(UPLOAD_AUDIO_DIR, f"audio_{int(time.time())}.mp3")
        async with aiofiles.open(audio_path, "wb") as f:
            await f.write(await audio_file.read())
        
        logging.info(f"Audio file saved at {audio_path}")
        
        # Transcribe audio
        transcribed_text = transcribe_audio(audio_path)
        logging.info(f"Transcribed text: {transcribed_text}")

        # Generate LLM response
        response_text = generate_response(transcribed_text)
        logging.info(f"Generated response: {response_text}")

        if not response_text:
            raise HTTPException(status_code=500, detail="Failed to generate response from LLM.")

        # Convert response to audio (TTS)
        audio_response_path = text_to_speech(response_text)
        logging.info(f"Response audio saved at {audio_response_path}")

        # Clean up uploaded file
        background_tasks.add_task(os.remove, audio_path)
        
        return FileResponse(audio_response_path, media_type="audio/mpeg", filename="response.mp3")
    
    except Exception as e:
        logging.error(f"Error during audio processing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Transcription (ASR)
def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, language='en')
        return result.get("text", "").strip()
    except Exception as e:
        logging.error(f"ASR transcription failed: {e}")
        raise e
        
## Generate response from LLM
def generate_response(query_text, session_id="default_session"):
    """Generate a response based on the query."""
    try:

        print(f"Invoking conversational_rag_chain with session_id: {session_id}")

        # Ensure the session ID is passed correctly
        response = conversational_rag_chain.invoke(
            {'input': query_text},
            config={'session_id': session_id}  # Correct the dictionary key
        )["answer"]

        # Normalize the response text
        response_cleaned = response.replace("\n", "")
        print(f"Generated response: {response_cleaned}")
        return response
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return {"error": str(e)}


# Text-to-Speech (TTS)
def text_to_speech(response_text):
    try:
        audio_path = os.path.join(RESPONSE_AUDIO_DIR, f"response_{int(time.time())}.mp3")
        gTTS(text=response_text, lang="en", slow=False).save(audio_path)
        return audio_path
    except Exception as e:
        logging.error(f"TTS conversion failed: {e}")
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
