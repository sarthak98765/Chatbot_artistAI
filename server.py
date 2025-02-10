from flask import Flask, render_template, request, jsonify
import requests
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

app = Flask(__name__)

# Initialize LlamaIndex embedding model
embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v1")
Settings.embed_model = embedding_model
# Load and index documents from the 'data/' directory
def load_and_index_documents(directory="data"):
    reader = SimpleDirectoryReader(input_dir=directory)
    documents = reader.load_data()

    # service_context = ServiceContext.from_defaults(embed_model=embedding_model)
    index = VectorStoreIndex.from_documents(documents)

    return index

# Create index from "data.txt"
index = load_and_index_documents()
retriever = index.as_retriever()

# Helper function to load text from a file
def load_text_file(file_name):
    try:
        with open(file_name, "r") as file:
            return file.read().lower()
    except Exception as e:
        print(f"⚠ Error loading text file '{file_name}': {e}")
        return ""

# Helper function to append chat logs to a file
# def append_chat_to_file(user_message, bot_response):
#     try:
#         with open("chat_log.txt", "a") as file:
#             file.write(f"User: {user_message}\n")
#             file.write(f"Bot: {bot_response}\n")
#             file.write("-" * 50 + "\n")  # Separator for readability
#     except Exception as e:
#         print(f"⚠ Error writing chat to file: {e}")

# Helper function to find location descriptions in the text file
def find_location_description(location_name, data_text):
    location_name_lower = location_name.lower()
    if location_name_lower in data_text:
        start_idx = data_text.find(location_name_lower)
        end_idx = data_text.find("\n", start_idx + len(location_name_lower))
        return data_text[start_idx:end_idx].strip()
    return None

# Helper function to retrieve relevant information using RAG
def retrieve_relevant_text(query):
    retrieved_docs = retriever.retrieve(query)
    return "\n".join([doc.text for doc in retrieved_docs[:3]])  # Retrieve top 3 results

# Chat logic (MODIFIED TO INCLUDE RAG)
def chat_with_bot(user_message, data_text, last_location=None):
    user_message_lower = user_message.lower()

    # Retrieve relevant context using LlamaIndex RAG
    retrieved_context = retrieve_relevant_text(user_message)

    if user_message_lower.startswith("artist ai"):
        command = user_message[len("artist ai"):].strip()
        command_lower = command.lower()
        system_message = (
            "You are a creative AI assistant for game artists specializing in pixel art. "
            "You help design locations and buildings with detailed visual descriptions, "
            "including colors, textures, architecture, and artistic composition. "
            "Focus on making your answers detailed, formatted, and tailored for pixel art environments."
        )

        if command_lower.startswith("describe "):
            location_name = command[9:].strip()
            location_details = find_location_description(location_name, data_text)

            if location_details:
                prompt = f"""
                Strictly describe the location '{location_name}' as it is in the document.
                Provide a well-organized format with these sections:
                1. *Location Overview*: <br>
                   A brief summary of the location, including its purpose and significance. <br><br>
                   
                2. *Visual Details*: <br>
                   Describe the environment, lighting, colors, and overall atmosphere. Mention how these details would look in a pixel art style, including suggestions for textures and palette choices. <br><br>
                   
                3. *Key Areas and Features*: <br>
                   Highlight the key landmarks, streets, or zones with sizes measured in tiles (e.g., 4x4 tiles, 8x8 tiles). <br><br>
                   
                4. *Building Styles*: <br>
                   Describe the architecture and materials used for buildings. Include guidance on roof shapes, window placements, and how to pixelate these details effectively. <br><br>
                   
                5. *Layout and Composition*: <br>
                   Explain how the location is arranged spatially. Provide ideas for creating depth and perspective in pixel art. <br><br>

                6. *population*: <br>
                   Explain how the location population like how much npc can be used types of npc. Provide ideas for creating depth and perspective in pixel art. <br><br>
                """
                last_location = location_name  # Update last location
            else:
                prompt = f"""
                Strictly describe the location '{location_name}' as it is in the document.
                Provide a well-organized format with these sections:
                1. *Location Overview*: <br>
                   A brief summary of the location, including its purpose and significance. <br><br>
                   
                2. *Visual Details*: <br>
                   Describe the environment, lighting, colors, and overall atmosphere. Mention how these details would look in a pixel art style, including suggestions for textures and palette choices. <br><br>
                   
                3. *Key Areas and Features*: <br>
                   Highlight the key landmarks, streets, or zones with sizes measured in tiles (e.g., 4x4 tiles, 8x8 tiles). <br><br>
                   
                4. *Building Styles*: <br>
                   Describe the architecture and materials used for buildings. Include guidance on roof shapes, window placements, and how to pixelate these details effectively. <br><br>
                   
                5. *Layout and Composition*: <br>
                   Explain how the location is arranged spatially. Provide ideas for creating depth and perspective in pixel art. <br><br>

                6. *population*: <br>
                   Explain how the location population like how much npc can be used types of npc. Provide ideas for creating depth and perspective in pixel art. <br><br>
                """
                last_location = None  # Reset last location
        elif last_location and command_lower in {"population", "population density", "how many people can live here?"}:
            prompt = f"Estimate the population density for the '{last_location}' area based on its size and typical characteristics."
        else:
            prompt = command
    else:
        system_message = "You are a helpful assistant."
        prompt = user_message

    # Enhance prompt with retrieved knowledge
    if retrieved_context:
        prompt = f"Relevant Information:\n{retrieved_context}\n\nUser Query:\n{prompt}"

    # API Configuration (Your existing OpenAI-like setup)
    api_key = "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"  # Replace with your actual API key
    url = "https://api.groq.com/openai/v1/chat/completions"
    model_id = "llama3-8b-8192"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code == 200:
            chatbot_response = response.json()
            bot_message = chatbot_response["choices"][0]["message"]["content"]
            # append_chat_to_file(user_message, bot_message)  # Save to chat log
            return bot_message, last_location
        else:
            return "An error occurred while processing your request.", last_location
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again.", last_location
    except Exception as e:
        return f"An unexpected error occurred: {e}", last_location
def append_to_file(text):
    try:
        with open("data/data.txt", "a") as file:
            file.write(f"### New Entry ###\n{text}\n\n")  # Add formatting
    except Exception as e:
        print(f"⚠ Error writing to file: {e}")
        return False
    return True
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    global last_location  # Use the global last_location variable
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "⚠ Please enter a message."})
    bot_response, last_location = chat_with_bot(user_message, data_text, last_location)
    return jsonify({"response": bot_response})
@app.route("/add-to-file", methods=["POST"])
def add_to_file():
    text_to_add = request.json.get("text", "")
    if not text_to_add:
        return jsonify({"message": "⚠ No text provided to add."}), 400

    success = append_to_file(text_to_add)
    if success:
        return jsonify({"message": "✅ Text successfully added to the file."})
    else:
        return jsonify({"message": "⚠ Failed to add text to the file."}), 500


if __name__ == "__main__":
    # Load data from the file
    data_text = load_text_file("data/data.txt")
    last_location = None  # Initialize last location as a global variable
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port
    app.run(host="0.0.0.0", port=port, debug=False)
