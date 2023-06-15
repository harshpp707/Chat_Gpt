from flask import Flask, request, jsonify
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os
import traceback

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = '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'

# Define the global variables
index = None

UPLOAD_FOLDER = "C:\\CHAT_GPT_project\\files"
INDEX_FOLDER = "C:\\CHAT_GPT_project\\index"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["INDEX_FOLDER"] = INDEX_FOLDER

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 90
    chunk_size_limit = 1200

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=10, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk(os.path.join(app.config['INDEX_FOLDER'], 'index.json'))

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    dir_path = app.config['UPLOAD_FOLDER']
    index = construct_index(dir_path)
    return 'File uploaded successfully', 200

# @app.route('/', methods=['GET'])
# def default():
#     return "Flask API is running."

@app.route('/chat', methods=['GET','POST'])
def chat():
    try:
        input_text = request.json['text']
        response = chatbot(input_text)
        return jsonify({'response': response})
    except Exception as e:
        traceback.print_exc()
        return 'Internal Server Error', 500

if __name__ == '__main__':
    # dir_path = "C:\\CHAT_GPT_project\\files"
    # index = construct_index(dir_path)
    app.run()
    
