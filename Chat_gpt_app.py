from flask import Flask, request, jsonify
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os

os.environ["OPENAI_API_KEY"] = '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'

app = Flask(__name__)


def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 90
    chunk_size_limit = 1200

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=-10, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


@app.route('/chat', methods=['POST'])
def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

if __name__ == '__main__':
    dir_path = "C:\\CHAT_GPT_project\\files"
    index = construct_index(dir_path)
    app.run()
    iface = gr.Interface(fn=chatbot,
                         inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                         outputs="text",
                         title="Custom-trained AI Chatbot")

# # iface.launch(share=True)
# from flask import Flask, request, jsonify
# from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from langchain.chat_models import ChatOpenAI
# import os

# os.environ["OPENAI_API_KEY"] = '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'

# app = Flask(__name__)

# def construct_index(directory_path):
#     max_input_size = 4096
#     num_outputs = 2000
#     max_chunk_overlap = 90
#     chunk_size_limit = 1200

#     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=-10, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

#     documents = SimpleDirectoryReader(directory_path).load_data()

#     index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

#     index.save_to_disk('index.json')

#     return index

# def chatbot(input_text):
#     index = GPTSimpleVectorIndex.load_from_disk('index.json')
#     response = index.query(input_text, response_mode="compact")
#     return response.response

# @app.route('/chat', methods=['POST'])
# def chat():
#     input_text = request.json['text']
#     response = chatbot(input_text)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     dir_path = "C:\\CHAT_GPT_project\\files"
#     index = construct_index(dir_path)
#     app.run()


# from flask import Flask, request, jsonify
# from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from langchain.chat_models import ChatOpenAI
# import os

# os.environ["OPENAI_API_KEY"] = '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'

# app = Flask(__name__)

# def construct_index(directory_path):
#     max_input_size = 4096
#     num_outputs = 2000
#     max_chunk_overlap = 90
#     chunk_size_limit = 1200

#     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=-10, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

#     documents = SimpleDirectoryReader(directory_path).load_data()

#     index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

#     index.save_to_disk('index.json')

#     return index

# def chatbot(input_text):
#     index = GPTSimpleVectorIndex.load_from_disk('index.json')
#     response = index.query(input_text, response_mode="compact")
#     return response.response

# @app.route('/chat', methods=['POST'])
# def chat():
#     input_text = request.json['text']
#     response = chatbot(input_text)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     dir_path = "C:\\CHAT_GPT_project\\files"
#     index = construct_index(dir_path)
    # app.run()