import streamlit as st
from streamlit_chat import message
from oneke_wrapper import OneKEWrapper
import json
st.set_page_config(
    page_title="Data2text - Demo",
    page_icon=":robot:"
)


st.header("Data2text - Demo")
# st.markdown("[Github](https://github.com/ai-yash/st-chat)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def load_model():
    print("---------------load_model--------------------")
    model_path = "./cache/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5"
    lora_path = "./lora/baichuan7B-data2text-continue"
    device = "cuda:0"
    oneke_wrapper = OneKEWrapper(model_path, lora_path, device, False)
    return oneke_wrapper
    
if 'model' not in st.session_state:
    print("---------------session_state------------------")
    st.session_state['model'] = load_model()
    # oneke_wrapper = 

def query(payload):
    text = payload["inputs"]["text"]
    response = st.session_state['model'].inference([text])
    return response

def get_text():
    input_text = st.text_input("You: ", key="input", placeholder = "Please input the data in JSON format.")
    return input_text


user_input = get_text()

if user_input:
    user_input_json = json.loads(user_input)
    print(type(user_input))
    output = query({
        "inputs": {
            "past_user_inputs": st.session_state.past,
            "generated_responses": st.session_state.generated,
            "text": user_input_json,
        },"parameters": {"repetition_penalty": 1.33},
    })
    output_string = json.dumps(output, ensure_ascii=False)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output_string)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')