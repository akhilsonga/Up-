import streamlit as st
from dataclasses import dataclass
import sys
import os
sys.path.append('D:/CodePhilly')
import back
import base64
@st.cache_data
def process_chat_input(chat_message,tool):
    # response = back.start(chat_message,tool)
    if tool=="General":
        response=back.O_LLM_gemini(chat_message)
    elif tool=="Email Manager":
        response=back.write_email(chat_message)
    elif tool=="Web Surfer":
        response=back.internet(chat_message)
    elif tool=="Report Analyst":
        response=back.generate_report(chat_message)
    elif tool=='Inventory Manager':
        response=back.Inventory_Management_Handler(chat_message)
    elif tool=='Catalogue Business Analyst':
        response=back.Report_catalogue(chat_message)
        print("response: ",response)
        print(type(response))
    return response

@dataclass
class Message:
    actor: str
    payload: str

def main():
    # a,b,c,d,e,g=back.google_sheets_access()
    st.sidebar.title('Up!')
    btn=st.sidebar.selectbox('Choose what you want to work with',['General','Web Surfer','Inventory Manager', 'Report Analyst', 'Email Manager','Catalogue Business Analyst'])
    print(btn)
    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.actor).write(msg.payload)
    prompt = st.chat_input("Ask a question!")
    
    if prompt:
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        st.chat_message(USER).write(prompt)
        response = process_chat_input(prompt,btn)
        if len(response)==3 and btn=='Inventory Manager':
            st.write('Old Data Frame')
            st.dataframe(response[0])
            st.write('Latest Data Frame')
            st.dataframe(response[1])
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response[2])
        elif btn=='Report Analyst' and response=='Report Generated':
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response)
            with open('Report_vendor_gemini.pdf', "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
                st.markdown(pdf_display, unsafe_allow_html=True)
            #btn=='Catalogue Business Analyst'
        elif btn=='Catalogue Business Analyst':
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response)
        else:
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response)
if __name__ == "__main__":
    # back.create_data()
    main()