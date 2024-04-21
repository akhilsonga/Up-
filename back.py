from duckduckgo_search import DDGS
import re
import os
import pyttsx3
from openai import OpenAI
import json
from datetime import datetime
from groq import Groq
from google.oauth2.service_account import Credentials
import gspread
import pandas as pd
if os.getenv("GROQ_API_KEY") is None:
    os.environ["GROQ_API_KEY"] = ''



def read_entire_gspread_sheet_to_pandas(credentials_file, sheet_id):

    scopes = [""]
    creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
    client = gspread.authorize(creds)

    try:
        sheet = client.open_by_key(sheet_id).sheet1  # Access the first sheet

        # Read all values efficiently (consider spreadsheet size for optimization)
        values = sheet.get_all_values()
        if not values:
            return pd.DataFrame()  # Return an empty DataFrame if no data

        data = pd.DataFrame(values[1:], columns=values[0])  # Skip header row
        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None to indicate an error

def google_sheets_access():
    credentials_file = r"C:\Users\akhil\Downloads\cred_google_sheet.json"
    sheet_id = ""

    df = read_entire_gspread_sheet_to_pandas(credentials_file, sheet_id)
    # display(df)

    if df is not None:
        
        df.columns = ["time", "model", "agents", "rag", "email", "send_copy","company", "description"]
        # df = df_replace_model_names(df)
        # display(df)
        recent_df = df.tail(1)
        # display(recent_df)
        model_req = str(recent_df["model"].values[0])
        tools_req = str(recent_df["agents"].values[0])
        email_req = str(recent_df["email"].values[0])
        company = str(recent_df["company"].values[0])
        description = str(recent_df["description"].values[0])
        print(company)
        return model_req, tools_req, email_req, company, description
    else:
        print("Error reading the Google Sheet. Check credentials or sheet ID.")

a,b,c,d,e=google_sheets_access()
def get_todays_date():
    date = datetime.today().date()
    dt = "Today's date is:" + str(date)
    return dt

def O_LLM_GroQ(query):
    print("Groq!")
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="mixtral-8x7b-32768",
        #model="gemma-7b-it",
        temperature = 0,
    )

    response = chat_completion.choices[0].message.content
    return response

import google.generativeai as genai

#Gemini 1.0 pro, good not great need to test on 1.5 pro in google hackathon
def O_LLM_gemini(query):
    #global Gemini_API
    Gemini_API = ""
    model = genai.GenerativeModel('gemini-pro')
    genai.configure(api_key=Gemini_API)
    response = model.generate_content(query)
    resp = response.text
   # print(response.text)
    return resp

def O_LLM_openai(query):
    client = OpenAI(api_key='')
    messages = [{"role": "user","content": query}]
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",messages=messages)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None
        
def extract_info(texts):
    tools = {}
    for text in texts:
        # Extract tool using regular expression
        tool = re.findall(r'^\w+', text)[0]

        # Extract input using regular expression
        inp = re.findall(r'\[(.*?)\]', text)[0]

        # Add tool and input to the dictionary
        tools[tool] = inp
    return tools

def duck_go(Keyword):
    print(Keyword)
    results = DDGS().text(Keyword, max_results=12)
    bodies = [item['body'] for item in results]
    paragraph = ' '.join(bodies)
    return paragraph

    
import subprocess

def execute_python(code):
    #print("Code recieved for execution Terminal: ",code)
    result = subprocess.run(["python", "-c", code], capture_output=True, text=True)
    err = 0
    # Check if there's an error
    if result.returncode != 0:
        print("Error Found")
        err = 1
        return result.stderr, err
    else:
        
        output = result.stdout
        return output, err
        
#------------------------------------------------------------------------------------------------
def extract_text(input_string, option):
    if option == 1:
        pattern = r'\```Python(.*?)\```'
        matches = re.search(pattern, input_string, re.DOTALL)
        if matches:
            return matches.group(1).strip()
        else:
            return None
    else:
        pattern = r'\```(.*?)\```'
        matches = re.search(pattern, input_string, re.DOTALL)
        if matches:
            return matches.group(1).strip()
        else:
            return None
#------------------------------------------------------------------------------------------------
def check_substring(main_string, substring):

    if substring.lower() in main_string.lower():
        return True
    else:
        return False
#------------------------------------------------------------------------------------------------
Error_Counter = 0

def code_processing(answer):
    #answer = O_LLM(query)
    global Error_Counter
    print("Preprocessing code for execution")
    main_string = answer
    substring = "```Python"
    substring_sub = "```"
    print("\n\n")
    if check_substring(main_string, substring_sub):
        #print("```, FOUND PREPROCESSING... ")
        
        if check_substring(main_string, substring):
            #print("```python, FOUND PREPROCESSING... ")
            input_string =  answer
            extracted_text = extract_text(input_string, 1)
            
            if extracted_text:
                answer = extracted_text
                #print("Extracted Text: \n", answer)
                code = answer
            else:
                #print("No text found between ``` and ```.")
                code = answer
        else:
            print("")
            if check_substring(main_string, substring_sub):
                print("```Python, FOUND PREPROCESSING... ")
                input_string =  answer
                extracted_text = extract_text(input_string, 0)

                if extracted_text:
                    answer = extracted_text
                    #print("Extracted Text: \n", answer)
                    code = answer
                else:
                    print("No text found between ``` and ```.")
                    code = answer
            
    else:
        print("```Python ,NOT FOUND")
        code = answer
    print("Code Extracted: ",code)
    code_to_execute = code    
    result, err = execute_python(code_to_execute)
    return result


def Voice(voice_response):
    text = voice_response
    engine = pyttsx3.init()
    engine.setProperty('rate', 190)    # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1
    engine.say(text)
    engine.runAndWait()
    return "Speaking completed"

def handle_request(data, thought):
    #
    print('Data from handle: ',data)
    if "Search" in data:
        output = duck_go(data["Search"])
        param = data["Search"]
#         print("In DuckDuckGo Search Question:",thought)
        print("Duck Duck Go :",output)
        print("\n\n\n")
        prompt = f"consider the text based on the reference, if you found no connection between the Question and Reference, find the most relevant answer for the question \n Question: {thought}\n\n\n Reference: {output}"
        output = O_LLM_gemini(prompt)
        return output
    elif "Python" in data:
        output = code_processing(data["Python"])
        return output
    elif "Voice" in data:
        output = Voice(data["Voice"])
        return output
    else:
        print("Invalid Tool key. Please use appropiate tool key.")

        
def convert_list_to_dict(data):
    result = {}
    for item in data:
        try:
            key, value = item.split('[', 1)
            value = value.rsplit(']', 1)[0].strip()  # Get text from beginning to last ']'
            if value:  # Check if value is not empty (null)
                result[key.strip()] = value
        except ValueError:
            continue  # Skip to the next iteration if splitting fails
    return result
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------


def extract_actions(text):
    pattern = r'[Aa]ction(?:[:\s\d]+)?\s*([^\[\]]+\[[^\[\]]+\])'
    matches = re.findall(pattern, text)
    return matches

def token_count(text):
    tokens = text.split()
    num_tokens = len(tokens)
    return num_tokens


def actions_perform(resp,thought):
    actions_list = extract_actions(resp)
    print("actions_list: ",actions_list)
    len_ac_list = len(actions_list)
    if len_ac_list == 0:
        print("In none actions list")
        
        lowercase_text = resp.lower()
        lowercase_hello = "python"

        # Check if the lowercase "hello" is in the lowercase text
        if lowercase_hello in lowercase_text:
            print("Python found")
            python_text = f"Python[{resp}]"
            actions_list = [python_text]
            print("Inside python Action manual input, actions_list: ",actions_list)
        
        else:
            print("Python Not found")
        
        
    print("Outside actions_list: ",actions_list)
    output_list = []
    for i in actions_list:
        i = [i]
        actions_tools_dic = convert_list_to_dict(i)
        print("Action Tools Found: (List) ",actions_tools_dic)
        print(type(actions_tools_dic))
        out = handle_request(actions_tools_dic, thought)
        output_list.append(out)
        
    output = " ".join(output_list)
    return output

def summary_context(text):
    context_len = token_count(text)
    print("Token Length: ",context_len)
    if context_len > 300:
        summm_prompt = f"""
        Your a Editor working in a company AGNOS, your task is to summarize the text given by your manager. You have to perform this job carefully as the company development is dependent on your work. 
        Now summarize this text without loosing any important information, which may include, numbers, values, names, strategies, list or nested lists or any other. 
        You can delete any matter if it doesn't belongs to the context your working.
        You cannot rewrite the summary once writen so carefully do the work. All the best.

        Text:
        {text}
        """
        text = O_LLM_gemini(summm_prompt)
    return text

#------------------------------------------------------------------------------------------------

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

def df_replace_model_names(df):
    replace_dict = {
        'GEMINI 1.0 PRO': 'gemini',
        'GROQ-MIXTRAL': 'mixtral',
        'MISTRAL': 'mistral',
        'OPEN AI': 'gpt3'
    }

    # Perform replacements
    df['model'] = df['model'].replace(replace_dict)

    return df

def read_entire_gspread_sheet_to_pandas(credentials_file, sheet_id):

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
    client = gspread.authorize(creds)

    try:
        sheet = client.open_by_key(sheet_id).sheet1  # Access the first sheet

        # Read all values efficiently (consider spreadsheet size for optimization)
        values = sheet.get_all_values()
        if not values:
            return pd.DataFrame()  # Return an empty DataFrame if no data

        data = pd.DataFrame(values[1:], columns=values[0])  # Skip header row
        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None to indicate an error

def google_sheets_access():
    credentials_file = r"D:\CodePhilly\cred_google_sheet.json"
    sheet_id = ""

    df = read_entire_gspread_sheet_to_pandas(credentials_file, sheet_id)
    

    if df is not None:
        
        df.columns = ["time", "model", "Tools", "rag", "email", "send_copy"]
        df = df_replace_model_names(df)
        # display(df)
        recent_df = df.tail(1)
        # display(recent_df)
        model_req = str(recent_df["model"].values[0])
        tools_req = str(recent_df["Tools"].values[0])
        rag_req = str(recent_df["rag"].values[0])
        email_req = str(recent_df["email"].values[0])

    else:
        print("Error reading the Google Sheet. Check credentials or sheet ID.")
    return model_req, tools_req, rag_req, email_req

#------------------------------------------------------------------------------------------------------------------------------------------
def convert_to_format(text,format):
    prompt=f"Convert the user_input into the given format\n{text}\Structure:{format}"
    return O_LLM_gemini(prompt)

def write_email(inp):
    format="""
    Consider you are writing an email to a person. I want you to convert the user_input into a list structure.
    Structure:
    [To: abc@xyz.com; subject: This is the subject of the email; body: Hi abc\n, body of the email\n Regards,\nabc]
    [To: pqr@xyz.com; subject: This is the subject of the email; body: Hi pqr\n, body of the email\n Regards,\npqr]

    """
    con_inp=convert_to_format(inp,format)
    import smtplib
    prompt=con_inp.replace("To: ",'').replace(' subject: ','').replace(' body: ','').replace('[','').replace(']','').split(';')
    receiver_email=prompt[0]
    subject=prompt[1]
    message=prompt[2]
    email = 'saitanmai.r@gmail.com'
    text = f"Subject: {subject}\n\n{message}"
    server = smtplib.SMTP("smtp.gmail.com", 587) 
    server.starttls()
    server.login(email, "hcva lstk mcpe enir ")
    server.sendmail(email, receiver_email, text)
    return "Email has been sent to " + str(receiver_email) +f"\n {text}"

def duck_go(Keyword):
    # print(Keyword)
    results = DDGS().text(Keyword, max_results=12)
    bodies = [item['body'] for item in results]
    paragraph = ' '.join(bodies)
    return paragraph

def internet(inp):
    format="""
    Consider you are a web-surfer. I want you to search the internet based on the user input and return the summary of what you found in first person.
    User: Who are the founders of xyz - Format: Search[Founders of xyz]
    User: What is IPL - Format: Search[IPL]
    """
    con_inp=convert_to_format(inp,format)
    key=con_inp.replace('Search[','').replace(']','')
    web_result=duck_go(key)
    prompt=f"""Consider you are a web-surfer. I want use the reslut from internet for the user query.
    
    Based on the user input and the output from web answer the query in first person.

    Web_result = {web_result}

    User_query = {inp}

    If you cannot answer, apologize for not being able to answer.
    """
    return(O_LLM_gemini(prompt))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#report rase code

Example_prompt_thoughts = """
Consider yourself a manager at a company called AGNOS Business Solutions, and break down this complex task from your boss (of clients) into multiple simple tasks as thoughts for your assistant to complete. Don't respond to any other tools except these, as they are new and cannot be used other than these: 

Tools available to use: Search[Text or URL to search in the internet]

Task: I want a detailed analysis report of Competators of Luxury shoe market for investors to launch my new shoe brand

Thought 1: First I need to find, which companies work in luxury shoe market in internet
Thought 2: Second, Make a list of all the companies
Thought 3: Third, Now search which products, Number of products, revenue, SWOT analysis of each company listed
Thought 4: Fourth, Now With all the companies information write a important summary
Thought 5: With all the information, I need to find where can we build new shoe brand without much competation 

"""

Example_prompt_Actions = """
You were an assistant to the manager at AGNOS business solutions which have many clients; previously, he gave you tasks and multiple thoughts, which you performed perfectly. Now he gave you the most important task and thoughts. You need to respond to the thoughts carefully and correctly, as your promotion is in his hands. 

He said to use only these Tools: Search[Text to search in the internet or URL], Calculator[Expression or numbers to calculate]
Previous Task: I want a detailed analysis report of Competators of Luxury shoe market for investors to launch my new shoe brand
Thought 1: First I need to find, which companies work in luxury shoe market
Observation: "The most expensive shoe brand in the world is reportedly Stuart Weitzman, who designed a pair of shoes valued at $3 million. Jimmy Choo shoes range in price from $395 to $4,595. Alexander McQueen shoes start at a price point of $620. Valentino's shoe collection starts at a price point of $845.Feb 7, 2024" 
Thought 2: Second, Make a list of all the companies
1. Stuart Weitzman
2. Jimmy Choo shoes
3. Alexander McQueen
4. Valentino shoe
Thought 3: Third, Now search which products, Number of products. 
Action: Search[Stuart Weitzman shoes all products]  
Action: Search[Jimmy Choo shoes all products]  
Action: Search[Alexander McQueen Shoes all products]  
Action: Search[Valentino Shoes all products] 

Completed

"""
improve_prompt = "Consider yourself as a prompt engineer and improve this prompt for better generation of reports as requested by user."

import markdown2
import pdfkit

Markdown_full_text = " "

def Markdown_pdf(markdown_text, output_path):
    global Markdown_full_text
    
    
    Markdown_full_text = markdown_text
    
    html_text = markdown2.markdown(markdown_text)
    print("==================START=======================")
    print(markdown_text)
    print("===================END========================")

def To_do_list(text):
    thoughts = re.findall(r'(?i)(?<=thought\s)\d+:\s(.+)', text)
    return thoughts

def Report(Task):
    global Example_prompt_Actions
    global improve_prompt

    improve_full_prompt = f"{improve_prompt} \n User Query: {Task}"
    Task = O_LLM_openai(improve_full_prompt)

    Task_promp_thoughts = f"""{Example_prompt_thoughts}
    Task: {Task}

    Now write simple multiple Thoughts for this Task and use only tools mentioned. Write Thoughts for this task below and Dont write any actions its not your work to perform.
    """

    print(Task_promp_thoughts)


    thoughts_resp = O_LLM_openai(Task_promp_thoughts)
    thoughts_resp = thoughts_resp.replace('*', '')
    print(thoughts_resp)

    thoughts_list = To_do_list(thoughts_resp)
    thoughts_list.append(f"With all the information give me a report for the task: {Task}")
    print(thoughts_list)
    #------------------------------------------------------------------------------------------------------
    Action_prompt = f"""
    Task: {Task}
    Thought : {thoughts_list[0]}
    """

    Action_disclaimer = " Write search Thought one by one .Write an simple Action for this Thought with correct syntax"

    Markdown_prompt_editor = """
    Your a Editorial Manager in AGNOS Business solutions. Name 'Tillu', where you need to provide a summary report report to your client regrading their request. 
    Here is the final draft of the report, try to build some hidden insights from this and write it in final report, write this draft into beautiful markdown, if already in markdown, try to make it better and clear.
    And in final write your opinion in paragraph. Make the report better and bigger.
    Write a easy to understand markdown text with each sections seperatation.
    """
    
    OBSERVATIONS = []
    i = 0
    for i in range(len(thoughts_list)):
        print(f"------------ITEARATION {i}------------------")
        if i > 0: 
            try:
                Action_prompt = Action_prompt + "Observation: " + observation
            except Exception as e:
                observation = " "
                Action_prompt = Action_prompt + "Observation: " + observation

            Action_prompt = summary_context(Action_prompt)
            Action_prompt = f"{Action_prompt}\n Thought : {thoughts_list[i]} "

        Action_prompt_full = f"{Example_prompt_Actions}\n {Action_prompt} \n {Action_disclaimer} "
        print("********************FULL-PROMPT-START********************")
        print(Action_prompt_full)
        print("********************FULL-PROMPT-END********************")
        print("\n\n")
        Action_resp = O_LLM_gemini(Action_prompt_full)
        Action_resp = Action_resp.replace('*', '')
        print(Action_resp)
        try:
            observation = actions_perform(Action_resp,thoughts_list[i])
            OBSERVATIONS.append(observation)
            print(observation)
        except Exception as e:
            print(e)
            observation = " "

        itr = len(thoughts_list)
        if i == (itr-1):
            Markdown_prompt_full = f"{Markdown_prompt_editor} \n Draft: {Action_prompt} \n {observation} {OBSERVATIONS}"

            fn_output_path = 'Report_Vendor_gemini.pdf'

            final_report = O_LLM_gemini(Markdown_prompt_full)
            Markdown_pdf(final_report,fn_output_path)

            fn_output_path = 'Report_Vendor_openai.pdf'

            final_report = O_LLM_openai(Markdown_prompt_full)
            Markdown_pdf(final_report,fn_output_path)


            ob_output_path = 'Observation_Report_Vendor.pdf'

            OBSERVATIONS = " ".join(OBSERVATIONS)
            Markdown_pdf(OBSERVATIONS,ob_output_path)

            
#Report("query")

#-----------------------------------------------------------------------------------------------------------------------------------

import json
import csv
import re




#----------------------------------------------------------------
#Inventory Management


import csv
import json
import pandas as pd
import re

def check_inventory_file():

    df = pd.read_csv("data.csv")
    return df

def check_inventory_file_small():

    df = pd.read_csv("data.csv")
    return df.head(7)

def Inventory_management(vendor_query):
    
    global Inventory
    
    df = check_inventory_file_small()
    Inventory = df.to_json(orient='records')

    print(Inventory)

    exmaple_prompt = f"""Consider yourself as a Inventory management python code generator, you roles is to generate python to manage inventory for vendor. 
    When Vendor ask you to edit the inventory like 'update the inventory of coffee small i sold 13 coffees today' 
    you need to build a python code to perform that task considering file saved as data.csv
    Vendor: Show me my inventory
    Bot:
    ```Python
    import pandas as pd
    df = pd.read_csv("data.csv")
    df.to_csv("data.csv", index=False)
    print(df)
    ```
    data.csv: {Inventory}
    Note: always respond python code in between tags of ```Python and ```. If you perform any operation update the data.csv file.
    """
    
    prompt2  = f"{exmaple_prompt} \n Vendor: {vendor_query} \n Bot:"
    resp = O_LLM_openai(prompt2)
    print("Response: ", resp)
    Action_python_text = f"{resp}"
    return Action_python_text, resp

#-----------------------------------------------------------------------------------------------

def Inventory_Management_Handler(Vendor_query):
    df_old = check_inventory_file()
    Action_python_code, resp =  Inventory_management(Vendor_query)
    out = actions_perform(Action_python_code,"Python code")
    new_df = check_inventory_file()
    print("\n\n")
    return df_old, new_df, out

def generate_report(Vendor_query):
    Report(Vendor_query)
    return "Report Generated"





def Report_catalogue(Task):
    global Example_prompt_Actions

    Catalogue_header = """
    Consider yourself as a Catalogue Business Analyst, where you have searched internet, and gather information for user's query. 
    Now by looking at the information, write a analysis in structured markdown.
    """
    
    df = check_inventory_file()
    Inventory = df.to_json(orient='records')
    
    Task_promp_thoughts = f"""{Example_prompt_thoughts}
    Task: {Task}
    User Inventory: {Inventory}
    Now write simple multiple Thoughts for this Task and use only tools mentioned. Write Thoughts for this task below and Dont write any actions its not your work to perform.
    """

    print(Task_promp_thoughts)

    thoughts_resp = O_LLM_openai(Task_promp_thoughts)
    thoughts_resp = thoughts_resp.replace('*', '')
    print(thoughts_resp)

    thoughts_list = To_do_list(thoughts_resp)
    thoughts_list.append(f"With all the information give me a report for the task: {Task}")
    print(thoughts_list)
    #------------------------------------------------------------------------------------------------------
    Action_prompt = f"""
    User Inventory: {Inventory}
    Task: {Task}
    Thought : {thoughts_list[0]}
    """

    Action_disclaimer = " Write search Thought one by one .Write an simple Action for this Thought with correct syntax"

    Markdown_prompt_editor = """
    Your a Editorial Manager in AGNOS Business solutions. Name 'Tillu', where you need to provide a summary report report to your client regrading their request. 
    Here is the final draft of the report, try to build some hidden insights from this and write it in final report, write this draft into beautiful markdown, if already in markdown, try to make it better and clear.
    And in final write your opinion in paragraph. Make the report better and bigger.
    Write a easy to understand markdown text with each sections seperatation.
    """
    
    OBSERVATIONS = []
    i = 0
    for i in range(len(thoughts_list)):
        print(f"------------ITEARATION {i}------------------")
        if i > 0: 
            try:
                Action_prompt = Action_prompt + "Observation: " + observation
            except Exception as e:
                observation = " "
                Action_prompt = Action_prompt + "Observation: " + observation

            Action_prompt = summary_context(Action_prompt)
            Action_prompt = f"{Action_prompt}\n Thought : {thoughts_list[i]} "

        Action_prompt_full = f"{Example_prompt_Actions}\n {Action_prompt} \n {Action_disclaimer} "
        print("********************FULL-PROMPT-START********************")
        print(Action_prompt_full)
        print("********************FULL-PROMPT-END********************")
        print("\n\n")
        Action_resp = O_LLM_gemini(Action_prompt_full)
        Action_resp = Action_resp.replace('*', '')
        print(Action_resp)
        try:
            observation = actions_perform(Action_resp,thoughts_list[i])
            OBSERVATIONS.append(observation)
            print(observation)
        except Exception as e:
            print(e)
            observation = " "
            
        itr = len(thoughts_list)
        if i == (itr-1):
            Catalogue_prompt = f"{Catalogue_header} Draft: {Action_prompt} \n {observation} {OBSERVATIONS}"
            Final_answer = O_LLM_gemini(Catalogue_prompt)
            return Final_answer
