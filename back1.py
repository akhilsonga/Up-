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

    pass

def get_todays_date():
    date = datetime.today().date()
    dt = "Today's date is:" + str(date)
    return dt

dt = get_todays_date()


def O_LLM_Mixtral(query):
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



def O_LLM_llama3(query):
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
        model="llama3-8b-8192",
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
    
    
def O_LLM(query, model):
    if model == "gpt3":
        output = O_LLM_openai(query)
    elif model == "gemini":
        output = O_LLM_gemini(query)
    elif model == "mixtral":
        output = O_LLM_Mixtral(query)
    elif model == "llama3":
        output = O_LLM_llama3(query)
    else:
        output = O_LLM()
    return output
    
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
    return result, err

code_manager_prompt = """
Consider yourself as a python code generator and your responsibility is to satisfy user queries by writing a perfectly working python code.

you need to build a python code to perform that task considering file saved as data.csv
Vendor: Show me my inventory
Bot:
```Python
import pandas as pd
df = pd.read_csv("data.csv")
print(df)
```

Note: always respond python code in between tags of ```Python and ```. 

User Query:
"""


error_counter = 0
def Code_execution_manager(answer,thought):
    global error_counter
    
    result, err = code_processing(answer)
    if err!= 0 and error_counter == 0 :
        print()
        model = "gpt3"
        code_prompt = f"{code_manager_prompt} {thought} Error code: ```Python\n{answer}\n``` \n\n Bot:"
        code_resp = O_LLM(code_prompt, model)
        result, err = code_processing(code_resp)
        error_counter = error_counter + 1
        return result
    if err!= 0 and error_counter > 0:
        code_prompt = f"{code_manager_prompt} {thought} \n\n Bot: "
        code_resp = O_LLM(code_prompt, model)
        result, err = code_processing(code_resp)
        return result
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
#         print("In DuckDuckGo Search Question:", thought)
        print("Duck Duck Go :",output)
        print("\n\n\n")
        model = "gemini"
        # prompt = f"Write a small information rich summary of this text. \n\n\n Text: {output}"
        # output = O_LLM(prompt, model)
        prompt = f"consider the text based on the reference, if you found no connection between the Question and Reference write a summary of the text, \n Question: {thought}\n\n\n Reference: {output}"
        output = O_LLM_gemini(prompt)
        
        return output
    
    elif "Python" in data:
        global error_counter
        error_counter = 0
        output = Code_execution_manager(data["Python"],thought)

        print("Output from the Executor: ", output)
        print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        prompt = f"""Consider yourself as a helpful AI assistant and your task is to answer user query with the help of context provided.
        User query: {thought} 
        Context: {output} 
        Write the answer of the user's question in only beautiful markdown format."""

        print(prompt)
        print("\n\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        model = "gemini"
        output = O_LLM(prompt, model)
        #output = output.replace('```', '')
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
    
    print("output_list: ",output_list)
    output = " ".join(output_list)
    print("ALL ACTIONS PERFORMED: ",output)
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
        model = "gemini"
        text = O_LLM(summm_prompt, model)
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
    model = "gemini"

    prompt=f"Convert the user_input into the given format\n{text}\Structure:{format}"

    return O_LLM(prompt, model)

def write_email(inp):
    format="""
    Consider you are writing an email to a person. I want you to convert the user_input into a list structure, you can improve the body of the mail but dont make it very lengthy.
    Structure:
    [To: abc@xyz.com; subject: This is the subject of the email; body: Hi abc\n, body of the email\n Regards,\nabc]
    [To: pqr@xyz.com; subject: This is the subject of the email; body: Hi pqr\n, body of the email\n Regards,\npqr]
    """
    model = "gemini"
#     prompt_improve = f"Consider yourself as a assistant working under your boss and he want to send the this letter to his employees, Now with your knowledege write a short improved letter like your boss by checking this. {inp} "
#     up_text = O_LLM(prompt_improve,model)
    
    con_inp=convert_to_format(inp,format)
    
    import smtplib
    
    prompt=con_inp.replace("To: ",'').replace(' subject: ','').replace(' body: ','').replace('[','').replace(']','').split(';')
    receiver_email=prompt[0]
    subject=prompt[1]
    message=prompt[2]
    email = ''
    text = f"Subject: {subject}\n\n{message}"
    server = smtplib.SMTP("", 000) 
    server.starttls()
    server.login(email, "")
    server.sendmail(email, receiver_email, text)
    return "Email has been sent to " + str(receiver_email) +f"\n {text}"

def duck_go(Keyword):
    # print(Keyword)
    results = DDGS().text(Keyword, max_results=12)
    bodies = [item['body'] for item in results]
    paragraph = ' '.join(bodies)
    return paragraph

def internet(inp):
    format = """
    Consider you are a web-surfer. I want you to search the internet based on the user input and return the summary of what you found in first person.
    User: Who are the founders of xyz ?
    Bot:
    Action: Search[Founders of xyz]
    
    User: What is IPL?
    Bot:
    Action: Search[IPL]
    
    User: Best selling product in philadelphia today?
    Bot:
    Action: Search[Best selling product in philadelphia today]
    
    User: What is best selling book and who is the author of harry potter?
    Bot: 
    Action: Search[What is best selling book]
    Action: Search[author of harry potter]
    
    """
    
    respo =convert_to_format(inp,format)

    output = actions_perform(respo,inp)
    query = f"Answer this question: {inp}, \n\n context: {output}"
    resp = O_LLM(query, "gemini")
    print("QUERY:", query)
    print("\n\n\nOutput respons multiple gunur\n\n\n", resp)
    return resp

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
    model = "gpt3"
    Task = O_LLM(improve_full_prompt, model)

    Task_promp_thoughts = f"""{Example_prompt_thoughts}
    Task: {Task}

    Now write simple multiple Thoughts for this Task and use only tools mentioned. Write Thoughts for this task below and Dont write any actions its not your work to perform.
    """

    print(Task_promp_thoughts)

    model = "gpt3"
    thoughts_resp = O_LLM(Task_promp_thoughts, model)
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

    Action_disclaimer = " Write search Thought one by one. Write an simple Action for this Thought with correct syntax"

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
        model = "gemini"
        Action_resp = O_LLM(Action_prompt_full, model)
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


def get_all_columns():
    df = pd.read_csv('data.csv')
    columns = df.columns.tolist()
    return columns


def Inventory_management(vendor_query):
    
    global Inventory
    
    df = check_inventory_file_small()
    Inventory = df.to_json(orient='records')

    print(Inventory)
    columns = get_all_columns()
    Inventory_prompt_improve = f"""Consider yourself as a task understanding bot of the inventory, your job is to write a cleaner and clear question of the User. 
    User: I want coffee.
    Bot: I want list of all coffee items from the inventory
    User: What is the best product?
    Bot: Give me the most selling product and highest price product.

    Here are all the features available in the inventory: {Inventory}

    User: {vendor_query}
    Bot:"""


    vendor_query = O_LLM(Inventory_prompt_improve, "gemini")
    print("VENDOR QUERY UPGRADED:",vendor_query)
    print("_+_+_+_+_+_+_+___+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_")

    exmaple_prompt = f"""Consider yourself as a Inventory management python code generator, you roles is to generate python to manage inventory for vendor. If User have preference like bread, eggs, steam milk , etc try checking it in Description.
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
    return Action_python_text, resp, vendor_query

#-----------------------------------------------------------------------------------------------

def Inventory_Management_Handler(Vendor_query):
    df_old = check_inventory_file()
    Action_python_code, resp, vendor_query =  Inventory_management(Vendor_query)
    out = actions_perform(Action_python_code,vendor_query)
    new_df = check_inventory_file()
    print("\n\n")
    return df_old, new_df, out

def generate_report(Vendor_query):
    Report(Vendor_query)
    return "Report Generated"



# Vendor_query = "How much money did i earned?"
# df_old, new_df, out = Inventory_Management_Handler(Vendor_query)


# Vendor_query = "Perform analysis of which coffee is best seller at philadelphia, PA."
# generate_report(Vendor_query)





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
