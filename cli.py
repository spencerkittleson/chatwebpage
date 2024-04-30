from rag import ChatWebPage
from dotenv import dotenv_values

if __name__ == '__main__':
    config = dotenv_values(".env")
    assistant = ChatWebPage(
        config['model'], config['ollama_url'])
    print('ingesting')
    assistant.ingest(webpage="https://expert-help.nice.com/Integrations/API/Authorization_Tokens/Token_Exchange")
    while True:
        print(assistant.ask(input('ask: ')))

        