from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()  # legge automaticamente ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",  # usa l'ID esatto del modello
    max_tokens=1024,
    system="Sei un professore esperto di informatica. Rispondi in modo semplice e sintetico.",  # <--- System prompt aggiunto qui
    messages=[
        {"role": "user", "content": "Spiegami cos'è un token in una frase."}
    ]
)

print(message.content[0].text)