from ollama import chat

SYSTEM_PROMPT = """
You are an enterprise email assistant.

Generate professional emails.

Always:
- Include subject
- Professional tone
- Clear action items
"""

while True:

    user_input = input("You: ")

    response = chat(
        model="llama3",
        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":user_input
            }
        ]
    )

    print(response["message"]["content"])