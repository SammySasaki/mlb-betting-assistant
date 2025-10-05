from openai import OpenAI
from graph.workflow import build_graph

app = build_graph()



if __name__ == "__main__":
    print("Welcome to the MLB chatbot! Type 'quit' or 'exit' to leave.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        
        result = app.invoke({"input": user_input})
        # Assuming your result dict has a key 'output' for the chatbot response
        print("Bot:", result.get("output", "No response"))

# if __name__ == "__main__":
#     # user_input = "Should I bet the over in tonight's Dodgers game?"
#     # user_input = "How many home runs does Shohei Ohtani have this season?"
#     user_input = "How many runs have the Giants scored in each of their last 5 games against the Dodgers?"
#     result = app.invoke({"input": user_input})
#     print(result)