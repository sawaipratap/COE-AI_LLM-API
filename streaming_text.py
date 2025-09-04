from coeai import LLMinfer

api_key = "your-api-key"
llm = LLMinfer(api_key)

response = llm.generate(
    model="llama4-16x17b",
    inference_type="text-to-text",
    prompt="Write a 5-line poem about AI.",
    max_tokens=150,
    temperature=0.8,
    top_p=0.9,
    stream=True,
    print_stream=True  # Prints partial outputs
)
print("\\nCollected response:", response)
