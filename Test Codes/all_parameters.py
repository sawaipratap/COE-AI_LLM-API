from coeai import LLMinfer

api_key = "your-api-key"
llm = LLMinfer(api_key)

# Test with all available parameters
response = llm.generate(
    model="deepseek-r1-70b",
    inference_type="text-to-text",
    prompt="Write a technical explanation of blockchain technology.",
    max_tokens=400,
    temperature=0.3,  # Low temperature for technical accuracy
    top_p=0.85,
    stream=False
)
print(response)