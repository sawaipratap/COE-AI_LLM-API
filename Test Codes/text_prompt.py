from coeai import LLMinfer

api_key = "your-api-key"
llm = LLMinfer(api_key, host="http://10.9.6.165:8001")

response = llm.generate(
    model="llama4-16x17b",
    inference_type="text-to-text",
    prompt="Explain the difference between supervised and unsupervised learning.",
    max_tokens=256,
    temperature=0.5,
    top_p=0.9,
    stream=False
)
print(response)