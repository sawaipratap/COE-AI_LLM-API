from coeai import LLMinfer

api_key = "your-api-key"
llm = LLMinfer(api_key)

response = llm.generate(
    model="llama4-16x17b",
    inference_type="image-to-text",
    files=["/Users/coe-ai/Downloads/image.jpeg"],
    prompt="Describe this image in detail",
    max_tokens=512
)
print(response)