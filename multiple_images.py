from coeai import LLMinfer

api_key = "your-api-key"
llm = LLMinfer(api_key)

image_paths = [
    "/Users/coe-ai/Downloads/image1.jpeg",
    "/Users/coe-ai/Downloads/image2.jpeg"
]

response = llm.generate(
    model="llama4-16x17b",
    inference_type="image-to-text",
    files=image_paths,
    prompt="Compare the images and describe similarities and differences",
    max_tokens=512,
    temperature=0.7,
    top_p=1.0
)
print(response)
