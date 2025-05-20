from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="nvidia/llama-3.3-nemotron-super-49b-v1",
  messages=[{"role":"system","content":"detailed thinking off"},{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=1024,
  frequency_penalty=0,
  presence_penalty=0,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

