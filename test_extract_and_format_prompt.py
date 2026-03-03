from extract_and_format_prompt import generate_user_prompt
import json

# Generate the prompt from the PDF
user_prompt = generate_user_prompt('m16534-protocol-v4-0_redboxed.pdf')

# Write the generated prompt to a file
with open('output_extract_and_format_prompt.json', 'w') as f:
    json.dump(user_prompt, f, indent=2)

print("Output has been written to output_extract_and_format_prompt.json") 