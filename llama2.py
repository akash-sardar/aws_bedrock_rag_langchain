import json
import os
import sys
import boto3


prompt_data = """
You are a journalist and report on India's cricket world cup final win over south africa
"""

bedrock = boto3.client(service_name = "bedrock-runtime")
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len" : 512,
    "temperature" : 0.1,
    "top_p": 0.9
}
body = json.dumps(payload)
model_id = "meta.llama2-70b-chat-v1"
response = bedrock.invoke_model(
    modelId = model_id,
    body = body,
    contentType = "application/json",
    accept =  "application/json",
)
# get the body the response in JSON format from the response
response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]
print(response_text)
