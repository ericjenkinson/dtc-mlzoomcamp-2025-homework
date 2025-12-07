import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
data = {'url': 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'}

response = requests.post(url, json=data)

# --- DEBUGGING LINES ---
print("Status Code:", response.status_code)
print("Raw Response Text:")
print(response.text) 
# -----------------------

# Only try to parse JSON if the status is OK (200)
if response.status_code == 200:
    print("Model Output:", response.json())