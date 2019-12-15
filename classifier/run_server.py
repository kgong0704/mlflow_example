import requests
import json
service_url = " http://127.0.0.1:52185/invocations"
#service_url = "http://nsdml.analyst.ai/services/85/predict"
data = {"columns": ["url", "model_name"],
        "data": [["http://abc-crawler.oss-cn-hangzhou.aliyuncs.com/texts/52dedce05b5c6142f37f35819ac5078c4b8f66c40275e0dc440968bb572bf4d1/all.json", "jzy160_model_hs.bin"]]}
try:
 r = requests.post(url=service_url, data=json.dumps(data), headers={"content-type": "application/json"})
except Exception as e:
 # TODO localize message
 raise RuntimeError("Unable to connect to model service. Url={}".format(service_url))

if r.status_code != 200:
 print("Failed to get result from model service. StatusCode=%d, Response=%s", r.status_code, r.content)
 raise RuntimeError("Calling model service failed. StatusCode={}".format(r.status_code))

api_result = json.loads(r.content.decode())
print(api_result)
