# to install
# python3 -m pip install rxinferclient 

from rxinferclient import RxInferClient

# Initialize with default settings (auto-generates API key)
client = RxInferClient()

# Or initialize with custom server URL
# client = RxInferClient(server_url="http://localhost:8000/v1")

# Or initialize with your own API key
# client = RxInferClient(api_key="your-api-key")

# Ping the server to check if it's running
response = client.server.ping_server()
print(response.status)  # 'ok'

# Create a model instance
response = client.models.create_model_instance({ 
    "model_name": "BetaBernoulli-v1"
})
instance_id = response.instance_id

# Delete the model instance when done
client.models.delete_model_instance(instance_id=instance_id)