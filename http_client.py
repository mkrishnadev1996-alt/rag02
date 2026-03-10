import httpx

# Configure httpx clients to disable SSL verification
client = httpx.Client(verify=False)
async_client = httpx.AsyncClient(verify=False)