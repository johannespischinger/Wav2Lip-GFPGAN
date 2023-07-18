import os
import asyncio
import httpx
from dotenv import load_dotenv

async def run_inference(event):

    load_dotenv('../.env')
    # Set the appropriate Content-Type header
    headers = {
        "Content-Type": "application/json",
        os.environ['API_KEY_NAME']: os.environ['INFERENCE_API_KEY'],
    }

    url = "http://127.0.0.1:8000/inference"  # Adjust IP address

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=event, headers=headers, timeout=20)
            response.raise_for_status()
            response_data = response.json()
        except httpx.HTTPStatusError as e:
            print(e)
            raise Exception(f"Could not connect to API")

    return response_data


def handler(event, context):
    asyncio.run(run_inference(event))
    print(event)
    return {
        "statusCode": 200,
        "body": "Inference triggered successfully",
    }


event = {'Records': [{'eventVersion': '2.1', 'eventSource': 'aws:s3', 'awsRegion': 'eu-central-1', 'eventTime': '2023-06-28T07:54:34.149Z', 'eventName': 'ObjectCreated:Put', 'userIdentity': {'principalId': 'AWS:AIDA2XEJBWKGRJRQRBWQS'}, 'requestParameters': {'sourceIPAddress': '24.134.130.189'}, 'responseElements': {'x-amz-request-id': '6RY4Z0X9FFVVS9YF', 'x-amz-id-2': '4CHasCb6sl9TtNrns22Bierj/hUUfmGuI1uUyONqR372BEghfxNiNuoUW+eQTuF75NPnXdTkcuzrGbLE5JjYm+bNosEkK39W'}, 's3': {'s3SchemaVersion': '1.0', 'configurationId': 'susio-dev-audiotransfer-3fa24e0836e79f7bd46874ffab17b742', 'bucket': {'name': 'dev.susio.videogeneration', 'ownerIdentity': {'principalId': 'A2BPFWF8CFD43W'}, 'arn': 'arn:aws:s3:::dev.susio.videogeneration'}, 'object': {'key': 'UID001/SID001/CID021/audio/CID021_Simon.wav', 'size': 441102, 'eTag': 'f429108f646378075138b529f9426c1b', 'sequencer': '00649BE739DBC947A8'}}}]}

if __name__ == "__main__":
    handler(event, None)
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(handler(event, None))

