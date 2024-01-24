# coding=utf-8
# @Time : 2024/1/24 上午10:03
# @File : paypal_test.py
import base64
import requests

client_id = 'AcOS-pWypZ1RaAiDtVKgdZrDQswef6T5BnTAnZ9mS6CNvtGGbhPOMhZndF071NsphTy36dhshHDt1Y3e'
client_secret = 'EBnWq7Wr5gi54nSwpSpvbKRzecgi8X7G73Xy7SO_lz1HJP9CAbKU4Jw5pSRYUG3KOn9yJz6QC6b8pcaB'
base_url = 'https://api-m.sandbox.paypal.com'


def aspayapl_generate_ccess_token():
    # Encode client ID and client secret as base64
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    response = requests.post(f"{base_url}/v1/oauth2/token",
                             data="grant_type=client_credentials",
                             headers={"Authorization": f"Basic {auth}"})
    # Parse the JSON response
    data = response.json()

    # Return the access token
    return data["access_token"]

access_token = aspayapl_generate_ccess_token()

# list product
headers = {
    "Authorization": f"Bearer {access_token}",
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

params = (
    ('page_size', '2'),
    ('page', '1'),
    ('total_required', 'true'),
)

response = requests.get('https://api-m.sandbox.paypal.com/v1/catalogs/products', headers=headers, params=params)
print(response.json())

# list plan
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Prefer': 'return=representation',
}

params = (
    ('sort_by', 'create_time'),
    ('sort_order', 'desc'),
)

response = requests.get('https://api-m.sandbox.paypal.com/v1/billing/plans', headers=headers, params=params)
print(response.json())