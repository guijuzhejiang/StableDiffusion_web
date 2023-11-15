# coding=utf-8
# @Time : 2023/8/24 下午3:26
# @File : supabase_test.py
import aiosupabase
import gotrue
# from gotrue import SyncGoTrueAdminAPI
# from gotrue import SyncGoTrueAdminAPI
from gotrue import check_response

user_id = '80e86a7c-eddb-4c48-bd37-f8f8b3c9614d'
url = "https://www.guijutech.com:8888/"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogInNlcnZpY2Vfcm9sZSIsCiAgImlzcyI6ICJzdXBhYmFzZSIsCiAgImlhdCI6IDE2OTU4MzA0MDAsCiAgImV4cCI6IDE4NTM2ODMyMDAKfQ.QanqKpEYyjqgvl1ElcWw7JJAvUEzIC0e0w1pFfPOITE"
# supabase_client = create_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'])
supabase_client = aiosupabase.Supabase
supabase_client.configure(
        url=url,
        key=key,
        debug_enabled=True,
)
# data = supabase_client.table("transaction").update(
#                             {"status": 0}).eq("id", 1).execute().data
# data = supabase_client.table("account").select("*").eq("id", user_id).execute()

# headers = self.headers
#         url = f"{self.url}/admin/users"
#         response = self.http_client.get(url, headers=headers)
#         check_response(response)
#         users = response.json().get("users")
#         if users is None:
#             raise APIError("No users found in response", 400)
#         if not isinstance(users, list):
#             raise APIError("Expected a list of users", 400)
#         return parse_obj_as(List[User], users)

h = supabase_client.auth.headers
response = supabase_client.auth.sync_api.http_client.get(f"{supabase_client.auth.url}/admin/users?per_page=9999", headers=h)
check_response(response)
users = response.json().get("users")
print(users)
# print(len(supabase_client.auth.list_users()))
# a = supabase_client.from_('users').select('*').execute()
# print(a)
