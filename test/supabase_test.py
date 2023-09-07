# coding=utf-8
# @Time : 2023/8/24 下午3:26
# @File : supabase_test.py
from supabase import create_client

from utils.global_vars import CONFIG


user_id = '80e86a7c-eddb-4c48-bd37-f8f8b3c9614d'
supabase_client = create_client(CONFIG['supabase']['url'], CONFIG['supabase']['key'])

# data = supabase_client.table("transaction").update(
#                             {"status": 0}).eq("id", 1).execute().data
data = supabase_client.table("account").select("*").eq("id", user_id).execute()

print(data)
