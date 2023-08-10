# -*- coding: utf-8 -*-
import datetime
from functools import wraps

import jwt
from sanic.exceptions import SanicException
from sanic.response import redirect

class JWTAuth(object):
    _SECRET = 'xczxc$^#BCrg9756guijutech!1201%&#cbhtexlpou'
    _SECRET_WEB = 'xczxc$YFBFEWCig3jutech!1201%&#cbhtexlpou'
    _TYPE = 'jwt'
    _ALGORITHM = 'HS256'

    _instance = None

    # singleton
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        pass

    @classmethod
    def create_token(cls, type='web' , payload={}, scope="*", expired_days=None):
        """
        创建token
        :param type:  'web'
        :param payload:  例如：{'user_id':1,'username':'xxx@xxx.xx'}用户信息
        :return:
        """
        headers = {
            'typ': cls._TYPE,
            'alg': cls._ALGORITHM
        }

        payload['scope'] = scope
        if expired_days:
            payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(days=expired_days)
        result = jwt.encode(payload=payload, key=cls._SECRET_WEB if type=='web' else cls._SECRET, algorithm=cls._ALGORITHM, headers=headers)
        return result

    @classmethod
    def check_token(cls, request):
        if not request.token:
            return False

        try:
            jwt.decode(
                request.token, cls._SECRET, algorithms=["HS256"]
            )
        except jwt.exceptions.InvalidTokenError:
            return False
        else:
            return True

    @classmethod
    def jwt_protected(cls, required_scope):
        @wraps(required_scope)
        def decorator(method):
            @wraps(method)
            async def decorated_function(c, request, *args, **kwargs):
                access_token = request.args.get("access_token")

                if access_token:
                    try:
                        payload = jwt.decode(access_token, JWTAuth._SECRET, algorithms=[JWTAuth._ALGORITHM])
                        scope = payload.get("scope")
                        if scope != '*' and required_scope not in scope:
                            raise SanicException("Not authenticated.", status_code=401)
                        else:
                            return await method(c, request, *args, **kwargs)
                    except jwt.ExpiredSignatureError:
                        raise SanicException("Token expired.", status_code=401)
                    except Exception:
                        raise SanicException("Could not validate credentials.", status_code=401)

                else:
                    raise SanicException("Not authenticated.", status_code=401)

            return decorated_function

        return decorator

    @staticmethod
    def jwt_login_required(method):
        async def decorated_function(cls, request, *args, **kwargs):
            try:
                access_token = request.cookies.get("PTOKEN")
                payload = jwt.decode(access_token, JWTAuth._SECRET_WEB, algorithms=[JWTAuth._ALGORITHM])
                # scope = payload.get("scope")
                # if scope != '*' and required_scope not in scope:
                #     raise SanicException("Not authenticated.", status_code=401)
                # else:
                return await method(cls, request, *args, **kwargs)
            except jwt.ExpiredSignatureError:
                raise SanicException("Token expired.", status_code=401)
            except Exception:
                # raise SanicException("Could not validate credentials.", status_code=401)
                return redirect('login')
            else:
                raise SanicException("Not authenticated.", status_code=401)

        return decorated_function

    @classmethod
    def web_login(cls, payload):
        return cls.create_token(type='web', payload=payload, expired_days=1)

    @classmethod
    def web_logout(cls):
        pass

# @classmethod
# def protected(wrapped):
#     def decorator(f):
#         @wraps(f)
#         async def decorated_function(request, *args, **kwargs):
#             is_authenticated = JWTAuth.check_token(request)
#
#             if is_authenticated:
#                 response = await f(request, *args, **kwargs)
#                 return response
#             else:
#                 return text("You are unauthorized.", 401)
#
#         return decorated_function
#
#     return decorator(wrapped)
# print(JWTAuth.create_token())