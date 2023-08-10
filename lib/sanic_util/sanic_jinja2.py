# -*- coding: utf-8 -*-
import sys

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sanic.response import html


class SanicJinja2:
    template_paths = ['./templates/']

    # 配置同步环境
    env_sync = Environment(loader=FileSystemLoader(template_paths),
                           autoescape=select_autoescape(['html', 'xml', 'tpl']),
                           enable_async=False)

    # 配置异步环境
    env_async = Environment(loader=FileSystemLoader(template_paths),
                            autoescape=select_autoescape(['html', 'xml', 'tpl']),
                            enable_async=sys.version_info >= (3, 6))

    # 同步模板渲染
    @classmethod
    def template_render_sync(cls, template_file, **kwargs):
        template = cls.env_sync.get_template(template_file)
        rendered_template = template.render(kwargs)
        return html(rendered_template)

    # 异步模板渲染
    @classmethod
    async def template_render_async(cls, template_file, **kwargs):
        template = cls.env_async.get_template(template_file)
        rendered_template = await template.render_async(kwargs)
        return html(rendered_template)

# SanicJinja2 = SanicJinja2()
