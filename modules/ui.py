import html
import json
import math
import mimetypes
import os
import random
import string
import sys
from functools import partial, reduce
import warnings

import gradio as gr
import gradio.routes
import gradio.utils
from PIL import Image

from modules import sd_hijack, sd_models, localization, script_callbacks, ui_extensions, deepbooru, sd_vae, \
    extra_networks, postprocessing, ui_components, ui_common, ui_postprocessing, progress
from modules.ui_components import FormRow, FormColumn, FormGroup, ToolButton, FormHTML
from modules.paths import script_path, data_path

from modules.shared import opts, cmd_opts, restricted_opts

import modules.codeformer_model
import modules.gfpgan_model
import modules.hypernetworks.ui
import modules.scripts
import modules.shared as shared
import modules.styles
import modules.textual_inversion.ui
from modules import prompt_parser
from modules.sd_hijack import model_hijack
import modules.hypernetworks.ui
from modules.generation_parameters_copypaste import image_from_url_text
import modules.extras
from guiju.segment_anything_util.dino import dino_model_list
from guiju.segment_anything_util.sam import sam_model_list, sam_predict

warnings.filterwarnings("default" if opts.show_warnings else "ignore", category=UserWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok

    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_region
    )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5'  # ⇅
restore_progress_symbol = '\U0001F300'  # 🌀


def plaintext_to_html(text):
    return ui_common.plaintext_to_html(text)


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])


def visit(x, func, path=""):
    if hasattr(x, 'children'):
        if isinstance(x, gr.Tabs) and x.elem_id is not None:
            # Tabs element can't have a label, have to use elem_id instead
            func(f"{path}/Tabs@{x.elem_id}", x)
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(f"{path}/{x.label}", x)


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show() for x in range(4)]

    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    shared.prompt_styles.save_styles(shared.styles_filename)

    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(2)]


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    from modules import processing, devices

    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale,
                                                    hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)

    with devices.autocast():
        p.init([""], [0], [0])

    return f"resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "no image selected"

    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def apply_styles(prompt, prompt_neg, styles):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]


def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a'))

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def create_seed_inputs(target_interface):
    with FormRow(elem_id=f"{target_interface}_seed_row", variant="compact", visible=False):
        seed = (gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(label='Seed', value=-1,
                                                                        elem_id=f"{target_interface}_seed")
        seed.style(container=False)
        random_seed = ToolButton(random_symbol, elem_id=f"{target_interface}_random_seed", label='Random seed')
        reuse_seed = ToolButton(reuse_symbol, elem_id=f"{target_interface}_reuse_seed", label='Reuse seed')

        seed_checkbox = gr.Checkbox(label='Extra', elem_id=f"{target_interface}_subseed_show", value=False)

    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []

    with FormRow(visible=False, elem_id=f"{target_interface}_subseed_row") as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        subseed = gr.Number(label='Variation seed', value=-1, elem_id=f"{target_interface}_subseed")
        subseed.style(container=False)
        random_subseed = ToolButton(random_symbol, elem_id=f"{target_interface}_random_subseed")
        reuse_subseed = ToolButton(reuse_symbol, elem_id=f"{target_interface}_reuse_subseed")
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01,
                                     elem_id=f"{target_interface}_subseed_strength")

    with FormRow(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0,
                                       elem_id=f"{target_interface}_seed_resize_from_w")
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0,
                                       elem_id=f"{target_interface}_seed_resize_from_h")

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}

    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)

    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox


def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component,
                       is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""

    def copy_seed(gen_info_string: str, index):
        res = -1

        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        except json.decoder.JSONDecodeError as e:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)

        return [res, gr_show(False)]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, dummy_component],
        outputs=[seed, dummy_component]
    )


def update_token_counter(text, steps):
    try:
        text, _ = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts],
                                  key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"

    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6, visible=False):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3,
                                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt",
                                                     show_label=False, lines=3,
                                                     placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")

        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_classes="interrogate-col", visible=False):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")

        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt",
                                      elem_classes="generate-box-interrupt")
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip", elem_classes="generate-box-skip")
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                skip.click(
                    fn=lambda: shared.state.skip(),
                    inputs=[],
                    outputs=[],
                )

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

            with gr.Row(elem_id=f"{id_part}_tools", visible=False):
                paste = ToolButton(value=paste_symbol, elem_id="paste")
                clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
                extra_networks_button = ToolButton(value=extra_networks_symbol, elem_id=f"{id_part}_extra_networks")
                prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id=f"{id_part}_style_apply")
                save_style = ToolButton(value=save_style_symbol, elem_id=f"{id_part}_style_create")
                restore_progress_button = ToolButton(value=restore_progress_symbol,
                                                     elem_id=f"{id_part}_restore_progress", visible=False)

                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter",
                                        elem_classes=["token-counter"], visible=False)
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter",
                                                 elem_classes=["token-counter"], visible=False)
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")

                clear_prompt_button.click(
                    fn=lambda *x: x,
                    _js="confirm_clear_prompt",
                    inputs=[prompt, negative_prompt],
                    outputs=[prompt, negative_prompt],
                )

            with gr.Row(elem_id=f"{id_part}_styles_row", visible=False):
                prompt_styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles",
                                            choices=[k for k, v in shared.prompt_styles.styles.items()], value=[],
                                            multiselect=True)
                create_refresh_button(prompt_styles, shared.prompt_styles.reload,
                                      lambda: {"choices": [k for k, v in shared.prompt_styles.styles.items()]},
                                      f"refresh_{id_part}_styles")

    return prompt, prompt_styles, negative_prompt, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button, restore_progress_button


def setup_progressbar(*args, **kwargs):
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()

    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)

        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()

    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()

    opts.save(shared.config_filename)
    return getattr(opts, key)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button


def create_output_panel(tabname, outdir):
    return ui_common.create_output_panel(tabname, outdir)


def create_sampler_and_steps_selection(choices, tabname):
    if opts.samplers_in_dropdown:
        with FormRow(elem_id=f"sampler_selection_{tabname}", visible=False):
            sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling",
                                        choices=[x.name for x in choices], value=choices[0].name, type="index")
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps",
                              value=20)
    else:
        with FormGroup(elem_id=f"sampler_selection_{tabname}"):
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps",
                              value=20)
            sampler_index = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling",
                                     choices=[x.name for x in choices], value=choices[0].name, type="index")

    return steps, sampler_index


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder.split(","))}

    for i, category in sorted(enumerate(shared.ui_reorder_categories),
                              key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def get_value_for_setting(key):
    value = getattr(opts, key)

    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision'}}

    return gr.update(value=value, **args)


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings",
                           multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=len(x) > 0),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown


def create_ui():
    import modules.img2img

    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)


    with gr.Blocks(analytics_enabled=False, title="cloths_inpaint") as demo:
        with gr.Row():
            sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list,
                                         value=sam_model_list[0] if len(sam_model_list) > 0 else None, visible=True)
            dino_box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3,
                                           step=0.001,
                                           visible=True)
            dino_text_prompt = gr.Textbox(value='clothing',
                                          placeholder="You must enter text prompts to enable groundingdino. Otherwise this extension will fall back to point prompts only.",
                                          label="GroundingDINO Detection Prompt", elem_id=f"dino_text_prompt",
                                          visible=True)
            dino_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)",
                                          choices=dino_model_list,
                                          value=dino_model_list[1], visible=True)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Image for Segment Anything", elem_id=f"input_image", source="upload",
                                       type="pil", image_mode="RGBA")

            with gr.Column(scale=1):
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"result_gallery").style(grid=3)
                sam_result = gr.Text(value="", label="Segment Anything status")

        # img2img input args
        with gr.Row(visible=False):
            dummy_component = gr.Label(visible=False)

        def predict(_sam_model_name, _dino_model_name, _dino_text_prompt, _box_threshold, _input_image):
            sam_result_gallery, sam_result = sam_predict(_sam_model_name, _dino_model_name, _dino_text_prompt,
                                                         _box_threshold,
                                                         _input_image)

            sam_result_tmp_png_fp = []

            for sam_mask_img in sam_result_gallery:
                cache_fp = f"tmp/{''.join([random.choice(string.ascii_letters) for c in range(15)])}.png"
                sam_mask_img.save(cache_fp)
                sam_result_tmp_png_fp.append({'name':cache_fp})

            task_id = f"task({''.join([random.choice(string.ascii_letters) for c in range(15)])})"

            sd_positive_prompt = '(RAW photo, best quality), (realistic, photo-realistic:1.3), masterpiece, an extremely delicate and beautiful, extremely detailed, 2k wallpaper, light smile, extremely detailed CG unity 8k wallpaper, ultra-detailed, highres, beautiful detailed girl, detailed fingers, light on face, 1girl, cute, young, realistic face, realistic body, girl posing for a photo,   good hand,looking at viewer, (simple background:1.3), (white background:1.3)'
            lora_positive_prompt = ',<lora:shojovibe_v11:0.4> ,<lora:koreanDollLikeness:0.4>'
            lora_positive_prompt += lora_positive_prompt
            sd_negative_prompt = 'NSFW,paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, ((watermark:2)), (white letters:1), (multi nipples), bad anatomy, bad hands, text, error, missing fingers, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, bad feet, Multiple people, blurry, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many'
            prompt_styles = None
            init_img = None
            sketch = None
            init_img_with_mask = None
            inpaint_color_sketch = None
            inpaint_color_sketch_orig = None
            init_img_inpaint = None
            init_mask_inpaint = None
            steps = 20
            sampler_index = 16
            mask_blur = 4
            mask_alpha = 0
            inpainting_fill = 1
            restore_faces = True
            tiling = False
            n_iter = 1
            batch_size = 1
            cfg_scale = 7
            image_cfg_scale = 1.5
            denoising_strength = 0.75
            seed = -1.0
            subseed = -1.0
            subseed_strength = 0
            seed_resize_from_h = 0
            seed_resize_from_w = 0
            seed_enable_extras = False
            selected_scale_tab = 0
            height = 512
            width = 512
            scale_by = 1
            resize_mode = 2
            inpaint_full_res = 0
            inpaint_full_res_padding = 32
            inpainting_mask_invert = 1
            img2img_batch_input_dir = ''
            img2img_batch_output_dir = ''
            img2img_batch_inpaint_mask_dir = ''
            override_settings_texts = []

            sam_args = [0, True, False, 0, _input_image,
                        sam_result_tmp_png_fp,
                        2, False, [], [], False, 0, 1, False, False, 0, None, [], -2, False, [],
                        '<ul>\n<li><code>CFG Scale</code>should be 2 or lower.</li>\n</ul>\n',
                        True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None',
                        '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>',
                        128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'],
                        False, False, 'positive', 'comma', 0, False, False, '',
                        '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>',
                        64, 0, 2, 1, '', [], 0, '', [], 0, '', [], True, False, False, False, 0
                        ]

            res = modules.img2img.img2img(task_id, 4, sd_positive_prompt, sd_negative_prompt, prompt_styles, init_img,
                                          sketch,
                                          init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig,
                                          init_img_inpaint, init_mask_inpaint,
                                          steps, sampler_index, mask_blur, mask_alpha, inpainting_fill, restore_faces,
                                          tiling,
                                          n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength, seed,
                                          subseed,
                                          subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras,
                                          selected_scale_tab, height, width, scale_by, resize_mode, inpaint_full_res,
                                          inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir,
                                          img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                                          override_settings_texts,
                                          *sam_args)

            return res[0], res[1]

        input_image.change(
            fn=predict,
            inputs=[sam_model_name,
                    dino_model_name,
                    dino_text_prompt,
                    dino_box_threshold,
                    input_image],
            outputs=[result_gallery, sam_result]
        )

    modules.scripts.scripts_current = None

    return demo


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'

def javascript_html():
    # Ensure localization is in `window` before scripts
    head = f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n'

    script_js = os.path.join(script_path, "script.js")
    head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    for script in modules.scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    if cmd_opts.theme:
        head += f'<script type="text/javascript">set_theme(\"{cmd_opts.theme}\");</script>\n'

    return head

def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        head += stylesheet(cssfile)

    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))

    return head

def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response

if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse

def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    tag = launch.git_tag()

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
version: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{tag}</a>
 • 
python: <span title="{sys.version}">{python_version}</span>
 • 
torch: {getattr(torch, '__long_version__', torch.__version__)}
 • 
xformers: {xformers_version}
 • 
gradio: {gr.__version__}
 • 
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""

def setup_ui_api(app):
    from pydantic import BaseModel, Field
    from typing import List

    class QuicksettingsHint(BaseModel):
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"],
                      response_model=List[QuicksettingsHint])

    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])
