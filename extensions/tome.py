import tomesd
import gradio as gr

from modules import script_callbacks, shared


def on_model_loaded(sd_model):
	if hasattr(shared.opts, 'token_merging_enabled') and shared.opts.token_merging_enabled:
		print('Applying ToMe patch...')
		
		try:
			tomesd.apply_patch(
				sd_model,
				ratio=shared.opts.token_merging_ratio,
				max_downsample=int(shared.opts.token_merging_max_downsample),
				sx=shared.opts.token_merging_stride_x,
				sy=shared.opts.token_merging_stride_y,
				use_rand=shared.opts.token_merging_use_rand,
				merge_attn=shared.opts.token_merging_merge_attn,
				merge_crossattn=shared.opts.token_merging_merge_crossattn,
				merge_mlp=shared.opts.token_merging_merge_mlp
			)
		except Exception as e:
			print('Failed to apply ToMe patch, continuing as normal', e)
			return
		
		print('ToMe patch applied')
	else:
		try:
			print('Removing ToMe patch (if exists)')
			tomesd.remove_patch(sd_model)
		except Exception as e:
			print('Exception thrown when removing ToMe patch, continuing as normal', e)
			return

def on_ui_settings():
	section = ('token_merging', 'Token Merging')
	shared.opts.add_option('token_merging_enabled', shared.OptionInfo(
		False, 'Enable Token Merging', section=section
	))
	shared.opts.add_option("token_merging_ratio", shared.OptionInfo(
		0.5, "Token Merging - Ratio",
		gr.Slider, {"minimum": 0, "maximum": 0.75, "step": 0.1}, section=section
	))
	shared.opts.add_option("token_merging_max_downsample", shared.OptionInfo(
		"1", "Token Merging - Max downsample",
		gr.Dropdown, lambda: {"choices": ["1","2","4","8"]}, section=section
	))
	shared.opts.add_option("token_merging_stride_x", shared.OptionInfo(
		2, "Token Merging - Stride X",
		gr.Slider, {"minimum": 2, "maximum": 8, "step": 2}, section=section
	))
	shared.opts.add_option("token_merging_stride_y", shared.OptionInfo(
		2, "Token Merging - Stride Y",
		gr.Slider, {"minimum": 2, "maximum": 8, "step": 2}, section=section
	))
	shared.opts.add_option('token_merging_use_rand', shared.OptionInfo(
		True, 'Token Merging - Use random perturbations', section=section
	))
	shared.opts.add_option('token_merging_merge_attn', shared.OptionInfo(
		True, 'Token Merging - Merge attention', section=section
	))
	shared.opts.add_option('token_merging_merge_crossattn', shared.OptionInfo(
		False, 'Token Merging - Merge cross-attention', section=section
	))
	shared.opts.add_option('token_merging_merge_mlp', shared.OptionInfo(
		False, 'Token Merging - Merge mlp layers', section=section
	))

script_callbacks.on_model_loaded(on_model_loaded)
script_callbacks.on_ui_settings(on_ui_settings)