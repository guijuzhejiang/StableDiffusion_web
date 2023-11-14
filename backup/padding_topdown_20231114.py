# coding=utf-8
# @Time : 2023/11/14 上午9:35
# @File : padding_topdown_20231114.py
if len(sam_bg_result) > 0:
    sam_bg_tmp_png_fp = []
    top_down_space = 20
    left_right_space = 32

    target_rect = [0, 0, 0, 0]
    target_resize = [0, 0]
    # check top
    top_down_space_scale = round(top_down_space * _output_model_height / _output_final_height)
    if person_box[1] > top_down_space_scale and person_box[3] < _output_model_height - top_down_space_scale:
        # 计算高宽缩放比例
        scale = (_output_final_height - top_down_space * 2) / person_height

    elif person_box[1] <= top_down_space_scale and person_box[3] >= _output_model_height - top_down_space_scale:
        scale = 1
    else:
        scale = (_output_final_height - top_down_space) / person_height

    if person_box[1] <= top_down_space_scale or person_box[3] >= _output_model_height - top_down_space_scale:
        target_rect[1] = 0
    else:
        target_rect[1] = top_down_space

    target_rect[0] = int(person_box[0] * scale)
    target_rect[1] = int(person_box[1] * scale)
    target_rect[2] = int(person_box[2] * scale)
    target_rect[3] = int(person_box[3] * scale)

    target_resize = [target_rect[2] - target_rect[0], target_rect[3] - target_rect[1]]

    target_rect[0] = target_rect[0] + int((_output_final_width - _output_model_width) / 2)

    for idx, sam_mask_img in enumerate(sam_bg_result):
        person_img = sam_mask_img.crop(person_box)
        person_img = person_img.resize(target_resize)

        if idx == 1:
            new_canvas = Image.new("RGBA", (_output_final_width, _output_final_height), (0, 0, 0, 255))
        else:
            new_canvas = Image.new("RGBA", (_output_final_width, _output_final_height), (255, 255, 255, 0))

        new_canvas.paste(person_img, [target_rect[0], top_down_space])
        sam_bg_result[idx] = new_canvas

        # if person_box[1] <= 4 or person_box[3] >= _output_final_height - 4:
        #     new_canvas = Image.new("RGB", (_output_final_width, _output_final_height),
        #                            (255, 255, 255))
        #     new_canvas.paste(person_img, (int((512 - person_width)/2), 0))
        #
        # elif person_box[1] < top_down_space:
        #     new_canvas = Image.new("RGB", (_output_final_width, _output_final_height),
        #                            (255, 255, 255))
        #     new_y1 = top_down_space
        #     new_y2 = person_box[3] + top_down_space - person_box[1]
        #     if new_y2 > _output_final_height-top_down_space:
        #         new_y2 = _output_model_height - top_down_space
        #
        #     new_height = new_y2-new_y1
        #     new_width = int(_output_final_width/_output_final_height*new_height)
        #     person_img = person_img.resize((new_width, new_height))
        #     new_canvas.paste(person_img, (int((_output_final_width - new_width) / 2), new_y1))
        # elif _output_final_height-person_box[3]<top_down_space:
        #     new_canvas = Image.new("RGB", (_output_final_width, _output_final_height),
        #                            (255, 255, 255))
        #     new_y2 = _output_final_height - top_down_space
        #     new_y1 = person_box[1] - (person_box[3] - new_y2)
        #
        #     if new_y1 < top_down_space:
        #         new_y1 = top_down_space
        #
        #     new_height = new_y2 - new_y1
        #     new_width = int(_output_final_width / _output_final_height * new_height)
        #     person_img = person_img.resize((new_width, new_height))
        #     new_canvas.paste(person_img,
        #                      (int((_output_final_width - new_width) / 2), new_y1))
        #
        #
        #     sam_bg_result[idx] = new_canvas
        #
        # else:
        #     new_canvas = Image.new("RGB", (_output_final_width, _output_final_height),
        #                            (255, 255, 255))
        #     new_canvas.paste(sam_mask_img,
        #                      (int((_output_final_width - sam_mask_img.size[0]) / 2), 0))
        #
        #     sam_bg_result[idx] = new_canvas

        cache_fp = f"tmp/model_only_person_seg_{res_idx}_{idx}_{pic_name}{'_save' if idx == 0 else ''}.png"
        sam_bg_result[idx].save(cache_fp)
        sam_bg_tmp_png_fp.append({'name': cache_fp})
    else:
        sam_bg_tmp_png_fp_list.append(sam_bg_tmp_png_fp)
    ok_img_count += 1
    ok_res.append(sam_bg_result[2])
    ok_sam_res.append(sam_bg_result[2])