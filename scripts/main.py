import datetime
import gradio as gr
import os.path as osp
from pathlib import Path
from typing import cast, Literal

import modules.scripts as scripts
from modules import script_callbacks, shared
from modules.paths_internal import data_path

from liveportrait.utils.helper import load_description
from liveportrait.config.argument_config import ArgumentConfig
from liveportrait.config.crop_config import CropConfig
from liveportrait.config.inference_config import InferenceConfig
from liveportrait.gradio_pipeline import GradioPipeline, GradioPipelineAnimal

from internal_liveportrait.utils import is_valid_cuda_version, isMacOS


repo_root = Path(__file__).parent.parent

gradio_pipeline = None
gradio_pipeline_animal = None


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Live Portrait"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def on_ui_tabs():
    def init_gradio_pipeline():
        global gradio_pipeline, gradio_pipeline_animal
        if not gradio_pipeline:
            output_dir = osp.join(data_path, "outputs", "live-portrait", f"{datetime.date.today()}")
            gradio_pipeline_animal = None
            
            crop_model = cast(
                Literal['insightface', 'mediapipe', 'facealignment'],
                cast(str, shared.opts.data.get("live_portrait_human_face_detector", 'insightface')).lower()
            )

            face_alignment_detector = cast(
                Literal['blazeface', 'blazeface_back_camera', 'sfd'],
                cast(str, shared.opts.data.get("live_portrait_face_alignment_detector", 'blazeface_back_camera')).lower().replace(' ', '_')
            )

            flag_do_torch_compile = cast(
                bool,
                shared.opts.data.get("live_portrait_flag_do_torch_compile", False)
            )
            
            gradio_pipeline = GradioPipeline(
                inference_cfg=InferenceConfig(
                    flag_do_torch_compile=flag_do_torch_compile
                ),
                crop_cfg=CropConfig(
                    model=crop_model,
                    face_alignment_detector=face_alignment_detector
                ),
                args=ArgumentConfig(
                    output_dir=output_dir
                )
            )
        return gradio_pipeline
    
    def init_gradio_pipeline_animal():
        global gradio_pipeline, gradio_pipeline_animal
        if not gradio_pipeline_animal:
            output_dir = osp.join(data_path, "outputs", "live-portrait", f"{datetime.date.today()}")
            gradio_pipeline = None

            flag_do_torch_compile = cast(
                bool,
                shared.opts.data.get("live_portrait_flag_do_torch_compile", False)
            )

            gradio_pipeline_animal = GradioPipelineAnimal(
                inference_cfg=InferenceConfig(
                    flag_do_torch_compile=flag_do_torch_compile
                ),
                crop_cfg=CropConfig(),
                args=ArgumentConfig(
                    output_dir=output_dir
                )
            )
        return gradio_pipeline_animal

    def gpu_wrapped_execute_video(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        return pipeline.execute_video(*args, **kwargs)

    def gpu_wrapped_init_retargeting_image(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        source_eye_ratio, source_lip_ratio = pipeline.init_retargeting_image(*args, **kwargs)
        return float(source_eye_ratio), float(source_lip_ratio)

    def gpu_wrapped_execute_image_retargeting(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        return pipeline.execute_image_retargeting(*args, **kwargs)

    def gpu_wrapped_execute_video_retargeting(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        return pipeline.execute_video_retargeting(*args, **kwargs)
    
    def gpu_wrapped_execute_video_animal(*args, **kwargs):
        pipeline = init_gradio_pipeline_animal()
        return pipeline.execute_video(*args, **kwargs)

    def reset_sliders(*args, **kwargs):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, True, True

    # assets
    title_md = repo_root / "assets/gradio/gradio_title.md"
    example_portrait_dir = repo_root / "assets/examples/source"
    example_video_dir = repo_root / "assets/examples/driving"
    data_examples_i2v = [
        [osp.join(example_portrait_dir, "s9.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
        [osp.join(example_portrait_dir, "s6.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
        [osp.join(example_portrait_dir, "s10.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
        [osp.join(example_portrait_dir, "s5.jpg"), osp.join(example_video_dir, "d18.mp4"), True, True, True, False],
        [osp.join(example_portrait_dir, "s7.jpg"), osp.join(example_video_dir, "d19.mp4"), True, True, True, False],
        [osp.join(example_portrait_dir, "s2.jpg"), osp.join(example_video_dir, "d13.mp4"), True, True, True, True],
    ]
    data_examples_v2v = [
        [osp.join(example_portrait_dir, "s13.mp4"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False, False, 3e-7],
        # [osp.join(example_portrait_dir, "s14.mp4"), osp.join(example_video_dir, "d18.mp4"), True, True, True, False, False, 3e-7],
        # [osp.join(example_portrait_dir, "s15.mp4"), osp.join(example_video_dir, "d19.mp4"), True, True, True, False, False, 3e-7],
        [osp.join(example_portrait_dir, "s18.mp4"), osp.join(example_video_dir, "d6.mp4"), True, True, True, False, False, 3e-7],
        # [osp.join(example_portrait_dir, "s19.mp4"), osp.join(example_video_dir, "d6.mp4"), True, True, True, False, False, 3e-7],
        [osp.join(example_portrait_dir, "s20.mp4"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False, False, 3e-7],
    ]
    data_examples_i2v_animal = [
        [osp.join(example_portrait_dir, "s41.jpg"), osp.join(example_video_dir, "d3.mp4"), True, False, False, False],
        [osp.join(example_portrait_dir, "s40.jpg"), osp.join(example_video_dir, "d6.mp4"), True, False, False, False],
        [osp.join(example_portrait_dir, "s25.jpg"), osp.join(example_video_dir, "d19.mp4"), True, False, False, False],
    ]
    data_examples_i2v_animal_pickle = [
        [osp.join(example_portrait_dir, "s25.jpg"), osp.join(example_video_dir, "wink.pkl"), True, False, False, False],
        [osp.join(example_portrait_dir, "s40.jpg"), osp.join(example_video_dir, "talking.pkl"), True, False, False, False],
        [osp.join(example_portrait_dir, "s41.jpg"), osp.join(example_video_dir, "aggrieved.pkl"), True, False, False, False],
    ]

    #################### interface logic ####################

    # Define components first
    retargeting_source_scale = gr.Number(minimum=1.8, maximum=3.2, value=2.5, step=0.05, label="crop scale")
    video_retargeting_source_scale = gr.Number(minimum=1.8, maximum=3.2, value=2.3, step=0.05, label="crop scale")
    driving_smooth_observation_variance_retargeting = gr.Number(value=3e-6, label="motion smooth strength", minimum=1e-11, maximum=1e-2, step=1e-8)
    eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
    lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
    video_lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
    head_pitch_slider = gr.Slider(minimum=-15.0, maximum=15.0, value=0, step=1, label="relative pitch")
    head_yaw_slider = gr.Slider(minimum=-25, maximum=25, value=0, step=1, label="relative yaw")
    head_roll_slider = gr.Slider(minimum=-15.0, maximum=15.0, value=0, step=1, label="relative roll")
    mov_x = gr.Slider(minimum=-0.19, maximum=0.19, value=0.0, step=0.01, label="x-axis movement")
    mov_y = gr.Slider(minimum=-0.19, maximum=0.19, value=0.0, step=0.01, label="y-axis movement")
    mov_z = gr.Slider(minimum=0.9, maximum=1.2, value=1.0, step=0.01, label="z-axis movement")
    lip_variation_zero = gr.Slider(minimum=-0.09, maximum=0.09, value=0, step=0.01, label="pouting")
    lip_variation_one = gr.Slider(minimum=-20.0, maximum=15.0, value=0, step=0.01, label="pursing 😐")
    lip_variation_two = gr.Slider(minimum=0.0, maximum=15.0, value=0, step=0.01, label="grin 😁")
    lip_variation_three = gr.Slider(minimum=-90.0, maximum=120.0, value=0, step=1.0, label="lip close <-> open")
    smile = gr.Slider(minimum=-0.3, maximum=1.3, value=0, step=0.01, label="smile 😄")
    wink = gr.Slider(minimum=0, maximum=39, value=0, step=0.01, label="wink 😉")
    eyebrow = gr.Slider(minimum=-30, maximum=30, value=0, step=0.01, label="eyebrow 🤨")
    eyeball_direction_x = gr.Slider(minimum=-30.0, maximum=30.0, value=0, step=0.01, label="eye gaze (horizontal) 👀")
    eyeball_direction_y = gr.Slider(minimum=-63.0, maximum=63.0, value=0, step=0.01, label="eye gaze (vertical) 🙄")
    retargeting_input_image = gr.Image(type="filepath")
    retargeting_input_video = gr.Video()
    output_image = gr.Image(type="numpy")
    output_image_paste_back = gr.Image(type="numpy")
    retargeting_output_image = gr.Image(type="numpy")
    retargeting_output_image_paste_back = gr.Image(type="numpy")
    output_video = gr.Video(autoplay=False)
    output_video_paste_back = gr.Video(autoplay=False)
    output_video_i2v = gr.Video(autoplay=False)
    output_video_concat_i2v = gr.Video(autoplay=False)

    output_image_animal = gr.Image(type="numpy")
    output_image_animal_paste_back = gr.Image(type="numpy")
    output_video_animal_i2v = gr.Video(autoplay=False)
    output_video_animal_i2v_gif = gr.Image(type="numpy")
    output_video_animal_concat_i2v = gr.Video(autoplay=False)
    
    with gr.Blocks(analytics_enabled=False) as live_portrait:
        with gr.Tab("Humans"):
            gr.HTML(load_description(title_md))

            gr.Markdown(load_description(repo_root / "assets/gradio/gradio_description_upload.md"))
            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("🖼️ Source Image") as tab_image:
                            with gr.Accordion(open=True, label="Source Image"):
                                source_image_input = gr.Image(type="filepath")
                                gr.Examples(
                                    examples=[
                                        [osp.join(example_portrait_dir, "s9.jpg")],
                                        [osp.join(example_portrait_dir, "s6.jpg")],
                                        [osp.join(example_portrait_dir, "s10.jpg")],
                                        [osp.join(example_portrait_dir, "s5.jpg")],
                                        [osp.join(example_portrait_dir, "s7.jpg")],
                                        [osp.join(example_portrait_dir, "s12.jpg")],
                                        [osp.join(example_portrait_dir, "s22.jpg")],
                                        [osp.join(example_portrait_dir, "s23.jpg")],
                                    ],
                                    inputs=[source_image_input],
                                    cache_examples=False,
                                )

                        with gr.TabItem("🎞️ Source Video") as tab_video:
                            with gr.Accordion(open=True, label="Source Video"):
                                source_video_input = gr.Video()
                                gr.Examples(
                                    examples=[
                                        [osp.join(example_portrait_dir, "s13.mp4")],
                                        # [osp.join(example_portrait_dir, "s14.mp4")],
                                        # [osp.join(example_portrait_dir, "s15.mp4")],
                                        [osp.join(example_portrait_dir, "s18.mp4")],
                                        # [osp.join(example_portrait_dir, "s19.mp4")],
                                        [osp.join(example_portrait_dir, "s20.mp4")],
                                    ],
                                    inputs=[source_video_input],
                                    cache_examples=False,
                                )

                        tab_selection = gr.Textbox(visible=False)
                        tab_image.select(lambda: "Image", None, tab_selection)
                        tab_video.select(lambda: "Video", None, tab_selection)
                    with gr.Accordion(open=True, label="Cropping Options for Source Image or Video"):
                        with gr.Row():
                            flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                            scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                            vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                            vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)

                with gr.Column():
                    with gr.Accordion(open=True, label="Driving Video"):
                        driving_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                [osp.join(example_video_dir, "d0.mp4")],
                                [osp.join(example_video_dir, "d18.mp4")],
                                [osp.join(example_video_dir, "d19.mp4")],
                                [osp.join(example_video_dir, "d14.mp4")],
                                [osp.join(example_video_dir, "d6.mp4")],
                                [osp.join(example_video_dir, "d20.mp4")],
                            ],
                            inputs=[driving_video_input],
                            cache_examples=False,
                        )
                    # with gr.Accordion(open=False, label="Animation Instructions"):
                        # gr.Markdown(load_description(repo_root / "assets/gradio/gradio_description_animation.md"))
                    with gr.Accordion(open=True, label="Cropping Options for Driving Video"):
                        with gr.Row():
                            flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                            scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8, maximum=3.2, step=0.05)
                            vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5, maximum=0.5, step=0.01)
                            vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5, maximum=0.5, step=0.01)

            with gr.Row():
                with gr.Accordion(open=True, label="Animation Options"):
                    with gr.Row():
                        flag_relative_input = gr.Checkbox(value=True, label="relative motion")
                        flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                        flag_stitching_input = gr.Checkbox(value=True, label="stitching")
                        driving_option_input = gr.Radio(['expression-friendly', 'pose-friendly'], value="expression-friendly", label="driving option (i2v)")
                        driving_multiplier = gr.Number(value=1.0, label="driving multiplier (i2v)", minimum=0.0, maximum=2.0, step=0.02)
                        flag_video_editing_head_rotation = gr.Checkbox(value=False, label="relative head rotation (v2v)")
                        driving_smooth_observation_variance = gr.Number(value=3e-7, label="motion smooth strength (v2v)", minimum=1e-11, maximum=1e-2, step=1e-8)

            gr.Markdown(load_description(repo_root / "assets/gradio/gradio_description_animate_clear.md"))
            with gr.Row():
                process_button_animation = gr.Button("🚀 Animate", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion(open=True, label="The animated video in the original image space"):
                        output_video_i2v.render()
                with gr.Column():
                    with gr.Accordion(open=True, label="The animated video"):
                        output_video_concat_i2v.render()
            with gr.Row():
                process_button_reset = gr.ClearButton([source_image_input, source_video_input, driving_video_input, output_video_i2v, output_video_concat_i2v], value="🧹 Clear")

            with gr.Row():
                # Examples
                gr.Markdown("## You could also choose the examples below by one click ⬇️")
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("🖼️ Portrait Animation"):
                        gr.Examples(
                            examples=data_examples_i2v,
                            fn=gpu_wrapped_execute_video,
                            inputs=[
                                source_image_input,
                                driving_video_input,
                                flag_relative_input,
                                flag_do_crop_input,
                                flag_remap_input,
                                flag_crop_driving_video_input,
                            ],
                            outputs=[output_image, output_image_paste_back],
                            examples_per_page=len(data_examples_i2v),
                            cache_examples=False,
                        )
                    with gr.TabItem("🎞️ Portrait Video Editing"):
                        gr.Examples(
                            examples=data_examples_v2v,
                            fn=gpu_wrapped_execute_video,
                            inputs=[
                                source_video_input,
                                driving_video_input,
                                flag_relative_input,
                                flag_do_crop_input,
                                flag_remap_input,
                                flag_crop_driving_video_input,
                                flag_video_editing_head_rotation,
                                driving_smooth_observation_variance,
                            ],
                            outputs=[output_image, output_image_paste_back],
                            examples_per_page=len(data_examples_v2v),
                            cache_examples=False,
                        )

            # Retargeting Image
            gr.Markdown(load_description(repo_root / "assets/gradio/gradio_description_retargeting.md"), visible=True)
            with gr.Row(visible=True):
                flag_do_crop_input_retargeting_image = gr.Checkbox(value=True, label="do crop (source)")
                flag_stitching_retargeting_input = gr.Checkbox(value=True, label="stitching")
                retargeting_source_scale.render()
                eye_retargeting_slider.render()
                lip_retargeting_slider.render()
            with gr.Row(visible=True):
                with gr.Column():
                    with gr.Accordion(open=True, label="Facial movement sliders"):
                        with gr.Row(visible=True):
                            head_pitch_slider.render()
                            head_yaw_slider.render()
                            head_roll_slider.render()
                        with gr.Row(visible=True):
                            mov_x.render()
                            mov_y.render()
                            mov_z.render()
                with gr.Column():
                    with gr.Accordion(open=True, label="Facial expression sliders"):
                        with gr.Row(visible=True):
                            lip_variation_zero.render()
                            lip_variation_one.render()
                            lip_variation_two.render()
                        with gr.Row(visible=True):
                            lip_variation_three.render()
                            smile.render()
                            wink.render()
                        with gr.Row(visible=True):
                            eyebrow.render()
                            eyeball_direction_x.render()
                            eyeball_direction_y.render()
            with gr.Row(visible=True):
                reset_button = gr.Button("🔄 Reset")
                reset_button.click(
                    fn=reset_sliders,
                    inputs=None,
                    outputs=[
                        head_pitch_slider, head_yaw_slider, head_roll_slider, mov_x, mov_y, mov_z,
                        lip_variation_zero, lip_variation_one, lip_variation_two, lip_variation_three, smile, wink, eyebrow, eyeball_direction_x, eyeball_direction_y,
                        retargeting_source_scale, flag_stitching_retargeting_input, flag_do_crop_input_retargeting_image
                    ]
                )
            with gr.Row(visible=True):
                with gr.Column():
                    with gr.Accordion(open=True, label="Retargeting Image Input"):
                        retargeting_input_image.render()
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s9.jpg")],
                                [osp.join(example_portrait_dir, "s6.jpg")],
                                [osp.join(example_portrait_dir, "s10.jpg")],
                                [osp.join(example_portrait_dir, "s5.jpg")],
                                [osp.join(example_portrait_dir, "s7.jpg")],
                                [osp.join(example_portrait_dir, "s12.jpg")],
                                [osp.join(example_portrait_dir, "s22.jpg")],
                                # [osp.join(example_portrait_dir, "s23.jpg")],
                                [osp.join(example_portrait_dir, "s42.jpg")]
                            ],
                            inputs=[retargeting_input_image],
                            cache_examples=False,
                        )
                with gr.Column():
                    with gr.Accordion(open=True, label="Retargeting Result"):
                        retargeting_output_image.render()
                with gr.Column():
                    with gr.Accordion(open=True, label="Paste-back Result"):
                        retargeting_output_image_paste_back.render()
            with gr.Row(visible=True):
                process_button_reset_retargeting = gr.ClearButton(
                    [
                        retargeting_input_image,
                        retargeting_output_image,
                        retargeting_output_image_paste_back,
                    ],
                    value="🧹 Clear"
                )

            # Retargeting Video
            gr.Markdown(load_description(repo_root / "assets/gradio/gradio_description_retargeting_video.md"), visible=True)
            with gr.Row(visible=True):
                flag_do_crop_input_retargeting_video = gr.Checkbox(value=True, label="do crop (source)")
                video_retargeting_source_scale.render()
                video_lip_retargeting_slider.render()
                driving_smooth_observation_variance_retargeting.render()
            with gr.Row(visible=True):
                process_button_retargeting_video = gr.Button("🍄 Retargeting Video", variant="primary")
            with gr.Row(visible=True):
                with gr.Column():
                    with gr.Accordion(open=True, label="Retargeting Video Input"):
                        retargeting_input_video.render()
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s13.mp4")],
                                # [osp.join(example_portrait_dir, "s18.mp4")],
                                [osp.join(example_portrait_dir, "s20.mp4")],
                                [osp.join(example_portrait_dir, "s29.mp4")],
                                [osp.join(example_portrait_dir, "s32.mp4")],
                            ],
                            inputs=[retargeting_input_video],
                            cache_examples=False,
                        )
                with gr.Column():
                    with gr.Accordion(open=True, label="Retargeting Result"):
                        output_video.render()
                with gr.Column():
                    with gr.Accordion(open=True, label="Paste-back Result"):
                        output_video_paste_back.render()
            with gr.Row(visible=True):
                process_button_reset_retargeting = gr.ClearButton(
                    [
                        video_lip_retargeting_slider,
                        retargeting_input_video,
                        output_video,
                        output_video_paste_back
                    ],
                    value="🧹 Clear"
                )

            # binding functions for buttons
            process_button_animation.click(
                fn=gpu_wrapped_execute_video,
                inputs=[
                    source_image_input,
                    source_video_input,
                    driving_video_input,
                    flag_relative_input,
                    flag_do_crop_input,
                    flag_remap_input,
                    flag_stitching_input,
                    driving_option_input,
                    driving_multiplier,
                    flag_crop_driving_video_input,
                    flag_video_editing_head_rotation,
                    scale,
                    vx_ratio,
                    vy_ratio,
                    scale_crop_driving_video,
                    vx_ratio_crop_driving_video,
                    vy_ratio_crop_driving_video,
                    driving_smooth_observation_variance,
                    tab_selection,
                ],
                outputs=[output_video_i2v, output_video_concat_i2v],
                show_progress='full'
            )

            retargeting_input_image.change(
                fn=gpu_wrapped_init_retargeting_image,
                inputs=[retargeting_source_scale, eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image],
                outputs=[eye_retargeting_slider, lip_retargeting_slider]
            )

            sliders = [eye_retargeting_slider, lip_retargeting_slider, head_pitch_slider, head_yaw_slider, head_roll_slider, mov_x, mov_y, mov_z, lip_variation_zero, lip_variation_one, lip_variation_two, lip_variation_three, smile, wink, eyebrow, eyeball_direction_x, eyeball_direction_y]
            for slider in sliders:
                # NOTE: gradio >= 4.0.0 may cause slow response
                slider.change(
                    fn=gpu_wrapped_execute_image_retargeting,
                    inputs=[
                        eye_retargeting_slider, lip_retargeting_slider, head_pitch_slider, head_yaw_slider, head_roll_slider, mov_x, mov_y, mov_z,
                        lip_variation_zero, lip_variation_one, lip_variation_two, lip_variation_three, smile, wink, eyebrow, eyeball_direction_x, eyeball_direction_y,
                        retargeting_input_image, retargeting_source_scale, flag_stitching_retargeting_input, flag_do_crop_input_retargeting_image
                    ],
                    outputs=[retargeting_output_image, retargeting_output_image_paste_back],
                )

            process_button_retargeting_video.click(
                fn=gpu_wrapped_execute_video_retargeting,
                inputs=[video_lip_retargeting_slider, retargeting_input_video, video_retargeting_source_scale, driving_smooth_observation_variance_retargeting, flag_do_crop_input_retargeting_video],
                outputs=[output_video, output_video_paste_back],
                show_progress='full'
            )
            
        with gr.Tab("Animals"):
            if isMacOS():
                gr.Markdown("Animal mode is not currently supported in MacOS.")
            elif not is_valid_cuda_version():
                gr.Markdown("Animal mode is not currently supported by pytorch version 2.1.x.")
            else:            
                gr.HTML(load_description(title_md))

                gr.Markdown(load_description(repo_root / "assets/gradio/gradio_description_upload_animal.md"))
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion(open=True, label="🐱 Source Animal Image"):
                            source_image_input = gr.Image(type="filepath")
                            gr.Examples(
                                examples=[
                                    [osp.join(example_portrait_dir, "s25.jpg")],
                                    [osp.join(example_portrait_dir, "s30.jpg")],
                                    [osp.join(example_portrait_dir, "s31.jpg")],
                                    [osp.join(example_portrait_dir, "s32.jpg")],
                                    [osp.join(example_portrait_dir, "s39.jpg")],
                                    [osp.join(example_portrait_dir, "s40.jpg")],
                                    [osp.join(example_portrait_dir, "s41.jpg")],
                                    [osp.join(example_portrait_dir, "s38.jpg")],
                                    [osp.join(example_portrait_dir, "s36.jpg")],
                                ],
                                inputs=[source_image_input],
                                cache_examples=False,
                            )

                        with gr.Accordion(open=True, label="Cropping Options for Source Image"):
                            with gr.Row():
                                flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                                scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                                vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                                vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)

                    with gr.Column():
                        with gr.Tabs():
                            with gr.TabItem("📁 Driving Pickle") as tab_pickle:
                                with gr.Accordion(open=True, label="Driving Pickle"):
                                    driving_video_pickle_input = gr.File()
                                    gr.Examples(
                                        examples=[
                                            [osp.join(example_video_dir, "wink.pkl")],
                                            [osp.join(example_video_dir, "shy.pkl")],
                                            [osp.join(example_video_dir, "aggrieved.pkl")],
                                            [osp.join(example_video_dir, "open_lip.pkl")],
                                            [osp.join(example_video_dir, "laugh.pkl")],
                                            [osp.join(example_video_dir, "talking.pkl")],
                                            [osp.join(example_video_dir, "shake_face.pkl")],
                                        ],
                                        inputs=[driving_video_pickle_input],
                                        cache_examples=False,
                                    )
                            with gr.TabItem("🎞️ Driving Video") as tab_video:
                                with gr.Accordion(open=True, label="Driving Video"):
                                    driving_video_input = gr.Video()
                                    gr.Examples(
                                        examples=[
                                            # [osp.join(example_video_dir, "d0.mp4")],
                                            # [osp.join(example_video_dir, "d18.mp4")],
                                            [osp.join(example_video_dir, "d19.mp4")],
                                            [osp.join(example_video_dir, "d14.mp4")],
                                            [osp.join(example_video_dir, "d6.mp4")],
                                            [osp.join(example_video_dir, "d3.mp4")],
                                        ],
                                        inputs=[driving_video_input],
                                        cache_examples=False,
                                    )

                                tab_selection = gr.Textbox(visible=False)
                                tab_pickle.select(lambda: "Pickle", None, tab_selection)
                                tab_video.select(lambda: "Video", None, tab_selection)
                        with gr.Accordion(open=True, label="Cropping Options for Driving Video"):
                            with gr.Row():
                                flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                                scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8, maximum=3.2, step=0.05)
                                vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5, maximum=0.5, step=0.01)
                                vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5, maximum=0.5, step=0.01)

                with gr.Row():
                    with gr.Accordion(open=False, label="Animation Options"):
                        with gr.Row():
                            flag_stitching = gr.Checkbox(value=False, label="stitching (not recommended)")
                            flag_remap_input = gr.Checkbox(value=False, label="paste-back (not recommended)")
                            driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0, step=0.02)

                gr.Markdown(load_description(repo_root / "assets/gradio/gradio_description_animate_clear.md"))
                with gr.Row():
                    process_button_animation = gr.Button("🚀 Animate", variant="primary")
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion(open=True, label="The animated video in the cropped image space"):
                            output_video_animal_i2v.render()
                    with gr.Column():
                        with gr.Accordion(open=True, label="The animated gif in the cropped image space"):
                            output_video_animal_i2v_gif.render()
                    with gr.Column():
                        with gr.Accordion(open=True, label="The animated video"):
                            output_video_animal_concat_i2v.render()
                with gr.Row():
                    process_button_reset = gr.ClearButton([source_image_input, driving_video_input, output_video_animal_i2v, output_video_animal_concat_i2v, output_video_animal_i2v_gif], value="🧹 Clear")

                with gr.Row():
                    # Examples
                    gr.Markdown("## You could also choose the examples below by one click ⬇️")
                with gr.Row():
                    with gr.Tabs():
                        with gr.TabItem("📁 Driving Pickle") as tab_video:
                            gr.Examples(
                                examples=data_examples_i2v_animal_pickle,
                                fn=gpu_wrapped_execute_video_animal,
                                inputs=[
                                    source_image_input,
                                    driving_video_pickle_input,
                                    flag_do_crop_input,
                                    flag_stitching,
                                    flag_remap_input,
                                    flag_crop_driving_video_input,
                                ],
                                outputs=[output_image_animal, output_image_animal_paste_back, output_video_animal_i2v_gif],
                                examples_per_page=len(data_examples_i2v_animal_pickle),
                                cache_examples=False,
                            )
                        with gr.TabItem("🎞️ Driving Video") as tab_video:
                            gr.Examples(
                                examples=data_examples_i2v_animal,
                                fn=gpu_wrapped_execute_video_animal,
                                inputs=[
                                    source_image_input,
                                    driving_video_input,
                                    flag_do_crop_input,
                                    flag_stitching,
                                    flag_remap_input,
                                    flag_crop_driving_video_input,
                                ],
                                outputs=[output_image_animal, output_image_animal_paste_back, output_video_animal_i2v_gif],
                                examples_per_page=len(data_examples_i2v_animal),
                                cache_examples=False,
                            )

                process_button_animation.click(
                    fn=gpu_wrapped_execute_video_animal,
                    inputs=[
                        source_image_input,
                        driving_video_input,
                        driving_video_pickle_input,
                        flag_do_crop_input,
                        flag_remap_input,
                        driving_multiplier,
                        flag_stitching,
                        flag_crop_driving_video_input,
                        scale,
                        vx_ratio,
                        vy_ratio,
                        scale_crop_driving_video,
                        vx_ratio_crop_driving_video,
                        vy_ratio_crop_driving_video,
                        tab_selection,
                    ],
                    outputs=[output_video_animal_i2v, output_video_animal_concat_i2v, output_video_animal_i2v_gif],
                    show_progress='full'
                )

    return [(live_portrait, "Live Portrait", "live_portrait")]


def on_ui_settings():
    section = ("live_portrait", "Live Portrait")

    shared.opts.add_option(
        "live_portrait_human_face_detector",
        shared.OptionInfo(
            default="InsightFace",
            label="Human face detector",
            component=gr.Radio,
            component_args={"choices": ["InsightFace", "MediaPipe", "FaceAlignment"]},
            section=section,
        ).needs_reload_ui(),
    )

    shared.opts.add_option(
        "live_portrait_face_alignment_detector",
        shared.OptionInfo(
            default="BlazeFace Back Camera",
            label="Face alignment detector",
            component=gr.Radio,
            component_args={"choices": ["BlazeFace", "BlazeFace Back Camera", "SFD"]},
            section=section,
        ).needs_reload_ui(),
    )

    shared.opts.add_option(
        "live_portrait_flag_do_torch_compile",
        shared.OptionInfo(
            False,
            "Enable torch.compile for faster inference",
            section=section
        ).needs_reload_ui(),
    )


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)