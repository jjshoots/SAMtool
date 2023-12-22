import argparse
import os

import gradio as gr
import numpy as np

from samtool.sammer import FileSeeker, Sammer


def create_app(imagedir: str, labeldir: str, annotations: str):
    with gr.Blocks() as app:
        seeker = FileSeeker(imagedir, labeldir, annotations)
        sam = Sammer(
            seeker.all_labels,
            imagedir,
            labeldir,
        )

        # hacky asynchronous update thing
        checkbox_asyncer = gr.Checkbox(value=False, visible=False, show_label=False)

        """BUILD INTERFACE"""
        # annotation tools
        with gr.Row():
            with gr.Column(scale=1):
                # annotation tools
                radio_label = gr.Radio(
                    choices=list(seeker.all_labels.keys()),
                    value=list(seeker.all_labels.keys())[0],
                    label="Label",
                )
                with gr.Row():
                    checkbox_validity = gr.Checkbox(value=True, label="Validity")
                    radio_mode = gr.Radio(
                        choices=["Normal", "Instant", "Crayon"],
                        value="Normal",
                        label="Mode",
                    )

            with gr.Column(scale=1):
                # file selection
                with gr.Row():
                    dropdown_filenumber = gr.Dropdown(
                        list(str(i) for i in range(len(seeker.all_images))),
                        label="FileNumber",
                    )
                    dropdown_filename = gr.Dropdown(
                        seeker.all_images, label="File Selection"
                    )
                    progress = gr.Textbox(show_label=False, interactive=False)
                with gr.Row():
                    button_prev_unlabelled = gr.Button(
                        value="Prev Unlabelled", variant="primary"
                    )
                    button_prev = gr.Button(value="Previous", variant="secondary")
                    button_next = gr.Button(value="Next", variant="secondary")
                    button_next_unlabelled = gr.Button(
                        value="Next Unlabelled", variant="primary"
                    )
                with gr.Row():
                    button_reset_selection = gr.Button(
                        value="Reset Selection", variant="secondary"
                    )
                    button_reset_label = gr.Button(
                        value="Reset Label", variant="secondary"
                    )
                    button_reset_all = gr.Button(value="Reset All", variant="secondary")

        # warning for crayon mode
        textbox_crayon = gr.Textbox(
            label="WARNING",
            value="""CRAYON MODE IS HIGHLY EXPERIMENTAL AND BUGGY! ACCURATE PERFORMANCE IS NOT GUARANTEED!
            That said, what you see on the right image is accurate, so play around if you must.""",
            lines=2,
            visible=False,
        )

        with gr.Row():
            # the displays for annotation
            display_partial_normal = gr.Image(interactive=False, show_label=False)
            display_partial_instant = gr.Image(
                interactive=False, show_label=False, visible=False
            )
            display_partial_crayon = gr.Image(
                interactive=True,
                show_label=False,
                tool="sketch",
                visible=False,
            )

            # the display for annotation
            display_complete = gr.Image(interactive=False, label="Complete Annotation")

        # accept the selection
        with gr.Row():
            button_accept_normal = gr.Button(
                value="Accept", variant="primary", visible=True
            )
            button_negate_normal = gr.Button(
                value="Negate", variant="secondary", visible=True
            )
            button_accept_crayon = gr.Button(
                value="Accept", variant="primary", visible=False
            )

        """DEFINE INTERFACE FUNCTIONALITY"""

        # filenumber change
        dropdown_filenumber.change(
            fn=lambda i: seeker.all_images[int(i)],
            inputs=dropdown_filenumber,
            outputs=dropdown_filename,
        )

        def surrogate_reset(filename, mode):
            """Resets everything because the filename has changed."""
            done_labels = len(os.listdir(labeldir))
            progress_string = f"{done_labels} of {len(seeker.all_images)} completed."
            filenumber = str(seeker.all_images.index(filename))
            base_image = sam.reset(filename)
            comp_image = sam.get_comp_image(filename)

            # choose which image to output to to save bandwidth
            which_base = []
            which_base.append(base_image if mode == "Normal" else None)
            which_base.append(base_image if mode == "Instant" else None)
            which_base.append(base_image if mode == "Crayon" else None)
            return (
                *which_base,
                comp_image,
                progress_string,
                filenumber,
            )

        # filename change
        dropdown_filename.change(
            fn=surrogate_reset,
            inputs=[dropdown_filename, radio_mode],
            outputs=[
                display_partial_normal,
                display_partial_instant,
                display_partial_crayon,
                display_complete,
                progress,
                dropdown_filenumber,
            ],
        )

        # file increment decrement operators
        button_prev_unlabelled.click(
            fn=lambda f: seeker.file_increment(
                ascend=False, unlabelled_only=True, filename=f
            ),
            inputs=dropdown_filename,
            outputs=dropdown_filename,
        )
        button_prev.click(
            fn=lambda f: seeker.file_increment(
                ascend=False, unlabelled_only=False, filename=f
            ),
            inputs=dropdown_filename,
            outputs=dropdown_filename,
        )
        button_next.click(
            fn=lambda f: seeker.file_increment(
                ascend=True, unlabelled_only=False, filename=f
            ),
            inputs=dropdown_filename,
            outputs=dropdown_filename,
        )
        button_next_unlabelled.click(
            fn=lambda f: seeker.file_increment(
                ascend=True, unlabelled_only=True, filename=f
            ),
            inputs=dropdown_filename,
            outputs=dropdown_filename,
        )

        def surrogate_clear_comp_mask(filename, label):
            sam.clear_comp_mask(filename, label)
            base_image = sam.reset(filename, compute_embeddings=False)
            comp_image = sam.get_comp_image(filename)
            return base_image, comp_image

        # clear the selection image
        button_reset_selection.click(
            fn=sam.clear_coords_validity_part, outputs=display_partial_normal
        )
        # clear only the labels in the complete image
        button_reset_label.click(
            fn=lambda f, l: surrogate_clear_comp_mask(filename=f, label=l),
            inputs=[dropdown_filename, radio_label],
            outputs=[display_partial_normal, display_complete],
        )
        # clear everything
        button_reset_all.click(
            fn=lambda f: surrogate_clear_comp_mask(filename=f, label=None),
            inputs=dropdown_filename,
            outputs=[display_partial_normal, display_complete],
        )

        # normal update
        def update_prediction_normal(event: gr.SelectData, validity, label):
            sam.add_coords_validity(np.array(event.index), validity)
            return sam.update_part_image(label)

        def surrogate_part_to_comp_mask(filename, label, mode, add):
            sam.part_to_comp_mask(filename, label, add=add)
            base_image = sam.reset(filename, compute_embeddings=False)
            comp_image = sam.get_comp_image(filename)
            return base_image, base_image, comp_image

        # normal mode functionality
        display_partial_normal.select(
            fn=update_prediction_normal,
            inputs=[checkbox_validity, radio_label],
            outputs=display_partial_normal,
        )
        button_accept_normal.click(
            fn=lambda x, l, m: surrogate_part_to_comp_mask(x, l, m, add=True),
            inputs=[dropdown_filename, radio_label, radio_mode],
            outputs=[display_partial_normal, display_partial_crayon, display_complete],
        )
        button_negate_normal.click(
            fn=lambda x, l, m: surrogate_part_to_comp_mask(x, l, m, add=False),
            inputs=[dropdown_filename, radio_label, radio_mode],
            outputs=[display_partial_normal, display_partial_crayon, display_complete],
        )

        # instant update
        def update_prediction_instant(
            event: gr.SelectData, filename, label, validity, toggle
        ):
            # always work in valid selection mode, and use validity to determine whether to negate
            sam.add_coords_validity(np.array(event.index), True)
            sam.update_part_image(label)
            sam.part_to_comp_mask(filename, label, add=validity)

            # toggle the async function
            return not toggle

        # instant mode functionality
        display_partial_instant.select(
            fn=update_prediction_instant,
            inputs=[
                dropdown_filename,
                radio_label,
                checkbox_validity,
                checkbox_asyncer,
            ],
            outputs=checkbox_asyncer,
        )

        # crayon update
        def crayon_update(drawing: dict, filename, label, validity):
            mask = drawing["mask"][..., 0] == 255
            sam.part_mask = mask
            sam.part_to_comp_mask(filename, label, add=validity)
            return sam.get_comp_image(filename)

        # crayon only has one button
        button_accept_crayon.click(
            fn=crayon_update,
            inputs=[
                display_partial_crayon,
                dropdown_filename,
                radio_label,
                checkbox_validity,
            ],
            outputs=display_complete,
        )

        # hacky async update
        def async_update_prediction_instant(filename):
            return sam.get_comp_image(filename)

        # async hook
        checkbox_asyncer.change(
            fn=async_update_prediction_instant,
            inputs=dropdown_filename,
            outputs=display_complete,
        )

        # whether normal, instant, or crayon mode
        def mode_change(filename, mode):
            sam.clear_coords_validity_part()

            if mode == "Normal":
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    sam.base_image,
                    None,
                    None,
                )
            elif mode == "Instant":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    None,
                    sam.base_image,
                    None,
                )
            elif mode == "Crayon":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    None,
                    None,
                    sam.base_image,
                )
            else:
                raise ValueError(f"Unknown mode {mode}.")

        # hook the mode change button
        radio_mode.change(
            fn=mode_change,
            inputs=[dropdown_filename, radio_mode],
            outputs=[
                button_accept_normal,
                button_negate_normal,
                button_accept_crayon,
                textbox_crayon,
                button_reset_selection,
                display_partial_normal,
                display_partial_instant,
                display_partial_crayon,
                display_partial_normal,
                display_partial_instant,
                display_partial_crayon,
            ],
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        prog="SAMTool Gradio",
        description="Semantic Segmentation Dataset Creation Tool powered by Segment Anything Model from Meta.",
    )
    parser.add_argument("--imagedir", required=True)
    parser.add_argument("--labeldir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--share", default=False, action="store_true")
    args = parser.parse_args()

    create_app(args.imagedir, args.labeldir, args.annotations).launch(share=args.share)
