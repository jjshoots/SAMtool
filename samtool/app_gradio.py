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

        """BUILD INTERFACE"""
        # annotation tools
        with gr.Row():
            with gr.Column(scale=1):
                # status
                progress = gr.Textbox(show_label=False, interactive=False)
                warnings = gr.Textbox(show_label=False, interactive=False)

            with gr.Column(scale=1):
                # annotation tools
                radio_label = gr.Radio(
                    choices=list(seeker.all_labels.keys()),
                    value=list(seeker.all_labels.keys())[0],
                    label="Label",
                )
                with gr.Row():
                    checkbox_validity = gr.Checkbox(value=True, label="Validity")
                    checkbox_instant = gr.Checkbox(value=False, label="Instant Mode")

            with gr.Column(scale=1):
                # file selection
                dropdown_filename = gr.Dropdown(
                    seeker.all_images, label="File Selection"
                )
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

        with gr.Row():
            # the displays for annotation
            display_partial_normal = gr.Image(
                interactive=False, show_label=False
            ).style(width=480)
            display_partial_instant = gr.Image(
                interactive=False, show_label=False, visible=False
            ).style(width=480)

            display_complete = gr.Image(
                interactive=False, label="Complete Annotation"
            ).style(width=480)

        # approve the selection
        with gr.Row():
            button_accept = gr.Button(value="Approve", variant="primary", visible=True)
            button_negate = gr.Button(value="Negate", variant="secondary", visible=True)

        """DEFINE INTERFACE FUNCTIONALITY"""

        def surrogate_reset(filename):
            """Resets everything because the filename has changed."""
            done_labels = len(os.listdir(labeldir))
            progress_string = f"{done_labels} of {len(seeker.all_images)} completed."
            base_image, comp_image = sam.reset(filename)
            return base_image, base_image, comp_image, progress_string

        # filename change
        dropdown_filename.change(
            fn=surrogate_reset,
            inputs=dropdown_filename,
            outputs=[
                display_partial_normal,
                display_partial_instant,
                display_complete,
                progress,
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

        # clear the selection image
        button_reset_selection.click(
            fn=sam.clear_coords_validity, outputs=display_partial_normal
        )
        # clear only the labels in the complete image
        button_reset_label.click(
            fn=lambda f, l: sam.clear_comp_mask(filename=f, label=l),
            inputs=[dropdown_filename, radio_label],
            outputs=[display_partial_normal, display_complete],
        )
        # clear everything
        button_reset_all.click(
            fn=lambda f: sam.clear_comp_mask(filename=f, label=None),
            inputs=dropdown_filename,
            outputs=[display_partial_normal, display_complete],
        )

        # normal update
        def update_prediction_normal(event: gr.SelectData, filename, validity, label):
            sam.add_coords_validity(np.array(event.index), validity)
            return sam.update_part_image(label)

        # normal mode functionality
        display_partial_normal.select(
            fn=update_prediction_normal,
            inputs=[dropdown_filename, checkbox_validity, radio_label],
            outputs=display_partial_normal,
        )
        button_accept.click(
            fn=lambda x, k: sam.part_to_comp_mask(x, k, add=True),
            inputs=[dropdown_filename, radio_label],
            outputs=[display_partial_normal, display_complete],
        )
        button_negate.click(
            fn=lambda x, k: sam.part_to_comp_mask(x, k, add=False),
            inputs=[dropdown_filename, radio_label],
            outputs=[display_partial_normal, display_complete],
        )

        # instant update
        def update_prediction_instant(event: gr.SelectData, filename, validity, label):
            # always work in valid selection mode, and use validity to determine whether to negate
            sam.add_coords_validity(np.array(event.index), True)
            sam.update_part_image(label)
            base_image, comp_image = sam.part_to_comp_mask(
                filename, label, add=validity
            )
            return comp_image

        # instant mode functionality
        display_partial_instant.select(
            fn=update_prediction_instant,
            inputs=[dropdown_filename, checkbox_validity, radio_label],
            outputs=display_complete,
        )

        # whether slow or instant mode
        def mode_change(instant_mode):
            if instant_mode:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                )

        # hook the mode change button
        checkbox_instant.change(
            fn=mode_change,
            inputs=checkbox_instant,
            outputs=[
                button_accept,
                button_negate,
                button_reset_selection,
                display_partial_normal,
                display_partial_instant,
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
