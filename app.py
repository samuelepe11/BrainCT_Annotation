import os
import zipfile
import tempfile
import gradio as gr
import pandas as pd
import numpy as np
import base64
from PIL import Image
from pathlib import Path
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from gradio_image_annotation import image_annotator

# Global variables
ANNOTATOR_CSS = """
/* Remove rotation and flipping controls from the annotator toolbar */
#ct-annotator button[title*="rotate" i],
#ct-annotator button[aria-label*="rotate" i],
#ct-annotator button[title*="rotation" i],
#ct-annotator button[aria-label*="rotation" i],
#ct-annotator button[title*="flip" i],
#ct-annotator button[aria-label*="flip" i],
#ct-annotator button[title*="mirror" i],
#ct-annotator button[aria-label*="mirror" i] {display: none !important}
/* Keep the restart button away from the download button */
#restart-section {
    margin-top: 150px;
    padding-top: 24px;
    border-top: 1px solid var(--border-color-primary);
"""
INITIAL_STATUS = ("Carica un nuovo file ZIP per iniziare una nuova annotazione. Il file deve contenere un file DICOM "
                  "(.dcm) per ogni slice della TC cerebrale di un solo paziente.")
CSV_COLUMNS = ["slice_idx", "image_name", "height", "width", "label", "x_min", "y_min", "x_max", "y_max"]
DELTA_SLICE = 1


# Functions
def avoid_clear_action(idx, state):
    if state is None:
        raise gr.Error("Non è presente alcuna sessione di annotazione attiva.")
    idx = int(idx)
    image_path = state["image_paths"][idx]

    # An empty box list deletes all CSV rows for this slice.
    has_annotations = store_slice_in_csv(slice_idx=idx, annotation={"boxes": []}, state=state)
    image = load_image(image_path)
    return (state, make_annotator_value(image, []), gr.update(value=state["csv_path"] if has_annotations else None,
                                                              interactive=has_annotations, visible=True),
            f"## Slice {idx + 1}/{len(state['image_paths'])}\n" + "Tutte le annotazioni della slice corrente sono state rimosse.")


def change_slice(new_idx, annotation, state):
    if state is None:
        raise gr.Error("Non è presente alcuna sessione di annotazione attiva.")
    new_idx = int(new_idx)
    if new_idx < 0 or new_idx >= len(state["image_paths"]):
        raise gr.Error("Indice della slice non valido.")

    # The annotator currently displays this slice.
    previous_idx = int(state["current_idx"])

    # Save the boxes currently displayed before changing image.
    has_annotations = store_slice_in_csv(slice_idx=previous_idx, annotation=annotation, state=state)

    # Load the newly selected slice.
    image_path = state["image_paths"][new_idx]
    image = load_image(image_path)

    # Retrieve this slice's previous boxes directly from the CSV.
    boxes = get_boxes_from_csv(new_idx, state)
    state["current_idx"] = new_idx
    return (state, make_annotator_value(image, boxes), gr.update(value=state["csv_path"] if has_annotations else None,
                                                                interactive=has_annotations, visible=True),
            f"## Slice {new_idx + 1}/{len(state['image_paths'])}\n" + f"Annotazioni attualmente memorizzate nel CSV per questa slice: **{len(boxes)}**")


def csv_contains_annotations(state):
    dataframe = read_annotation_csv(state)
    return not dataframe.empty


def extract_zip(zip_file):
    gr.Info("L'estrazione del file ZIP ed il caricamento delle immagini potrebbero richiedere qualche secondo...")
    if zip_file is None:
        raise gr.Error("Per favore carica un file ZIP contenente le immagini da annotare.")

    workdir = tempfile.mkdtemp(prefix="annotation_app_")
    extract_dir = os.path.join(workdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file.name, "r") as zf:
        zf.extractall(extract_dir)

    image_paths = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            path = Path(root) / file
            if path.suffix.lower() == ".dcm":
                image_paths.append(str(path))
    image_paths = sorted(image_paths)
    print("PATHS", image_paths)

    if len(image_paths) == 0:
        raise gr.Error("La cartella è vuota o non contiene immagini DICOM (.dcm). Per favore carica un file ZIP valido.")
    case_name = Path(zip_file.name).stem
    csv_path = os.path.join(workdir, f"{case_name}_annotations.csv")

    # Initially create an empty CSV containing only the column headers.
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
    state = {"workdir": workdir, "image_paths": image_paths, "csv_path": csv_path, "current_idx": 0 }
    first_img = load_image(image_paths[0])
    return (
        state,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(minimum=0, maximum=len(image_paths) - 1, value=0, step=1, visible=True),
        gr.update(value=make_annotator_value(first_img, []), visible=True),
        gr.update(value=None, visible=True, interactive=False),
        gr.update(visible=True),
        f"## Slice 1/{len(image_paths)}\n" + f"Sono state caricate **{len(image_paths)} slice**.",
    )


def get_boxes_from_csv(slice_idx, state):
    dataframe = read_annotation_csv(state)
    if dataframe.empty:
        return []
    slice_rows = dataframe[dataframe["slice_idx"].astype(int) == int(slice_idx)]
    boxes = []
    for _, row in slice_rows.iterrows():
        boxes.append({"label": str(row["label"]), "xmin": int(round(float(row["x_min"]))), "ymin": int(round(float(row["y_min"]))),
                      "xmax": int(round(float(row["x_max"]))), "ymax": int(round(float(row["y_max"])))})
    return boxes


def get_image_dimensions(path: str) -> tuple[int, int]:
    ds = dcmread(path, stop_before_pixels=True)
    height = int(ds.Rows)
    width = int(ds.Columns)
    return height, width


def load_image(path):
    ds = dcmread(path)
    arr = apply_modality_lut(ds.pixel_array, ds)
    arr = apply_voi_lut(arr, ds)
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr /= arr.max() + 1e-8
    arr = (255 * arr).astype(np.uint8)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        arr = 255 - arr
    return Image.fromarray(arr)


def make_annotator_value(image, boxes):
    return {"image": image, "boxes": boxes}


def next_slice(annotation, state):
    current_idx = int(state["current_idx"])
    last_idx = len(state["image_paths"]) - 1
    new_idx = min(last_idx, current_idx + DELTA_SLICE)
    state, annotator_value, download_update, status_text = change_slice(new_idx=new_idx, annotation=annotation, state=state)
    return state, gr.update(value=new_idx), annotator_value, download_update, status_text


def previous_slice(annotation, state):
    current_idx = int(state["current_idx"])
    new_idx = max(0, current_idx - DELTA_SLICE)
    state, annotator_value, download_update, status_text = change_slice(new_idx=new_idx, annotation=annotation,
                                                                        state=state)
    return state, gr.update(value=new_idx), annotator_value, download_update, status_text,


def read_annotation_csv(state):
    csv_path = state["csv_path"]
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        dataframe = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        dataframe = pd.DataFrame(columns=CSV_COLUMNS)
    return dataframe


def reset_app():
    return (None, gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(visible=False),
            INITIAL_STATUS)


def store_slice_in_csv(slice_idx, annotation, state):
    slice_idx = int(slice_idx)
    dataframe = read_annotation_csv(state)

    # Remove previous annotations belonging to this slice.
    if not dataframe.empty:
        dataframe = dataframe[dataframe["slice_idx"].astype(int) != slice_idx].copy()
    boxes = []
    if isinstance(annotation, dict):
        boxes = annotation.get("boxes", []) or []
    image_path = state["image_paths"][slice_idx]
    height, width = get_image_dimensions(image_path)
    new_rows = []
    for box in boxes:
        new_rows.append({"slice_idx": slice_idx, "image_name": Path(image_path).name, "height": height, "width": width,
                         "label": box["label"], "x_min": box["xmin"], "y_min": box["ymin"], "x_max": box["xmax"],
                         "y_max": box["ymax"]})
    if new_rows:
        new_dataframe = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
        dataframe = pd.concat([dataframe, new_dataframe], ignore_index=True)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(
            by=["slice_idx", "label"],
            kind="stable",
        ).reset_index(drop=True)

    # Write to a temporary file first, then replace the previous CSV
    temporary_path = state["csv_path"] + ".tmp"
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, state["csv_path"])
    return not dataframe.empty


def synchronize_current_slice(annotation, state):
    if state is None:
        return (state, gr.update(value=None, interactive=False))
    current_idx = int(state["current_idx"])
    has_annotations = store_slice_in_csv(slice_idx=current_idx, annotation=annotation, state=state)
    return state, gr.update(value=state["csv_path"] if has_annotations else None, interactive=has_annotations,
                            visible=True)


with gr.Blocks() as demo:
    img_path = Path(__file__).resolve().parent / "icons" / "ct.png"
    gr.HTML(
        f"""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 18px;">
            <img src=data:image/png;base64,{base64.b64encode(img_path.read_bytes()).decode("utf-8")} alt="Brain CT icon"
                style="width: 52px; height: 52px; object-fit: contain;">
            <h1 style="margin: 0;">Brain CT Annotation Platform</h1>
        </div>
        """
    )
    state = gr.State(None)

    with gr.Column(visible=True) as upload_area:
        zip_upload = gr.File(label="Carica file ZIP", file_types=[".zip"])
        start_btn = gr.Button("Inizia", icon="icons/next.png")

    with gr.Column(visible=False) as annotation_area:
        status = gr.Markdown(INITIAL_STATUS)
        with gr.Row():
            with gr.Column(min_width=500):
                annotator = image_annotator(label_list=["Proiettile", "Frammento di proiettile", "Frammento osseo", "Emorragia",
                                                        "Altro"], show_label=False, visible=False, elem_id="ct-annotator")
            with gr.Column():
                with gr.Row():
                    backward_btn = gr.Button(f"Indietro di {DELTA_SLICE} slice", icon="icons/back.png")
                    forward_btn = gr.Button(f"Avanti di {DELTA_SLICE} slice", icon="icons/next.png")
                slice_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Slice", visible=False)
                download_btn = gr.DownloadButton(label="Scarica report CSV", value=None, visible=False, interactive=False,
                                                 icon="icons/download.png")
                with gr.Column(elem_id="restart-section"):
                    gr.Markdown("""
                                ### Nuova annotazione
                                Utilizza il pulsante seguente solamente dopo aver scaricato il report CSV.
                                """)
                    restart_btn = gr.Button("Annota un altro paziente", visible=False)

    start_btn.click(fn=extract_zip, inputs=zip_upload, outputs=[state, upload_area, annotation_area, slice_slider,
                                                                annotator, download_btn, restart_btn, status])
    slice_slider.input(fn=change_slice, inputs=[slice_slider, annotator, state], outputs=[state, annotator, download_btn,
                                                                                          status])
    annotator.change(fn=synchronize_current_slice, inputs=[annotator, state], outputs=[state, download_btn])
    annotator.clear(fn=avoid_clear_action, inputs=[slice_slider, state], outputs=[state, annotator, download_btn, status])
    backward_btn.click(fn=previous_slice, inputs=[annotator, state], outputs=[state, slice_slider, annotator,
                                                                              download_btn, status])
    forward_btn.click(fn=next_slice, inputs=[annotator, state], outputs=[state, slice_slider, annotator, download_btn,
                                                                         status])

    print("""
    ===========================================================================
                            Brain CT Annotation Platform

    The application is running and should open automatically in your browser.
    If it does not, open visit http://127.0.0.1:7860

    Press Ctrl+C to stop the server when finished.
    ===========================================================================
    """)


if __name__ == "__main__":
    demo.launch(share=False, css=ANNOTATOR_CSS, inbrowser=True)
