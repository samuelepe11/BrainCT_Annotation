import os
import sys
sys.stderr = open(os.devnull, "w")

import zipfile
import tempfile
import gradio as gr
import pandas as pd
import numpy as np
import base64
import threading
import webbrowser
from PIL import Image, ImageDraw
from pathlib import Path
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from gradio_image_annotation import image_annotator
from datetime import datetime

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
    margin-top: 50px;
    padding-top: 14px;
    border-top: 1px solid var(--border-color-primary);
"""
INITIAL_STATUS = ("Carica un nuovo file ZIP per iniziare una nuova annotazione. Il file deve contenere un file DICOM "
                  "(.dcm) per ogni slice della TC cerebrale di un solo paziente.")
CSV_COLUMNS = ["slice_idx", "image_name", "height", "width", "label", "x_min", "y_min", "x_max", "y_max"]
DELTA_SLICE = 1
LABELS = ["Foro di entrata", "Proiettile", "Frammento di proiettile", "Segno di impatto osseo",
          "Probabile punto di passaggio", "Frammento osseo", "Foro di uscita"]
LABEL_COLORS = [(255, 168, 77), (92, 172, 238), (255, 99, 71), (118, 238, 118), (255, 145, 164), (186, 104, 200),
                (255, 250, 138)]

TRAJECTORY_LABELS = ["Foro di entrata", "Proiettile", "Frammento di proiettile", "Segno di impatto osseo",
                     "Probabile punto di passaggio", "Foro di uscita"]
TRAJECTORY_COLOR = (255, 0, 0)
TRAJECTORY_RADIUS = 8
TRAJECTORY_ALPHA = 80

DEFAULT_VIEW = {"rotation": 0, "flip_horizontal": False, "flip_vertical": False, "brightness": 0.0, "contrast": 1.0}

# Functions
def reset_view_settings(state):
    state["view"] = DEFAULT_VIEW.copy()
    return state


def get_view_settings(state):
    if "view" not in state:
        reset_view_settings(state)
    return state["view"]


def transform_point(x, y, width, height, view, inverse=False):
    rotation = int(view.get("rotation", 0)) % 360
    flip_horizontal = bool(view.get("flip_horizontal", False))
    flip_vertical = bool(view.get("flip_vertical", False))

    rotated_width, rotated_height = (height, width) if rotation in (90, 270) else (width, height)

    if inverse:
        if flip_horizontal:
            x = rotated_width - x
        if flip_vertical:
            y = rotated_height - y

        if rotation == 90:
            x, y = y, height - x
        elif rotation == 180:
            x, y = width - x, height - y
        elif rotation == 270:
            x, y = width - y, x
        return x, y

    if rotation == 90:
        x, y = height - y, x
    elif rotation == 180:
        x, y = width - x, height - y
    elif rotation == 270:
        x, y = y, width - x

    if flip_horizontal:
        x = rotated_width - x
    if flip_vertical:
        y = rotated_height - y
    return x, y


def transform_box(box, width, height, view, inverse=False):
    corners = [(box["xmin"], box["ymin"]), (box["xmax"], box["ymin"]),
               (box["xmin"], box["ymax"]), (box["xmax"], box["ymax"])]
    transformed = [transform_point(x, y, width, height, view, inverse=inverse) for x, y in corners]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]
    transformed_box = box.copy()
    transformed_box.update({"xmin": int(round(min(xs))), "ymin": int(round(min(ys))),
                            "xmax": int(round(max(xs))), "ymax": int(round(max(ys)))})
    return transformed_box


def apply_geometric_view(image, view):
    rotation = int(view["rotation"]) % 360
    if rotation == 90:
        image = image.transpose(Image.Transpose.ROTATE_270)
    elif rotation == 180:
        image = image.transpose(Image.Transpose.ROTATE_180)
    elif rotation == 270:
        image = image.transpose(Image.Transpose.ROTATE_90)

    if view["flip_horizontal"]:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if view["flip_vertical"]:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image


def apply_intensity_view(image, brightness, contrast):
    # Work directly on the displayed 8-bit DICOM pixels.
    # Brightness is an additive offset in gray levels; contrast is applied around mid-gray.
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr - 127.5) * float(contrast) + 127.5 + float(brightness)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_slice(slice_idx, state):
    view = get_view_settings(state)
    image = load_image(state["image_paths"][slice_idx])
    image = apply_intensity_view(image, view["brightness"], view["contrast"])
    image = draw_trajectory(image, slice_idx, state)
    return apply_geometric_view(image, view)


def update_temporary_view(annotation, state, rotation_delta=0, toggle_horizontal=False, toggle_vertical=False,
                          brightness=None, contrast=None):
    if state is None:
        raise gr.Error("Non è presente alcuna sessione di annotazione attiva.")

    current_idx = int(state["current_idx"])
    has_annotations = store_slice_in_csv(slice_idx=current_idx, annotation=annotation, state=state)
    view = get_view_settings(state)

    if rotation_delta:
        view["rotation"] = (int(view["rotation"]) + int(rotation_delta)) % 360
    if toggle_horizontal:
        view["flip_horizontal"] = not bool(view["flip_horizontal"])
    if toggle_vertical:
        view["flip_vertical"] = not bool(view["flip_vertical"])
    if brightness is not None:
        view["brightness"] = float(brightness)
    if contrast is not None:
        view["contrast"] = float(contrast)

    image = render_slice(current_idx, state)
    boxes = get_boxes_from_csv(current_idx, state)
    return (state, make_annotator_value(image, boxes),
            gr.update(value=state["csv_path"] if has_annotations else None, interactive=has_annotations, visible=True))


def rotate_left(annotation, state):
    return update_temporary_view(annotation, state, rotation_delta=-90)


def rotate_right(annotation, state):
    return update_temporary_view(annotation, state, rotation_delta=90)


def flip_horizontal(annotation, state):
    return update_temporary_view(annotation, state, toggle_horizontal=True)


def flip_vertical(annotation, state):
    return update_temporary_view(annotation, state, toggle_vertical=True)


def change_brightness(value, annotation, state):
    if state is None or value is None:
        return state, gr.skip(), gr.skip()
    value = float(value)
    view = get_view_settings(state)
    if value == float(view["brightness"]):
        return state, gr.skip(), gr.skip()
    return update_temporary_view(annotation, state, brightness=value)


def change_contrast(value, annotation, state):
    if state is None or value is None:
        return state, gr.skip(), gr.skip()
    value = float(value)
    view = get_view_settings(state)
    if value == float(view["contrast"]):
        return state, gr.skip(), gr.skip()
    return update_temporary_view(annotation, state, contrast=value)


# Functions
def avoid_clear_action(slice_number, state):
    if state is None:
        raise gr.Error("Non è presente alcuna sessione di annotazione attiva.")
    idx = int(slice_number) - 1

    # An empty box list deletes all CSV rows for this slice. The X also restores the original view.
    has_annotations = store_slice_in_csv(slice_idx=idx, annotation={"boxes": []}, state=state)
    reset_view_settings(state)
    image = render_slice(idx, state)
    return (state, make_annotator_value(image, []), gr.update(value=state["csv_path"] if has_annotations else None,
                                                              interactive=has_annotations, visible=True),
            f"## Slice {idx + 1}/{len(state['image_paths'])}\n" + "Tutte le annotazioni della slice corrente sono state rimosse.",
            gr.update(value=0.0), gr.update(value=1.0))


def change_slice(slice_number, annotation, state):
    if state is None:
        raise gr.Error("Non è presente alcuna sessione di annotazione attiva.")

    current_idx = int(state["current_idx"])

    # Validate the value entered in the slice slider/text box.
    if slice_number is None or slice_number == "":
        slice_number = current_idx + 1
    else:
        try:
            slice_number = int(float(slice_number))
        except (TypeError, ValueError):
            slice_number = current_idx + 1

    # Clamp the visible slice number to the valid range 1...N.
    slice_number = max(1, min(slice_number, len(state["image_paths"])))
    new_idx = slice_number - 1

    # The annotator currently displays this slice.
    previous_idx = current_idx

    # .change() also fires when the arrow buttons update the slider programmatically.
    # If the state already points to this slice, the arrow callback has already done all the work.
    if new_idx == previous_idx:
        return state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    # Save the boxes currently displayed before changing image.
    has_annotations = store_slice_in_csv(slice_idx=previous_idx, annotation=annotation, state=state)

    # A genuinely different slice always starts from the original, unmodified view.
    if new_idx != previous_idx:
        state["current_idx"] = new_idx
        reset_view_settings(state)

    image = render_slice(new_idx, state)

    # Retrieve this slice's previous boxes directly from the CSV.
    boxes = get_boxes_from_csv(new_idx, state)
    view = get_view_settings(state)

    return (state, make_annotator_value(image, boxes),
            gr.update(value=state["csv_path"] if has_annotations else None,
                      interactive=has_annotations, visible=True),
            f"## Slice {new_idx + 1}/{len(state['image_paths'])}\n"
            f"Annotazioni attualmente memorizzate nel CSV per questa slice: **{len(boxes)}**",
            gr.update(value=float(view["brightness"])),
            gr.update(value=float(view["contrast"])))

def csv_contains_annotations(state):
    dataframe = read_annotation_csv(state)
    return not dataframe.empty


def disable_start_button():
    return gr.update(interactive=False)


def enable_start_button(zip_file):
    if zip_file is None:
        return gr.update(interactive=False)
    return gr.update(interactive=True)


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
            if file.startswith("._") or file == ".DS_Store":
                continue
            path = Path(root) / file
            if path.suffix.lower() != ".dcm":
                continue
            try:
                dcmread(str(path), stop_before_pixels=True, defer_size=0)
            except Exception:
                print(f"Skipping non-DICOM file: {path}")
                continue
            image_paths.append(str(path))
    image_paths = sorted(image_paths)
    print("PATHS", image_paths)

    if len(image_paths) == 0:
        raise gr.Error("La cartella è vuota o non contiene immagini DICOM (.dcm). Per favore carica un file ZIP valido.")
    csv_path = os.path.join(workdir, f"annotation_session_{datetime.now().strftime('%Y%m%d%H%M%S')}_n{len(image_paths)}.csv")

    # Initially create an empty CSV containing only the column headers.
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
    state = {"workdir": workdir, "image_paths": image_paths, "csv_path": csv_path, "current_idx": 0,
             "trajectory_enabled": False, "view": DEFAULT_VIEW.copy()}
    first_img = render_slice(0, state)
    return (state, gr.update(visible=False), gr.update(visible=True), gr.update(minimum=1, maximum=len(image_paths), value=1, step=1, visible=True),
            gr.update(value=make_annotator_value(first_img, []), visible=True), gr.update(value=None, visible=True, interactive=False), gr.update(visible=True),
            f"## Slice 1/{len(image_paths)}\n" + f"Sono state caricate **{len(image_paths)} slice**.")


def get_boxes_from_csv(slice_idx, state):
    dataframe = read_annotation_csv(state)
    if dataframe.empty:
        return []
    slice_rows = dataframe[dataframe["slice_idx"].astype(int) == int(slice_idx)]
    height, width = get_image_dimensions(state["image_paths"][int(slice_idx)])
    view = get_view_settings(state)
    boxes = []
    for _, row in slice_rows.iterrows():
        label = str(row["label"])
        color = LABEL_COLORS[LABELS.index(label)]
        box = {"label": label, "xmin": int(round(float(row["x_min"]))), "ymin": int(round(float(row["y_min"]))),
               "xmax": int(round(float(row["x_max"]))), "ymax": int(round(float(row["y_max"]))), "color": color}
        boxes.append(transform_box(box, width, height, view, inverse=False))
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
    state, annotator_value, download_update, status_text, brightness_update, contrast_update = change_slice(slice_number=new_idx + 1, annotation=annotation, state=state)
    return state, gr.update(value=new_idx + 1), annotator_value, download_update, status_text, brightness_update, contrast_update


def open_app_in_dark_mode():
    webbrowser.open("http://127.0.0.1:7860/?__theme=dark")


def previous_slice(annotation, state):
    current_idx = int(state["current_idx"])
    new_idx = max(0, current_idx - DELTA_SLICE)
    state, annotator_value, download_update, status_text, brightness_update, contrast_update = change_slice(slice_number=new_idx + 1, annotation=annotation,
                                                                                                              state=state)
    return state, gr.update(value=new_idx + 1), annotator_value, download_update, status_text, brightness_update, contrast_update


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
            gr.update(interactive=False), INITIAL_STATUS, gr.update(value=0.0), gr.update(value=1.0))


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
    view = get_view_settings(state)
    boxes = [transform_box(box, width, height, view, inverse=True) for box in boxes]
    new_rows = []
    for box in boxes:
        new_rows.append({"slice_idx": slice_idx, "image_name": Path(image_path).name, "height": height, "width": width,
                         "label": box["label"], "x_min": box["xmin"], "y_min": box["ymin"], "x_max": box["xmax"],
                         "y_max": box["ymax"]})
    if new_rows:
        new_dataframe = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
        dataframe = pd.concat([dataframe, new_dataframe], ignore_index=True)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=["slice_idx", "label"], kind="stable").reset_index(drop=True)

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


def get_trajectory_landmarks(state):
    dataframe = read_annotation_csv(state)
    if dataframe.empty:
        return []
    dataframe = dataframe[dataframe["label"].isin(TRAJECTORY_LABELS)].copy()
    if dataframe.empty:
        return []
    dataframe["center_x"] = (dataframe["x_min"].astype(float) + dataframe["x_max"].astype(float)) / 2
    dataframe["center_y"] = (dataframe["y_min"].astype(float) + dataframe["y_max"].astype(float)) / 2
    landmarks = []
    for label in TRAJECTORY_LABELS:
        label_dataframe = dataframe[dataframe["label"] == label].copy()
        if label_dataframe.empty:
            continue
        label_dataframe = label_dataframe.sort_values(by="slice_idx")
        label_dataframe["group"] = (label_dataframe["slice_idx"].astype(int).diff().fillna(1).abs() > 1).cumsum()
        for _, group in label_dataframe.groupby("group"):
            landmarks.append({"label": label, "slice_idx": float(group["slice_idx"].astype(float).mean()),
                              "x": float(group["center_x"].mean()),
                              "y": float(group["center_y"].mean())})
    landmarks = sorted(landmarks, key=lambda point: point["slice_idx"])
    entry_points = [point for point in landmarks if point["label"] == "Foro di entrata"]
    exit_points = [point for point in landmarks if point["label"] == "Foro di uscita"]
    if entry_points:
        entry = entry_points[0]
        if abs(entry["slice_idx"] - landmarks[-1]["slice_idx"]) < abs(entry["slice_idx"] - landmarks[0]["slice_idx"]):
            landmarks.reverse()
    elif exit_points:
        exit_point = exit_points[0]
        if abs(exit_point["slice_idx"] - landmarks[0]["slice_idx"]) < abs(exit_point["slice_idx"] - landmarks[-1]["slice_idx"]):
            landmarks.reverse()
    return landmarks


def get_trajectory_position(slice_idx, landmarks):
    if len(landmarks) < 2:
        return None
    for i in range(len(landmarks) - 1):
        point_1 = landmarks[i]
        point_2 = landmarks[i + 1]
        z_1 = point_1["slice_idx"]
        z_2 = point_2["slice_idx"]
        if min(z_1, z_2) <= slice_idx <= max(z_1, z_2):
            if z_1 == z_2:
                return point_1["x"], point_1["y"]
            interpolation = (slice_idx - z_1) / (z_2 - z_1)
            x = point_1["x"] + interpolation * (point_2["x"] - point_1["x"])
            y = point_1["y"] + interpolation * (point_2["y"] - point_1["y"])
            return x, y
    return None


def draw_trajectory(image, slice_idx, state):
    if not state.get("trajectory_enabled", False):
        return image
    landmarks = get_trajectory_landmarks(state)
    if len(landmarks) < 2:
        return image
    position = get_trajectory_position(slice_idx, landmarks)
    if position is None:
        return image
    x, y = position
    x = int(round(x))
    y = int(round(y))
    is_landmark = any(abs(point["slice_idx"] - slice_idx) < 0.5 for point in landmarks)
    radius = TRAJECTORY_RADIUS + 3 if is_landmark else TRAJECTORY_RADIUS
    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Semi-transparent trajectory marker
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=(*TRAJECTORY_COLOR, TRAJECTORY_ALPHA))
    image_rgba = Image.alpha_composite(image_rgba, overlay)
    return image_rgba.convert("RGB")


def trace_trajectory(annotation, state):
    if state is None:
        raise gr.Error("Non è presente alcuna sessione di annotazione attiva.")
    current_idx = int(state["current_idx"])
    has_annotations = store_slice_in_csv(slice_idx=current_idx, annotation=annotation, state=state)
    landmarks = get_trajectory_landmarks(state)
    if len(landmarks) < 2:
        state["trajectory_enabled"] = False
        gr.Warning("Per tracciare il percorso sono necessarie almeno due annotazioni tra foro di entrata, segno di impatto osseo, proiettile, probabile punto di passaggio e foro di uscita.")
        image = render_slice(current_idx, state)
        boxes = get_boxes_from_csv(current_idx, state)
        return state, make_annotator_value(image, boxes), gr.update(value=state["csv_path"] if has_annotations else None,
                                                                    interactive=has_annotations, visible=True)
    state["trajectory_enabled"] = True
    image = render_slice(current_idx, state)
    boxes = get_boxes_from_csv(current_idx, state)
    return state, make_annotator_value(image, boxes), gr.update(value=state["csv_path"] if has_annotations else None,
                                                                interactive=has_annotations, visible=True)


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
        start_btn = gr.Button("Inizia", icon="icons/next.png", interactive=False, variant="primary")

    with gr.Column(visible=False) as annotation_area:
        status = gr.Markdown(INITIAL_STATUS)
        with gr.Row():
            with gr.Column(min_width=500):
                annotator = image_annotator(label_list=LABELS, label_colors=LABEL_COLORS.copy(), show_label=False, visible=False,
                                            elem_id="ct-annotator")
            with gr.Column():
                with gr.Row():
                    rotate_left_btn = gr.Button("Ruota SX", icon="icons/rotate_ccw.png")
                    rotate_right_btn = gr.Button("Ruota DX", icon="icons/rotate_cw.png")
                    flip_horizontal_btn = gr.Button("Specchia", icon="icons/flip_horiz.png")
                    flip_vertical_btn = gr.Button("Ribalta", icon="icons/flip_vert.png")
                with gr.Row():
                    brightness_slider = gr.Slider(minimum=-120, maximum=120, value=0, step=5, label="Luminosità")
                    contrast_slider = gr.Slider(minimum=0.25, maximum=3.0, value=1.0, step=0.05, label="Contrasto")
                with gr.Row():
                    backward_btn = gr.Button(f"Indietro di {DELTA_SLICE} slice", icon="icons/back.png")
                    forward_btn = gr.Button(f"Avanti di {DELTA_SLICE} slice", icon="icons/next.png")
                slice_slider = gr.Slider(minimum=1, maximum=2, value=1, step=1, label="Slice", visible=False)
                with gr.Row():
                    trajectory_btn = gr.Button("Traccia percorso approssimato", icon="icons/trajectory.png")
                    gr.Markdown("*Questa modalità di visualizzazione è consigliata solo in presenza di un singolo proiettile, nel cui percorso non sono attese biforcazioni.*")
        download_btn = gr.DownloadButton(label="Scarica report CSV", value=None, visible=False, interactive=False,
                                         icon="icons/download.png", variant="primary")
        with gr.Column(elem_id="restart-section"):
            gr.Markdown("""
                        ### Nuova annotazione
                        Utilizza il pulsante seguente solamente dopo aver scaricato il report CSV.
                        """)
            restart_btn = gr.Button("Annota un altro paziente", visible=False, icon="icons/back_arrow.png")

    zip_upload.upload(fn=enable_start_button, inputs=zip_upload, outputs=start_btn)
    zip_upload.clear(fn=disable_start_button, inputs=None, outputs=start_btn)
    start_btn.click(fn=extract_zip, inputs=zip_upload, outputs=[state, upload_area, annotation_area, slice_slider,
                                                                annotator, download_btn, restart_btn, status])
    annotator.change(fn=synchronize_current_slice, inputs=[annotator, state], outputs=[state, download_btn])
    annotator.clear(fn=avoid_clear_action, inputs=[slice_slider, state], outputs=[state, annotator, download_btn, status,
                                                                                 brightness_slider, contrast_slider])
    slice_slider.change(fn=change_slice, inputs=[slice_slider, annotator, state], outputs=[state, annotator, download_btn,
                                                                                          status, brightness_slider, contrast_slider], preprocess=False)
    backward_btn.click(fn=previous_slice, inputs=[annotator, state], outputs=[state, slice_slider, annotator,
                                                                              download_btn, status, brightness_slider, contrast_slider])
    forward_btn.click(fn=next_slice, inputs=[annotator, state], outputs=[state, slice_slider, annotator, download_btn,
                                                                         status, brightness_slider, contrast_slider])
    rotate_left_btn.click(fn=rotate_left, inputs=[annotator, state], outputs=[state, annotator, download_btn])
    rotate_right_btn.click(fn=rotate_right, inputs=[annotator, state], outputs=[state, annotator, download_btn])
    flip_horizontal_btn.click(fn=flip_horizontal, inputs=[annotator, state], outputs=[state, annotator, download_btn])
    flip_vertical_btn.click(fn=flip_vertical, inputs=[annotator, state], outputs=[state, annotator, download_btn])
    brightness_slider.change(fn=change_brightness, inputs=[brightness_slider, annotator, state], outputs=[state, annotator, download_btn])
    contrast_slider.change(fn=change_contrast, inputs=[contrast_slider, annotator, state], outputs=[state, annotator, download_btn])
    restart_btn.click(fn=reset_app, inputs=None, outputs=[state, upload_area, annotation_area, zip_upload, restart_btn,
                                                          start_btn, status, brightness_slider, contrast_slider])
    trajectory_btn.click(fn=trace_trajectory, inputs=[annotator, state], outputs=[state, annotator, download_btn])

    print("""
    ===========================================================================
                            Brain CT Annotation Platform

    The application is running and should open automatically in your browser.
    If it does not, open visit http://127.0.0.1:7860?__theme=dark

    Press Ctrl+C to stop the server when finished.
    ===========================================================================
    """)


if __name__ == "__main__":
    threading.Timer(1.5, open_app_in_dark_mode).start()
    demo.launch(share=False, css=ANNOTATOR_CSS, inbrowser=False, show_error=False, quiet=True)
