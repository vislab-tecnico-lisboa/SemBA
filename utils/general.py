# =============================================================================
# General Functions for SemBA x MS-Fovea
# =============================================================================

# =============================================================================
# 1. Numerical, rendering, and filesystem dependencies
# =============================================================================

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imageio
import os
import re


# =============================================================================
# 2. Run directories and static attention maps
# =============================================================================

def create_dir(base_folder):
    """Create the next run_N directory without reusing an existing run number.

    base_folder is the parent output directory, created if missing. Only
    directories named run_N contribute to numbering. Returns the new path
    as a string and prints its location; existing runs remain untouched.
    """

    os.makedirs(base_folder, exist_ok=True)

    # Collect numeric run suffixes; unrelated directory names do not count.
    pattern = re.compile(r"run_(\d+)$")
    existing_k = []

    for name in os.listdir(base_folder):
        path = os.path.join(base_folder, name)

        if os.path.isdir(path):
            match = pattern.match(name)
            if match:
                existing_k.append(int(match.group(1)))

    # Ignore unrelated folders and continue after the highest existing run number.
    next_k = max(existing_k, default=0) + 1

    new_folder = os.path.join(base_folder, f"run_{next_k}")
    os.makedirs(new_folder)

    print(f"\nResults for this run ({next_k}) are saved in: {new_folder}\n")

    return new_folder


def save_map(map, path, idx):
    """Save a grid heatmap as map_fix{idx}.png, without axes or margins.

    map is a two-dimensional attention grid; path is an existing output
    directory and idx is the fixation number. Uses Matplotlib color scaling
    and clears the current figure after saving. Returns None.
    """

    plt.imshow(map)
    plt.axis("off")
    plt.savefig(path + f"/map_fix{idx}.png", bbox_inches="tight", pad_inches=0, dpi=400)
    plt.clf()


# =============================================================================
# 3. Shared scanpath drawing and image rendering
# =============================================================================

def _draw_scanpath(ax, xs, ys):
    """Draw numbered fixations, highlighting the most recent fixation.

    ax is the destination Matplotlib axes. xs and ys contain corresponding
    scene-pixel coordinates in visit order. Adds arrows, circles, and
    zero-based fixation labels to the axes without saving a file.
    """

    # Draw connections first so fixation markers remain visible above the arrows.
    for i in range(len(xs)):
        if i > 0:
            ax.arrow(
                xs[i - 1],
                ys[i - 1],
                xs[i] - xs[i - 1],
                ys[i] - ys[i - 1],
                width=6,
                color="yellow",
                alpha=0.8,
            )

    # Number markers in visit order; distinguish the latest fixation by color.
    for i in range(len(xs)):

        cir_rad = 30
        if i == len(xs) - 1:
            circle = plt.Circle(
                (xs[i], ys[i]),
                radius=cir_rad,
                edgecolor="pink",
                facecolor="lightgreen",
                alpha=1.0,
            )
        else:
            circle = plt.Circle(
                (xs[i], ys[i]),
                radius=cir_rad,
                edgecolor="red",
                facecolor="yellow",
                alpha=1.0,
            )
        ax.add_patch(circle)
        ax.annotate(
            "{}".format(i), xy=(xs[i], ys[i] + 3), fontsize=14, ha="center", va="center"
        )


def plot_scanpath(img, xs, ys, file_name="scanpath.png", title=None):
    """Save the full scanpath on top of the original image.

    img is the scene; xs and ys provide fixation coordinates in pixels.
    file_name selects the output image and title optionally labels it.
    Creates and closes its own figure; returns None.
    """

    fig, ax = plt.subplots()
    ax.imshow(img)
    _draw_scanpath(ax, xs, ys)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    fig.savefig(file_name, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def _scanpath_frame(image, fixations):
    """Render the shared scanpath style at the image's exact pixel dimensions.

    image is an RGB frame and fixations is an (N, 2) array of (x, y) points.
    Returns an independent RGB pixel array with the cumulative scanpath.
    An off-screen Agg canvas avoids opening an interactive plotting window.
    """

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Match figure size to scene pixels and use the full canvas without margins.
    height, width = image.shape[:2]
    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    _draw_scanpath(ax, fixations[:, 0], fixations[:, 1])

    # Keep edge markers from changing the scene's extent.
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")

    # Copy RGB pixels out of the canvas so the returned frame owns its data.
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


# =============================================================================
# 4. Attention animation: scene overlays or previously saved maps
# =============================================================================

def generate_gif(
    path, fps=1, *, image=None, attention_maps=None, alpha=0.45, fixations=None
):
    """Save attention.gif, optionally overlaying smooth heatmaps on the RGB scene.

    Raw maps are supplied in fixation order, with blue for low attention and
    red for high attention. Per-frame normalization matches save_map.
    Alpha controls the overlay opacity. Optional (x, y) fixations are drawn
    cumulatively, through the fixation corresponding to each frame.
    Playback loops indefinitely, holding the final fixation for three seconds.
    Without a scene and raw maps, retain the saved-PNG workflow.

    path is the existing output directory. fps controls nonfinal frame
    durations and must be positive. Supply at least one attention map or
    saved map image. The function writes attention.gif and returns None.
    """

    # Validate paired overlay inputs and optional per-frame fixation coordinates.
    if (image is None) != (attention_maps is None):
        raise ValueError("Provide both image and attention_maps for an overlay.")
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be between 0 and 1.")

    if fixations is not None:
        if image is None:
            raise ValueError("Scanpaths require an image and attention maps.")
        fixations = np.asarray(fixations)
        if fixations.ndim != 2 or fixations.shape[1] != 2:
            raise ValueError("Fixations must be a sequence of (x, y) coordinates.")
        if len(fixations) < len(attention_maps):
            raise ValueError("Each attention map must have a corresponding fixation.")

    # Build frames in fixation order before handing them to the GIF encoder.
    images = []
    if image is not None:
        from skimage.util import img_as_ubyte

        scene = img_as_ubyte(image)
        height, width = scene.shape[:2]
        colormap = plt.get_cmap("jet")
        for index, attention in enumerate(attention_maps):
            attention = np.asarray(attention, dtype=float)

            # Normalize before interpolation to preserve the original color scale.
            normalized = plt.Normalize()(attention)
            smooth = cv.resize(
                np.asarray(normalized), (width, height), interpolation=cv.INTER_CUBIC
            )

            # Cubic interpolation can overshoot; clip before applying RGB colors.
            heatmap = np.round(colormap(np.clip(smooth, 0, 1))[..., :3] * 255).astype(
                np.uint8
            )

            # Blend in RGB space, then draw the path through this fixation.
            frame = cv.addWeighted(scene, 1 - alpha, heatmap, alpha, 0)
            if fixations is not None:
                frame = _scanpath_frame(frame, fixations[: index + 1])
            images.append(frame)
    else:

        # Sort by fixation number so map_fix10 follows map_fix9, not map_fix1.
        file_names = sorted(
            (
                name
                for name in os.listdir(path)
                if re.fullmatch(r"map_fix\d+\.png", name)
            ),
            key=lambda name: int(re.search(r"\d+", name).group()),
        )
        images = [imageio.v2.imread(os.path.join(path, name)) for name in file_names]

    # Pillow uses milliseconds; hold the final frame longer for inspection.
    durations = [1000 / fps] * (len(images) - 1) + [3000]
    imageio.mimsave(
        os.path.join(path, "attention.gif"),
        images,
        format="pillow",
        duration=durations if len(images) > 1 else 3000,
        loop=0,
    )


# =============================================================================
# 5. Area discretization and inhibition of return
# =============================================================================

def get_size_level(max_size, levels, area):
    """Return a one-based equal-width area bin, or 0 outside [0, max_size).

    max_size defines the upper area bound and levels the number of bins;
    both are expected to be positive. area uses the same units as max_size.
    Each bin includes its lower boundary and excludes its upper boundary.
    """

    interval = max_size / levels
    lvl = 0
    for i in range(levels):
        if area >= i * interval and area < (i + 1) * interval:
            lvl = i + 1
            break

    return lvl


def ior_in_area(map, ym, xm, dims):
    """Mark a 3x3 fixation neighborhood in place, clipping to the grid edges.

    Inhibition of return (IOR) marks cells that should not be selected again.
    Coordinates and dims use (row, column) order.

    map is a mutable grid, (ym, xm) identifies the fixation cell, and dims
    is its (row_count, column_count). Valid neighbors are assigned 1; other
    cells retain their values. The function returns None.
    """

    # Offsets -1, 0, and +1 include the fixation cell and its eight neighbors.
    for i in range(3):
        if ym + i - 1 < 0 or ym + i - 1 >= dims[0]:
            continue
        for j in range(3):
            if xm + j - 1 < 0 or xm + j - 1 >= dims[1]:
                continue
            map[ym + i - 1, xm + j - 1] = 1


# =============================================================================
# 6. Point and bounding-box coordinate helpers
# =============================================================================

def get_local_coordinates(center, bbox):
    """Return center minus bbox, treating both inputs as two-element points.

    center and bbox each contain (x, y); bbox denotes a reference point
    here, rather than a four-coordinate box. Returns a two-element NumPy
    array and raises TypeError when either input has the wrong length.
    """

    if len(center) != 2 or len(bbox) != 2:
        raise TypeError("Incorrect coordinate list dimensions!")

    u = center[0] - bbox[0]
    v = center[1] - bbox[1]

    return np.array([u, v])


def coordinate_converter(x, y, w, h):
    """Convert a top-left point and dimensions to two integer corner points.

    x and y locate the top-left corner; w and h are width and height in
    the same units. Returns ((xmin, ymin), (xmax, ymax)), truncating each
    coordinate toward zero with int.
    """

    p1 = (int(x), int(y))
    p2 = (int(x + w), int(y + h))

    return p1, p2


def box_center(p1, p2):
    """Return the integer (x, y) midpoint of two bounding-box corners.

    p1 and p2 are (x, y) corner points in the same coordinate system.
    Returns a two-element list, truncating the average of each axis
    toward zero with int.
    """

    center = [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]

    return center


# =============================================================================
# 7. Grid geometry and inclusive-pixel overlap
# =============================================================================

def in_cell(y_cell, x_cell, bbox, dims, h, w):
    """Check whether an xyxy box overlaps a grid cell under inclusive-pixel IoU.

    y_cell and x_cell are zero-based indices; dims is (rows, columns),
    and h and w are scene dimensions in pixels. bbox uses scene xyxy
    coordinates. Returns True for any positive intersection, including
    shared endpoints under the inclusive-pixel convention.
    """

    # Convert grid indices into scene-pixel boundaries without rounding.
    int_x = w / dims[1]
    int_y = h / dims[0]
    x_min = int_x * x_cell
    x_max = int_x * (x_cell + 1)
    y_min = int_y * y_cell
    y_max = int_y * (y_cell + 1)

    # Compare the detection with the cell rectangle using the shared IoU rule.
    box1 = [bbox[0], bbox[1], bbox[2], bbox[3]]
    box2 = [x_min, y_min, x_max, y_max]

    if intersection_over_union(box1, box2) > 0.0:
        return True
    else:
        return False


def cell_center(y_cell, x_cell, dims, h, w):
    """Return the cell center in scene pixels as (y, x), matching grid order.

    y_cell and x_cell are zero-based grid indices. dims is (rows, columns),
    and h and w are the scene height and width. Returns floating-point
    coordinates; callers needing integer (x, y) points must convert them.
    """

    # Convert grid indices into scene-pixel boundaries without rounding.
    int_x = w / dims[1]
    int_y = h / dims[0]
    x_min = int_x * x_cell
    x_max = int_x * (x_cell + 1)
    y_min = int_y * y_cell
    y_max = int_y * (y_cell + 1)

    # Average cell boundaries while retaining subpixel precision.
    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2

    return (yc, xc)


def intersection_over_union(boxA, boxB):
    """Compute IoU for xyxy boxes, counting both endpoints of each pixel span.

    boxA and boxB each contain [xmin, ymin, xmax, ymax] with ordered
    corners. Returns intersection area divided by union area; disjoint
    boxes yield zero. Coordinates are not clipped to an image boundary.
    """

    # Find the intersection rectangle before computing intersection and union.
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Clamp disjoint spans to zero; +1 includes both endpoints of a pixel span.
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


# =============================================================================
# 8. Grid summaries and detection visualization
# =============================================================================

def map_avg(map, dims):
    """Average values over the first dims[0] rows and dims[1] columns.

    map must support two-dimensional indexing over the requested dims.
    Returns the arithmetic mean of that region, assuming both dimensions
    are positive; values outside the region are ignored.
    """

    average = 0.0
    for i in range(dims[0]):
        for j in range(dims[1]):
            average += map[i][j]
    average = average / (dims[0] * dims[1])

    return average


def annotator(image, detections, labels, colors, n_classes):
    """Return an image copy with boxes colored by the final detection class ID.

    detections follow [xmin, ymin, xmax, ymax, *scores, class_id]. labels
    and colors are indexed by the final ID, while n_classes sets the score
    slice width. Returns a copy with rectangles in the supplied color order;
    the input image is unchanged and label text is currently not drawn.
    """

    img = image.copy()
    font_scale = 1

    # Detection rows end in a class ID used to index the palette and label list.
    for det in detections:

        color = tuple([int(c) for c in colors[int(det[-1])]])
        xmin, ymin, xmax, ymax = det[:4]

        label_text = f"{labels[int(det[-1])]}: {max(det[4:(n_classes+4)]):.2f}"
        (text_width, text_height), baseline = cv.getTextSize(
            label_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Draw the class-colored box; the computed label text is not rendered here.
        cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

    return img


def print_map(map, dims):
    """Print grid values row by row with three significant digits.

    map supports two-dimensional indexing and dims specifies the rows and
    columns to display. Writes tab-separated values to standard output,
    with one newline per row; returns None.
    """

    for i in range(dims[0]):
        for j in range(dims[1]):
            print("{:.3}".format(map[i][j]), end="\t")
        print("", end="\n")
