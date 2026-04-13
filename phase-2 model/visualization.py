import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe


PIXELS_PER_METER = 4.0
Y_MIN = -32.0
X_MAX = 32.0
DEFAULT_X_MIN = -32.0
DEFAULT_Y_MAX = 32.0

CLASS_NAMES = ["background", "road", "walkway", "centerline", "static", "vehicle", "pedestrian"]
CLASS_COLORS = ["#FFFFFF", "#D3D3D3", "#d4d19e", "#666666", "#edc948", "#699CDB", "#b07aa1"]


def _to_numpy(data):
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def _xy_from_pose_sequence(sequence):
    sequence = _to_numpy(sequence)
    if sequence.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return sequence[..., :2]


def _prepend_xy(current_xy, future_sequence):
    current_xy = np.asarray(current_xy, dtype=np.float32).reshape(1, 2)
    future_xy = _xy_from_pose_sequence(future_sequence)
    if future_xy.size == 0:
        return current_xy
    return np.concatenate([current_xy, future_xy], axis=0)


def _heading_from_cos_sin(cosine, sine):
    return np.arctan2(sine, cosine)


def _box_corners_xy(box):
    x, y, heading, length, width = box
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
        ]
    )
    rot = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    return local @ rot.T + np.array([x, y])


def _plot_box(ax, box, color, linewidth=1.5, alpha=1.0):
    corners = _box_corners_xy(box)
    ax.plot(corners[:, 1], corners[:, 0], color=color, linewidth=linewidth, alpha=alpha)


def _plot_traj(ax, traj_xy, color, label=None, linestyle="-", marker="o", linewidth=2.2, alpha=1.0):
    traj_xy = _to_numpy(traj_xy)
    if traj_xy is None or traj_xy.size == 0:
        return
    line, = ax.plot(
        traj_xy[:, 1],
        traj_xy[:, 0],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=4,
        alpha=alpha,
        label=label,
    )
    return line


def _style_bev_axis(ax, title, x_min=DEFAULT_X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=DEFAULT_Y_MAX):
    ax.set_title(title)
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")


def _draw_panel(ax, bev_map, boxes, scores, traj, title, box_color, traj_color):
    bev_map = _to_numpy(bev_map)
    height, width = bev_map.shape
    y_max = Y_MIN + width / PIXELS_PER_METER
    x_min = X_MAX - height / PIXELS_PER_METER

    ax.imshow(
        bev_map,
        cmap=ListedColormap(CLASS_COLORS),
        interpolation="nearest",
        origin="lower",
        extent=[Y_MIN, y_max, x_min, X_MAX],
        vmin=0,
        vmax=len(CLASS_COLORS) - 1,
    )
    valid_mask = _to_numpy(scores) > 0
    for box in _to_numpy(boxes)[valid_mask]:
        _plot_box(ax, box, color=box_color, linewidth=1.2)
    _plot_traj(ax, _xy_from_pose_sequence(traj), color=traj_color, label="trajectory")
    ax.scatter(0.0, 0.0, color="red", s=60, marker="*", label="ego")
    _style_bev_axis(ax, title, x_min=x_min, x_max=X_MAX, y_min=Y_MIN, y_max=y_max)
    ax.legend(loc="upper right")


def _lane_segments(lanes):
    lanes = _to_numpy(lanes)
    if lanes is None:
        return []

    segments = []
    for lane in lanes:
        valid = np.linalg.norm(lane[:, :2], axis=-1) > 0
        points = lane[valid, :2]
        if len(points) >= 2:
            segments.append(points)
    return segments


def _neighbor_boxes_from_history(neighbor_agents_past):
    neighbor_agents_past = _to_numpy(neighbor_agents_past)
    current_states = neighbor_agents_past[:, -1]
    valid_mask = np.linalg.norm(current_states[:, :2], axis=-1) > 0
    valid_states = current_states[valid_mask]

    boxes = []
    for state in valid_states:
        heading = _heading_from_cos_sin(state[2], state[3])
        width = state[6]
        length = state[7]
        boxes.append([state[0], state[1], heading, length, width])
    return np.asarray(boxes, dtype=np.float32)


def _active_neighbor_mask(neighbor_agents_past, max_neighbors=None):
    neighbor_agents_past = _to_numpy(neighbor_agents_past)
    mask = np.linalg.norm(neighbor_agents_past[:, -1, :2], axis=-1) > 0
    if max_neighbors is not None:
        mask = mask[:max_neighbors]
    return mask


def _static_boxes(static_objects):
    static_objects = _to_numpy(static_objects)
    valid_mask = np.linalg.norm(static_objects[:, :2], axis=-1) > 0
    valid_states = static_objects[valid_mask]

    boxes = []
    for state in valid_states:
        heading = _heading_from_cos_sin(state[2], state[3])
        width = state[4]
        length = state[5]
        boxes.append([state[0], state[1], heading, length, width])
    return np.asarray(boxes, dtype=np.float32)


def _resolve_decoder_output(output):
    if isinstance(output, (tuple, list)) and len(output) >= 2 and isinstance(output[1], dict):
        return output[1]
    return output


def show_transfuser_result(output, targets, features):
    pred_bev = _to_numpy(output["bev_semantic_map"][0]).argmax(axis=0)
    pred_traj = _to_numpy(output["trajectory"][0])
    pred_boxes = _to_numpy(output["agent_states"][0])
    pred_scores = _to_numpy(output["agent_labels"][0])

    gt_bev = _to_numpy(targets["bev_semantic_map"][0])
    gt_traj = _to_numpy(targets["trajectory"][0])
    gt_boxes = _to_numpy(targets["agent_states"][0])
    gt_scores = _to_numpy(targets["agent_labels"][0])
    camera_image = _to_numpy(features["camera_feature"][0].permute(1, 2, 0))
    lidar_feature = _to_numpy(features["lidar_feature"][0, 0])

    class_handles = [Patch(facecolor=color, edgecolor="none", label=name) for name, color in zip(CLASS_NAMES, CLASS_COLORS)]
    overlay_handles = [
        Patch(facecolor="deepskyblue", edgecolor="none", label="pred boxes (BEV)"),
        Patch(facecolor="lime", edgecolor="none", label="gt boxes (BEV)"),
        Patch(facecolor="green", edgecolor="none", label="pred boxes (lidar_feature)"),
        Patch(facecolor="red", edgecolor="none", label="gt boxes (lidar_feature)"),
    ]

    fig = plt.figure(figsize=(16, 24), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 2, 8])
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_cam = fig.add_subplot(gs[1, :])
    ax_lidar = fig.add_subplot(gs[2, :])

    _draw_panel(ax_pred, pred_bev, pred_boxes, pred_scores, pred_traj, "Prediction", "deepskyblue", "white")
    _draw_panel(ax_gt, gt_bev, gt_boxes, gt_scores, gt_traj, "Ground Truth", "lime", "magenta")

    ax_cam.imshow(camera_image)
    ax_cam.set_title("Stitched Front Camera Image")
    ax_cam.axis("off")

    ax_lidar.imshow(lidar_feature, cmap="gray", origin="lower", extent=[-32, 32, -32, 32])
    for box in gt_boxes[gt_scores > 0]:
        _plot_box(ax_lidar, box, color="red", linewidth=1.8)
    for box in pred_boxes[pred_scores > 0]:
        _plot_box(ax_lidar, box, color="green", linewidth=1.4)
    ax_lidar.scatter(0.0, 0.0, color="red", s=80, marker="*")
    ax_lidar.set_title("Model Input lidar_feature with Bounding Boxes")
    ax_lidar.set_xlim(-32, 32)
    ax_lidar.set_ylim(-32, 32)
    ax_lidar.invert_xaxis()
    ax_lidar.set_box_aspect(lidar_feature.shape[0] / lidar_feature.shape[1])
    ax_lidar.grid(True, alpha=0.2)
    ax_lidar.set_facecolor("#111111")
    ax_lidar.set_xlabel("y (m)")
    ax_lidar.set_ylabel("x (m)")

    fig.legend(
        handles=class_handles + overlay_handles,
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title="Semantic Classes",
    )
    plt.show()


def show_diffusion_planner_result(output, targets, features):
    decoder_output = _resolve_decoder_output(output)
    if "prediction" not in decoder_output:
        raise KeyError("Expected diffusion planner inference output to contain 'prediction'.")

    prediction = _to_numpy(decoder_output["prediction"][0])
    pred_ego = prediction[0]
    pred_neighbors = prediction[1:]

    gt_ego = _to_numpy(targets["ego_future_gt"][0])
    gt_neighbors = _to_numpy(targets["neighbors_future_gt"][0])
    neighbor_future_mask = _to_numpy(targets["neighbor_future_mask"][0]).astype(bool)
    gt_traj = _to_numpy(targets["trajectory"][0])

    ego_current = _to_numpy(features["ego_current_state"][0])
    neighbor_agents_past = _to_numpy(features["neighbor_agents_past"][0])
    lanes = _to_numpy(features["lanes"][0])
    route_lanes = _to_numpy(features["route_lanes"][0])

    current_neighbor_boxes = _neighbor_boxes_from_history(neighbor_agents_past)
    pred_neighbor_count = min(len(pred_neighbors), len(gt_neighbors))
    active_neighbor_mask = _active_neighbor_mask(neighbor_agents_past, pred_neighbor_count)
    current_neighbor_xy = neighbor_agents_past[:pred_neighbor_count, -1, :2]
    ego_current_xy = ego_current[:2]

    fig, ax_main = plt.subplots(figsize=(12, 12), constrained_layout=True)

    for segment in _lane_segments(lanes):
        ax_main.plot(segment[:, 1], segment[:, 0], color="#c7c7c7", linewidth=1.0, alpha=0.6)
    for segment in _lane_segments(route_lanes):
        ax_main.plot(segment[:, 1], segment[:, 0], color="#f2a104", linewidth=2.0, alpha=0.85)
    for box in current_neighbor_boxes:
        _plot_box(ax_main, box, color="#7f7f7f", linewidth=1.3, alpha=0.7)
    ax_main.scatter(0.0, 0.0, color="red", s=80, marker="*")
    _style_bev_axis(ax_main, "")

    pred_ego_line = _plot_traj(
        ax_main,
        _prepend_xy(ego_current_xy, pred_ego),
        color="#1f77b4",
        label="pred ego",
        linewidth=2.8,
    )
    pred_ego_line.set_path_effects([pe.Stroke(linewidth=4.6, foreground="black"), pe.Normal()])
    pred_neighbor_labeled = False
    for idx in range(pred_neighbor_count):
        if not active_neighbor_mask[idx]:
            continue
        _plot_traj(
            ax_main,
            _prepend_xy(current_neighbor_xy[idx], pred_neighbors[idx]),
            color="#4c78a8",
            label="pred neighbors" if not pred_neighbor_labeled else None,
            linewidth=1.8,
            alpha=0.95,
        )
        pred_neighbor_labeled = True
    _plot_traj(ax_main, _prepend_xy(ego_current_xy, gt_ego), color="#d62728", label="gt ego", linewidth=2.6)
    gt_neighbor_labeled = False
    for idx in range(pred_neighbor_count):
        if not active_neighbor_mask[idx]:
            continue
        valid = ~neighbor_future_mask[idx]
        _plot_traj(
            ax_main,
            _prepend_xy(current_neighbor_xy[idx], gt_neighbors[idx][valid]),
            color="#ff7f0e",
            label="gt neighbors" if not gt_neighbor_labeled else None,
            linewidth=1.8,
            alpha=0.95,
        )
        gt_neighbor_labeled = True
    _plot_traj(ax_main, _prepend_xy(ego_current_xy, gt_traj), color="#9467bd", label="trajectory target", linestyle="--", marker="x")
    ax_main.set_title("Prediction And Ground Truth")
    ax_main.legend(loc="upper right")

    legend_handles = [
        Line2D([0], [0], color="#c7c7c7", lw=1.5, label="lanes"),
        Line2D([0], [0], color="#f2a104", lw=2.0, label="route lanes"),
        Line2D([0], [0], color="#7f7f7f", lw=1.5, label="current neighbors"),
        Line2D([0], [0], color="#1f77b4", lw=2.5, label="pred ego"),
        Line2D([0], [0], color="#4c78a8", lw=2.0, label="pred neighbors"),
        Line2D([0], [0], color="#d62728", lw=2.5, label="gt ego"),
        Line2D([0], [0], color="#ff7f0e", lw=2.0, label="gt neighbors"),
        Line2D([0], [0], color="#9467bd", lw=2.0, label="trajectory target"),
    ]
    fig.legend(handles=legend_handles, loc="center right", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.show()
