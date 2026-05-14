from pathlib import Path
import argparse
import csv
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# readable bags

BAG_INFO_USABLE = {
    "ett1": {
        "display_name": "Early Direct EOH Run",
        "algorithm": "Direct baseline EOH publishing directly to /ackermann_cmd",
        "category": "Baseline EOH",
        "environment": "Real Hardware",
        "status": "partial",
        "comparison_group": "baseline_eoh",
        "website_caption": (
            "This bag captured an early real-hardware EOH run using direct LiDAR-to-Ackermann control. "
            "It demonstrates the baseline reactive wall-balancing controller before later tuning."
        ),
    },

    "real_track_1": {
        "display_name": "Successful Real Track Run",
        "algorithm": "Final or near-final EOH with steering-based speed cap",
        "category": "Final Race",
        "environment": "Real Hardware",
        "status": "successful",
        "comparison_group": "final_algorithm",
        "website_caption": (
            "This bag represents a successful real-track run using the tuned EOH controller. "
            "It provides strong evidence that the final reactive controller could complete the physical F1TENTH course."
        ),
    },

    "rttp2": {
        "display_name": "Race Track Tuning Pass 2",
        "algorithm": "Aggressive EOH race-speed tuning",
        "category": "Speed Tuning",
        "environment": "Real Hardware",
        "status": "failure",
        "comparison_group": "race_speed_tuning",
        "website_caption": (
            "This run tested more aggressive race-speed settings and exposed cases where the car entered turns too quickly."
        ),
    },

    "rttp3": {
        "display_name": "Race Track Tuning Pass 3",
        "algorithm": "Aggressive EOH race-speed tuning with adjusted gates",
        "category": "Speed Tuning",
        "environment": "Real Hardware",
        "status": "partial",
        "comparison_group": "race_speed_tuning",
        "website_caption": (
            "This bag captured a later race-track tuning attempt after speed gates were adjusted."
        ),
    },

    "scared_stop_debug3": {
        "display_name": "Scared Stop Debug 3",
        "algorithm": "EOH tuning with modified steering sensitivity",
        "category": "Safety and Steering Debug",
        "environment": "Real Hardware",
        "status": "debug",
        "comparison_group": "safety_debug",
        "website_caption": (
            "This bag tested steering sensitivity changes and helped reveal when the controller became too aggressive or unstable."
        ),
    },

    "scared_stop_debug4": {
        "display_name": "Scared Stop Debug 4",
        "algorithm": "Stable EOH retest with crawl/stop threshold changes",
        "category": "Safety and Crawl Tuning",
        "environment": "Real Hardware",
        "status": "debug",
        "comparison_group": "safety_debug",
        "website_caption": (
            "This run returned to a more stable EOH baseline and tested softer crawl behavior instead of hard stopping."
        ),
    },

    "scared_stop_debug5": {
        "display_name": "Improved EOH Turning Run",
        "algorithm": "Improved EOH with earlier turn reaction",
        "category": "EOH Tuning",
        "environment": "Real Hardware",
        "status": "partial",
        "comparison_group": "final_eoh_tuning",
        "website_caption": (
            "This bag captured an improved EOH configuration that completed most of the track but still struggled with a tight turn."
        ),
    },

    "teleop_run1_scan_only": {
        "display_name": "Teleop Scan-Only Track Reference",
        "algorithm": "Manual teleoperation, scan-only reference recording",
        "category": "Track Reference",
        "environment": "Real Hardware",
        "status": "teleop only",
        "comparison_group": "track_reference",
        "website_caption": (
            "This bag provides a manually driven scan reference of the physical track."
        ),
    },
}


#  Ackermann message registration

ACKERMANN_DRIVE_MSG = """
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""

ACKERMANN_DRIVE_STAMPED_MSG = """
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""


def build_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    custom_types = {}
    custom_types.update(
        get_types_from_msg(
            ACKERMANN_DRIVE_MSG,
            "ackermann_msgs/msg/AckermannDrive",
        )
    )
    custom_types.update(
        get_types_from_msg(
            ACKERMANN_DRIVE_STAMPED_MSG,
            "ackermann_msgs/msg/AckermannDriveStamped",
        )
    )

    typestore.register(custom_types)
    return typestore




def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )


def load_metadata(bag_dir):
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metadata_duration_sec(metadata):
    try:
        return metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"] / 1e9
    except Exception:
        return np.nan


def metadata_message_count(metadata):
    try:
        return metadata["rosbag2_bagfile_information"]["message_count"]
    except Exception:
        return 0


def metadata_topic_counts(metadata):
    topic_counts = {}

    try:
        topics = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    except Exception:
        return topic_counts

    for entry in topics:
        meta = entry["topic_metadata"]
        topic_counts[meta["name"]] = {
            "type": meta["type"],
            "count": entry["message_count"],
        }

    return topic_counts


def normalize_time(t_ns):
    t = np.array(t_ns, dtype=float) / 1e9
    if len(t) == 0:
        return t
    return t - t[0]


def valid_array(values):
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def compute_front_distance(scan_msg, front_angle_deg=10.0):
    ranges = np.array(scan_msg.ranges, dtype=float)

    if len(ranges) == 0:
        return np.nan

    angle_min = float(scan_msg.angle_min)
    angle_increment = float(scan_msg.angle_increment)

    if angle_increment == 0:
        return np.nan

    front_angle = math.radians(front_angle_deg)

    idx_min = int((-front_angle - angle_min) / angle_increment)
    idx_max = int((front_angle - angle_min) / angle_increment)

    idx_min = max(0, min(idx_min, len(ranges) - 1))
    idx_max = max(0, min(idx_max, len(ranges) - 1))

    if idx_min > idx_max:
        idx_min, idx_max = idx_max, idx_min

    cone = ranges[idx_min:idx_max + 1]
    valid = cone[np.isfinite(cone) & (cone > 0.05)]

    if len(valid) == 0:
        return np.nan

    return float(np.min(valid))


def steering_smoothness(steer, t):
    steer = np.array(steer, dtype=float)
    t = np.array(t, dtype=float)

    if len(steer) < 2 or len(t) < 2:
        return np.nan

    dt = np.diff(t)
    ds = np.diff(steer)

    valid = dt > 1e-6
    if not np.any(valid):
        return np.nan

    return float(np.mean(np.abs(ds[valid] / dt[valid])))


def count_button_rising_edges(button_array):
    if len(button_array) < 2:
        return 0

    arr = np.array(button_array, dtype=int)
    return int(np.sum(np.diff(arr) == 1))


# Bag analysis

def analyze_bag(bag_dir, bag_info, typestore):
    metadata = load_metadata(bag_dir)

    result = {
        "bag": bag_dir.name,
        "display_name": bag_info["display_name"],
        "algorithm": bag_info["algorithm"],
        "category": bag_info["category"],
        "environment": bag_info["environment"],
        "status": bag_info["status"],
        "comparison_group": bag_info["comparison_group"],
        "website_caption": bag_info["website_caption"],

        "metadata_duration_sec": np.nan,
        "metadata_total_messages": 0,

        "has_scan": False,
        "has_ackermann_cmd": False,
        "has_joy": False,
        "has_odom": False,
        "has_drive_raw": False,
        "has_scan_processed": False,

        "scan_metadata_count": 0,
        "ackermann_metadata_count": 0,
        "joy_metadata_count": 0,
        "odom_metadata_count": 0,

        "read_success": False,
        "read_error": "",

        "drive_count": 0,
        "scan_count": 0,
        "joy_count": 0,
        "odom_count": 0,

        "mean_speed": np.nan,
        "mean_positive_speed": np.nan,
        "max_speed": np.nan,
        "zero_speed_fraction_pct": np.nan,

        "mean_abs_steering": np.nan,
        "max_abs_steering": np.nan,
        "steering_smoothness_rad_per_sec": np.nan,
        "steering_saturation_fraction_pct": np.nan,

        "min_front_distance": np.nan,
        "mean_front_distance": np.nan,
        "front_below_0_50_pct": np.nan,
        "front_below_0_75_pct": np.nan,

        "joy_y_pressed_fraction_pct": np.nan,
        "joy_y_press_count": np.nan,
    }

    data = {
        "drive_t_ns": [],
        "speed": [],
        "steer": [],

        "scan_t_ns": [],
        "front_dist": [],

        "joy_t_ns": [],
        "joy_y": [],

        "odom_t_ns": [],
        "odom_x": [],
        "odom_y": [],

        "drive_raw_t_ns": [],
        "drive_raw_speed": [],
        "drive_raw_steer": [],
    }

    if metadata is not None:
        result["metadata_duration_sec"] = metadata_duration_sec(metadata)
        result["metadata_total_messages"] = metadata_message_count(metadata)
        topic_counts = metadata_topic_counts(metadata)

        result["has_scan"] = "/scan" in topic_counts
        result["has_ackermann_cmd"] = "/ackermann_cmd" in topic_counts
        result["has_joy"] = "/joy" in topic_counts
        result["has_odom"] = "/odom" in topic_counts
        result["has_drive_raw"] = "/ego_racecar/drive_raw" in topic_counts
        result["has_scan_processed"] = "/ego_racecar/scan_processed" in topic_counts

        result["scan_metadata_count"] = topic_counts.get("/scan", {}).get("count", 0)
        result["ackermann_metadata_count"] = topic_counts.get("/ackermann_cmd", {}).get("count", 0)
        result["joy_metadata_count"] = topic_counts.get("/joy", {}).get("count", 0)
        result["odom_metadata_count"] = topic_counts.get("/odom", {}).get("count", 0)

    wanted_topics = {
        "/ackermann_cmd",
        "/scan",
        "/joy",
        "/odom",
        "/ego_racecar/drive_raw",
    }

    try:
        with AnyReader([bag_dir], default_typestore=typestore) as reader:
            connections = [
                c for c in reader.connections
                if c.topic in wanted_topics
            ]

            print(f"\nAnalyzing: {bag_dir.name}")
            print(f"  Display name: {bag_info['display_name']}")
            print(f"  Useful topics found: {sorted(set(c.topic for c in connections))}")

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                except Exception as e:
                    print(f"  Warning: could not deserialize {connection.topic}: {e}")
                    continue

                if connection.topic == "/ackermann_cmd":
                    data["drive_t_ns"].append(timestamp)
                    data["speed"].append(float(msg.drive.speed))
                    data["steer"].append(float(msg.drive.steering_angle))

                elif connection.topic == "/ego_racecar/drive_raw":
                    data["drive_raw_t_ns"].append(timestamp)
                    data["drive_raw_speed"].append(float(msg.drive.speed))
                    data["drive_raw_steer"].append(float(msg.drive.steering_angle))

                elif connection.topic == "/scan":
                    data["scan_t_ns"].append(timestamp)
                    data["front_dist"].append(compute_front_distance(msg))

                elif connection.topic == "/joy":
                    data["joy_t_ns"].append(timestamp)
                    if len(msg.buttons) > 3:
                        data["joy_y"].append(int(msg.buttons[3]))
                    else:
                        data["joy_y"].append(0)

                elif connection.topic == "/odom":
                    data["odom_t_ns"].append(timestamp)
                    data["odom_x"].append(float(msg.pose.pose.position.x))
                    data["odom_y"].append(float(msg.pose.pose.position.y))

        result["read_success"] = True

    except Exception as e:
        result["read_success"] = False
        result["read_error"] = str(e)
        print(f"  FAILED to read {bag_dir.name}: {e}")
        return result, data

    # Compute metrics
    speed = valid_array(data["speed"])
    steer = valid_array(data["steer"])
    front_dist = valid_array(data["front_dist"])
    joy_y = np.array(data["joy_y"], dtype=float)
    drive_t = normalize_time(data["drive_t_ns"])

    result["drive_count"] = len(speed)
    result["scan_count"] = len(front_dist)
    result["joy_count"] = len(joy_y)
    result["odom_count"] = len(data["odom_x"])

    if len(speed) > 0:
        result["mean_speed"] = float(np.mean(speed))
        positive_speed = speed[speed > 0.05]
        if len(positive_speed) > 0:
            result["mean_positive_speed"] = float(np.mean(positive_speed))
        result["max_speed"] = float(np.max(speed))
        result["zero_speed_fraction_pct"] = float(100.0 * np.mean(np.abs(speed) < 0.05))

    if len(steer) > 0:
        abs_steer = np.abs(steer)
        result["mean_abs_steering"] = float(np.mean(abs_steer))
        result["max_abs_steering"] = float(np.max(abs_steer))
        result["steering_smoothness_rad_per_sec"] = steering_smoothness(steer, drive_t)
        result["steering_saturation_fraction_pct"] = float(100.0 * np.mean(abs_steer > 0.38))

    if len(front_dist) > 0:
        result["min_front_distance"] = float(np.min(front_dist))
        result["mean_front_distance"] = float(np.mean(front_dist))
        result["front_below_0_50_pct"] = float(100.0 * np.mean(front_dist < 0.50))
        result["front_below_0_75_pct"] = float(100.0 * np.mean(front_dist < 0.75))

    if len(joy_y) > 0:
        result["joy_y_pressed_fraction_pct"] = float(100.0 * np.mean(joy_y))
        result["joy_y_press_count"] = count_button_rising_edges(joy_y)

    return result, data


# Plotting

def plot_speed_steering(data, result, out_path):
    t = normalize_time(data["drive_t_ns"])
    speed = np.array(data["speed"], dtype=float)
    steer = np.array(data["steer"], dtype=float)

    if len(t) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Speed on left axis
    speed_color = "tab:blue"
    ax1.plot(t, speed, color=speed_color, linewidth=1.8, label="Commanded speed")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)", color=speed_color)
    ax1.tick_params(axis="y", labelcolor=speed_color)
    ax1.grid(True, alpha=0.3)

    # Steering on right axis
    steer_color = "tab:orange"
    ax2 = ax1.twinx()
    ax2.plot(t, steer, color=steer_color, linewidth=1.4, label="Steering angle")
    ax2.set_ylabel("Steering angle (rad)", color=steer_color)
    ax2.tick_params(axis="y", labelcolor=steer_color)

    plt.title(f"{result['display_name']}: Speed and Steering")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_front_distance(data, result, out_path):
    t = normalize_time(data["scan_t_ns"])
    front = np.array(data["front_dist"], dtype=float)
    valid = np.isfinite(front)

    t = t[valid]
    front = front[valid]

    if len(t) == 0:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, front, label="Front LiDAR clearance")
    ax.axhline(0.50, linestyle="--", label="0.50 m reference")
    ax.axhline(0.75, linestyle="--", label="0.75 m reference")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Front distance (m)")
    ax.set_title(f"{result['display_name']}: Front LiDAR Clearance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_speed_vs_abs_steering(data, result, out_path):
    speed = np.array(data["speed"], dtype=float)
    steer = np.abs(np.array(data["steer"], dtype=float))

    if len(speed) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(steer, speed, s=8, alpha=0.5)

    ax.axvline(0.22, linestyle="--", label="0.22 rad speed gate")
    ax.axvline(0.32, linestyle=":", label="0.32 rad sharp-turn gate")

    ax.set_xlabel("|Steering angle| (rad)")
    ax.set_ylabel("Commanded speed (m/s)")
    ax.set_title(f"{result['display_name']}: Speed vs Steering Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_steering_histogram(data, result, out_path):
    steer = np.array(data["steer"], dtype=float)

    if len(steer) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(steer, bins=40)

    ax.axvline(0.22, linestyle="--", label="+0.22 rad")
    ax.axvline(-0.22, linestyle="--", label="-0.22 rad")
    ax.axvline(0.32, linestyle=":", label="+0.32 rad")
    ax.axvline(-0.32, linestyle=":", label="-0.32 rad")

    ax.set_xlabel("Steering angle (rad)")
    ax.set_ylabel("Count")
    ax.set_title(f"{result['display_name']}: Steering Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_trajectory(data, result, out_path):
    x = np.array(data["odom_x"], dtype=float)
    y = np.array(data["odom_y"], dtype=float)

    if len(x) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y)
    ax.scatter([x[0]], [y[0]], label="Start")
    ax.scatter([x[-1]], [y[-1]], label="End")

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title(f"{result['display_name']}: Odometry Trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_raw_vs_final_drive(data, result, out_path):
    raw_t = normalize_time(data["drive_raw_t_ns"])
    raw_speed = np.array(data["drive_raw_speed"], dtype=float)
    final_t = normalize_time(data["drive_t_ns"])
    final_speed = np.array(data["speed"], dtype=float)

    if len(raw_t) == 0 or len(final_t) == 0:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(raw_t, raw_speed, label="/ego_racecar/drive_raw speed")
    ax.plot(final_t, final_speed, label="/ackermann_cmd final speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed command (m/s)")
    ax.set_title(f"{result['display_name']}: Raw vs Final Speed Command")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_comparison_bar(results, metric, ylabel, title, out_path):
    valid = [
        r for r in results
        if isinstance(r.get(metric), (int, float)) and not np.isnan(r.get(metric))
    ]

    if len(valid) == 0:
        return

    labels = [r["display_name"] for r in valid]
    values = [r[metric] for r in valid]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_topic_availability(results, out_path):
    topics = [
        "has_scan",
        "has_ackermann_cmd",
        "has_joy",
        "has_odom",
        "has_drive_raw",
        "has_scan_processed",
    ]

    topic_labels = [
        "/scan",
        "/ackermann_cmd",
        "/joy",
        "/odom",
        "/drive_raw",
        "/scan_processed",
    ]

    labels = [r["display_name"] for r in results]
    matrix = np.array([[1 if r[t] else 0 for t in topics] for r in results])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(topic_labels)))
    ax.set_xticklabels(topic_labels, rotation=25, ha="right")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "yes" if matrix[i, j] else "no",
                    ha="center", va="center", fontsize=8)

    ax.set_title("Readable Rosbag Topic Availability")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Output files

def write_summary_csv(results, out_path):
    fieldnames = [
        "bag",
        "display_name",
        "algorithm",
        "category",
        "environment",
        "status",
        "comparison_group",
        "metadata_duration_sec",
        "metadata_total_messages",
        "read_success",
        "read_error",
        "drive_count",
        "scan_count",
        "joy_count",
        "odom_count",
        "mean_speed",
        "mean_positive_speed",
        "max_speed",
        "zero_speed_fraction_pct",
        "mean_abs_steering",
        "max_abs_steering",
        "steering_smoothness_rad_per_sec",
        "steering_saturation_fraction_pct",
        "min_front_distance",
        "mean_front_distance",
        "front_below_0_50_pct",
        "front_below_0_75_pct",
        "joy_y_pressed_fraction_pct",
        "joy_y_press_count",
        "website_caption",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {}
            for key in fieldnames:
                value = r.get(key, "")
                if isinstance(value, float):
                    row[key] = "" if np.isnan(value) else f"{value:.4f}"
                else:
                    row[key] = value
            writer.writerow(row)


def write_html_table(results, out_path):
    rows = []

    rows.append("<table>")
    rows.append("  <thead>")
    rows.append("    <tr>")
    rows.append("      <th>Run</th>")
    rows.append("      <th>Algorithm</th>")
    rows.append("      <th>Status</th>")
    rows.append("      <th>Duration</th>")
    rows.append("      <th>Mean Speed</th>")
    rows.append("      <th>Max Speed</th>")
    rows.append("      <th>Mean |Steering|</th>")
    rows.append("      <th>Result</th>")
    rows.append("    </tr>")
    rows.append("  </thead>")
    rows.append("  <tbody>")

    for r in results:
        duration = f"{r['metadata_duration_sec']:.1f}s" if not np.isnan(r["metadata_duration_sec"]) else "N/A"
        mean_speed = f"{r['mean_positive_speed']:.2f} m/s" if not np.isnan(r["mean_positive_speed"]) else "N/A"
        max_speed = f"{r['max_speed']:.2f} m/s" if not np.isnan(r["max_speed"]) else "N/A"
        mean_steer = f"{r['mean_abs_steering']:.3f} rad" if not np.isnan(r["mean_abs_steering"]) else "N/A"

        rows.append("    <tr>")
        rows.append(f"      <td>{r['display_name']}</td>")
        rows.append(f"      <td>{r['algorithm']}</td>")
        rows.append(f"      <td>{r['status']}</td>")
        rows.append(f"      <td>{duration}</td>")
        rows.append(f"      <td>{mean_speed}</td>")
        rows.append(f"      <td>{max_speed}</td>")
        rows.append(f"      <td>{mean_steer}</td>")
        rows.append(f"      <td>{r['website_caption']}</td>")
        rows.append("    </tr>")

    rows.append("  </tbody>")
    rows.append("</table>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def print_summary(results):
    print("\n==============================")
    print("Readable Bag Analysis Summary")
    print("==============================")

    for r in results:
        print(f"\n{r['display_name']} ({r['bag']})")
        print(f"  Status: {r['status']}")
        print(f"  Algorithm: {r['algorithm']}")
        print(f"  Duration: {r['metadata_duration_sec']:.2f}s")
        print(f"  Read success: {r['read_success']}")

        if r["read_success"]:
            print(f"  Drive cmds: {r['drive_count']}")
            print(f"  Scan msgs: {r['scan_count']}")
            print(f"  Odom msgs: {r['odom_count']}")
            print(f"  Mean positive speed: {r['mean_positive_speed']}")
            print(f"  Max speed: {r['max_speed']}")
            print(f"  Mean |steer|: {r['mean_abs_steering']}")
        else:
            print(f"  Error: {r['read_error']}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder containing rosbag folders")
    parser.add_argument("--out", required=True, help="Output folder for result plots")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    mkdir(out)

    typestore = build_typestore()

    results = []
    all_data = {}

    for bag_name, info in BAG_INFO_USABLE.items():
        bag_dir = root / bag_name

        if not bag_dir.exists():
            print(f"\nSkipping {bag_name}: folder not found")
            continue

        if not (bag_dir / "metadata.yaml").exists():
            print(f"\nSkipping {bag_name}: metadata.yaml not found")
            continue

        result, data = analyze_bag(bag_dir, info, typestore)
        results.append(result)
        all_data[bag_name] = data

        slug = safe_slug(bag_name)

        if result["read_success"]:
            plot_speed_steering(data, result, out / f"{slug}_speed_steering.png")
            plot_front_distance(data, result, out / f"{slug}_front_distance.png")
            plot_speed_vs_abs_steering(data, result, out / f"{slug}_speed_vs_abs_steering.png")
            plot_steering_histogram(data, result, out / f"{slug}_steering_histogram.png")
            plot_trajectory(data, result, out / f"{slug}_trajectory.png")
            plot_raw_vs_final_drive(data, result, out / f"{slug}_raw_vs_final_speed.png")

    if len(results) == 0:
        print("No bags analyzed.")
        return

    write_summary_csv(results, out / "readable_bag_summary.csv")
    write_html_table(results, out / "website_metrics_table.html")

    plot_comparison_bar(
        results,
        "metadata_duration_sec",
        "Duration (s)",
        "Recorded Duration by Readable Bag",
        out / "comparison_duration.png",
    )

    plot_comparison_bar(
        results,
        "metadata_total_messages",
        "Total messages",
        "Total Recorded Messages by Readable Bag",
        out / "comparison_total_messages.png",
    )

    plot_comparison_bar(
        results,
        "mean_positive_speed",
        "Mean positive speed (m/s)",
        "Mean Positive Commanded Speed",
        out / "comparison_mean_positive_speed.png",
    )

    plot_comparison_bar(
        results,
        "max_speed",
        "Max speed command (m/s)",
        "Maximum Commanded Speed",
        out / "comparison_max_speed.png",
    )

    plot_comparison_bar(
        results,
        "zero_speed_fraction_pct",
        "Zero-speed commands (%)",
        "Fraction of Stop or Near-Stop Commands",
        out / "comparison_zero_speed_fraction.png",
    )

    plot_comparison_bar(
        results,
        "mean_abs_steering",
        "Mean absolute steering (rad)",
        "Mean Steering Effort",
        out / "comparison_mean_abs_steering.png",
    )

    plot_comparison_bar(
        results,
        "steering_smoothness_rad_per_sec",
        "Mean steering rate (rad/s)",
        "Steering Smoothness Comparison",
        out / "comparison_steering_smoothness.png",
    )

    plot_comparison_bar(
        results,
        "min_front_distance",
        "Minimum front distance (m)",
        "Minimum Front LiDAR Clearance",
        out / "comparison_min_front_distance.png",
    )

    plot_topic_availability(results, out / "topic_availability.png")

    print_summary(results)

    print("\nDone.")
    print(f"Saved plots and summaries to: {out}")


if __name__ == "__main__":
    main()