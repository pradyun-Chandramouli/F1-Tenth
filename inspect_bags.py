from pathlib import Path
import yaml

ROOT = Path(r"C:\Users\cprad\Desktop\VS coding projects\ECE 484\ece484_rosbags")

for bag_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    metadata_path = bag_dir / "metadata.yaml"

    if not metadata_path.exists():
        print(f"\n{bag_dir.name}: no metadata.yaml found")
        continue

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    print(f"\n==============================")
    print(f"Bag: {bag_dir.name}")

    try:
        info = metadata["rosbag2_bagfile_information"]
        duration_ns = info.get("duration", {}).get("nanoseconds", None)
        message_count = info.get("message_count", None)

        if duration_ns is not None:
            print(f"Duration: {duration_ns / 1e9:.2f} sec")
        if message_count is not None:
            print(f"Messages: {message_count}")

        print("Topics:")
        topics = info.get("topics_with_message_count", [])
        for topic_entry in topics:
            meta = topic_entry["topic_metadata"]
            name = meta["name"]
            msg_type = meta["type"]
            count = topic_entry["message_count"]
            print(f"  {name:40s} {msg_type:45s} count={count}")

    except Exception as e:
        print(f"Could not parse metadata cleanly: {e}")