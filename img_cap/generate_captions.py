import anthropic
import base64
import csv
import json
import os
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Config
API_KEY = "YOUR_API_KEY_HERE"
MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 50
TEST_DIR = Path(__file__).parent / "test" / "test"
PROGRESS_FILE = Path(__file__).parent / "progress.json"
OUTPUT_FILE = Path(__file__).parent / "submission.csv"

PROMPT = """บรรยายภาพนี้เป็นภาษาไทยในหนึ่งประโยค ให้บรรยายสิ่งที่เห็นในภาพอย่างละเอียด รวมถึงวัตถุ สี การกระทำ และตำแหน่ง
ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษ ตอบแค่ประโยคบรรยายเพียงประโยคเดียว ไม่ต้องมีคำนำหน้าหรือคำอธิบายเพิ่มเติม"""

client = anthropic.Anthropic(api_key=API_KEY)
lock = threading.Lock()


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(results):
    with lock:
        snapshot = dict(results)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def caption_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    for attempt in range(5):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt * 2
            print(f"  Error: {e}, retrying in {wait}s...")
            time.sleep(wait)

    return ""


def main():
    # Get all test images
    image_files = sorted(TEST_DIR.glob("*.jpg"))
    print(f"Found {len(image_files)} test images")

    # Load existing progress
    results = load_progress()
    print(f"Loaded {len(results)} cached results")

    # Filter remaining
    remaining = [f for f in image_files if f.stem not in results]
    print(f"Remaining: {len(remaining)} images")

    if not remaining:
        print("All done! Writing CSV...")
    else:
        # Process with thread pool
        completed = len(results)
        total = len(image_files)
        start_time = time.time()

        def process_one(img_path):
            nonlocal completed
            caption = caption_image(img_path)
            image_id = img_path.stem
            with lock:
                results[image_id] = caption
                completed += 1
            elapsed = time.time() - start_time
            rate = (completed - len(results) + len(remaining)) / max(elapsed, 1)
            eta = (len(remaining) - (completed - (total - len(remaining)))) / max(rate, 0.01) / 60
            if completed % 10 == 0:
                print(f"  [{completed}/{total}] {elapsed:.0f}s elapsed, ~{eta:.1f}min remaining")
                save_progress(results)
            return image_id, caption

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            list(executor.map(process_one, remaining))

        save_progress(results)

    # Write submission CSV
    # Read sample to get the order
    sample_path = Path(__file__).parent / "sample_submission.csv"
    image_ids = []
    with open(sample_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(row["image_id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "caption"])
        for img_id in image_ids:
            caption = results.get(img_id, "")
            writer.writerow([img_id, caption])

    print(f"\nDone! Wrote {OUTPUT_FILE}")
    print(f"Total captions: {sum(1 for v in results.values() if v)}/{len(image_ids)}")


if __name__ == "__main__":
    main()
