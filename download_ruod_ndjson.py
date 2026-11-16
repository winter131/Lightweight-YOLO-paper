import os
import json
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

NDJSON_PATH = os.path.abspath("dataset/ruod-real-world-underwater-object-detection.ndjson")
OUTPUT_DIR = os.path.abspath("dataset/ruod")

def main():
    print(f"[*] Membaca metadata dari: {NDJSON_PATH}")
    if not os.path.exists(NDJSON_PATH):
        raise FileNotFoundError(f"File {NDJSON_PATH} tidak ditemukan!")

    # 1. Parse header (line 1)
    with open(NDJSON_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        meta = json.loads(first_line)

    class_names = meta.get("class_names", {})
    # Urutkan class_names berdasarkan id integer
    sorted_classes = {int(k): v for k, v in class_names.items()}
    sorted_classes = dict(sorted(sorted_classes.items()))
    nc = len(sorted_classes)

    print(f"[+] Terdeteksi {nc} kelas:")
    for cid, cname in sorted_classes.items():
        print(f"    - {cid}: {cname}")

    # Buat direktori output
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    # 2. Tulis data.yaml
    yaml_data = {
        "path": "dataset/ruod",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": sorted_classes
    }
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yaml_data, yf, default_flow_style=False, sort_keys=False)
    print(f"[+] File konfigurasi tersimpan di: {yaml_path}")

    # 3. Kumpulkan daftar download & tulis label
    download_tasks = []
    print("[*] Menyiapkan anotasi label dan antrian unduhan...")

    with open(NDJSON_PATH, "r", encoding="utf-8") as f:
        # Lewati header
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") != "image":
                continue

            file_name = item["file"]
            stem = os.path.splitext(file_name)[0]
            split = item.get("split", "train")
            # Standarisasi val -> valid jika ada
            if split == "val":
                split = "valid"

            url = item["url"]
            boxes = item.get("annotations", {}).get("boxes", [])

            # Tulis label file (.txt)
            label_file = os.path.join(OUTPUT_DIR, split, "labels", f"{stem}.txt")
            with open(label_file, "w", encoding="utf-8") as lf:
                for b in boxes:
                    # format: class_id x_center y_center width height
                    cid = int(b[0])
                    coords = [f"{coord:.6f}" for coord in b[1:]]
                    lf.write(f"{cid} {' '.join(coords)}\n")

            # Target path gambar
            img_file = os.path.join(OUTPUT_DIR, split, "images", file_name)
            download_tasks.append((url, img_file))

    total_tasks = len(download_tasks)
    print(f"[+] Total gambar dalam antrian: {total_tasks} citra")

    # 4. Multi-Threaded Downloader
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    def download_image(task):
        url, dest_path = task
        # Auto-resume: lewati jika file sudah ada dan ukurannya > 0
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return True, "skipped"

        for attempt in range(3):
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.content) > 0:
                    with open(dest_path, "wb") as img_f:
                        img_f.write(resp.content)
                    return True, "downloaded"
            except Exception:
                pass
        return False, "failed"

    print("[*] Memulai multi-threaded download (20 paralel workers)...")
    success_count = 0
    skipped_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_image, t): t for t in download_tasks}
        with tqdm(total=total_tasks, desc="RUOD Download", unit="img") as pbar:
            for future in as_completed(futures):
                ok, status = future.result()
                if ok:
                    if status == "downloaded":
                        success_count += 1
                    else:
                        skipped_count += 1
                else:
                    failed_count += 1
                pbar.update(1)

    print("\n" + "="*50)
    print("DOWNLOAD RUOD SELESAI!")
    print(f"-> Berhasil diunduh : {success_count}")
    print(f"-> Dilewati (Ada)   : {skipped_count}")
    print(f"-> Gagal            : {failed_count}")
    print("="*50)

if __name__ == "__main__":
    main()
