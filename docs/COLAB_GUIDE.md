**Hướng dẫn thực hiện trên Colab, Drive và Git**

Đọc cùng [TRAINING_PLAN.md](TRAINING_PLAN.md). Mục tiêu: đọc số trên màn hình cân, hỗ trợ ảnh camera gốc và ảnh có phần zoom.

Luồng thực tế đã xác nhận: camera gắn tại trạm, người dùng bấm chụp và nhận ảnh như backup. Train reader trên các ảnh chụp này; sau train, ứng dụng gửi ảnh của từng lần bấm cho model đã triển khai để lấy số. Colab phục vụ huấn luyện; nơi chạy model tại trạm/backend sẽ chọn theo phần cứng và độ trễ thực tế.

Code huấn luyện và bốn notebook đã được tạo trong thư mục `scale-ocr/` của workspace. Index, split và danh sách 400 ảnh pilot đã sinh từ backup thật. **400/400 nhãn đã được người dùng duyệt**; bản cố định ở `data/v001/release/`. Dataset v001 đã xuất và đóng gói tại `data/v001/dataset.zip`. Chưa có checkpoint hay kết quả accuracy; chưa có smoke test thành công trên một runtime Colab.

**Khi Google Drive mount báo lỗi 400/403:** dùng [04_train_reader_upload.ipynb](../notebooks/04_train_reader_upload.ipynb), mở trên Colab với GPU T4. Notebook này tải bốn file trực tiếp từ máy tính lên `/content` bằng hộp chọn tệp, nên không dùng quyền mount Drive. Bốn file đã đặt chung tại `E:\backup-tramcan\colab_upload_v001`: `dataset.zip`, `dataset.zip.sha256`, `labels_reviewed_v001.zip`, `labels_reviewed_v001.zip.sha256`. Chạy lần lượt cell bootstrap, cell upload, cell cài PaddleOCR, cell tạo config, rồi cell smoke test 1 epoch. **Không bấm Run all** vì cell sau smoke test sẽ train toàn bộ. Sau khi smoke test thành công, gửi log để kiểm tra trước khi chạy run chính. Notebook ghi checkpoint vào `/content/scale-runs`; khi train kết thúc hoặc bị dừng sau ít nhất một epoch, chạy cell backup và tải ZIP checkpoint về máy. Colab xóa `/content` khi runtime kết thúc, nên phải xác nhận file đã tải xong trước khi ngắt; sau đó tải ZIP này lên Drive bằng trình duyệt. Cell cuối của notebook hỗ trợ upload ZIP đó trở lại để resume. Đường này là phương án dự phòng và chưa được chạy thử trên Colab GPU thực tế.

**Bước tiếp theo ngay bây giờ:** upload `data/v001/dataset.zip` và `dataset.zip.sha256` vào `MyDrive/tram-can/datasets/v001/`; upload `labels_reviewed_v001.zip` và checksum vào `MyDrive/tram-can/labels/v001/`. Sau đó mở `notebooks/03_train_reader.ipynb` trên Colab bằng đúng tài khoản Drive chứa thư mục `tram-can`. Repo GitHub public: `https://github.com/huybitvvt/scale-ocr`, không cần GitHub token. Điền commit code đã push vào `PROJECT_COMMIT` trước run chính. ZIP gốc đã nằm trong `MyDrive/tram-can/` theo ảnh Drive; notebook sẽ tự xác minh checksum của dataset đã xử lý sau khi mount.

**0. Kiểm tra luồng bấm chụp trước khi thu thêm dữ liệu**

1. Tại từng camera, chụp một lô pilot ở các mức cân và điều kiện ánh sáng khác nhau; lưu được ảnh nguồn và ảnh ghép tương ứng nếu phần mềm hỗ trợ. Đối chiếu trực tiếp dấu chấm và toàn bộ chữ số với màn hình thật.
2. Kiểm tra vùng zoom được tạo từ đúng frame nguồn của lần bấm. Nếu nó lấy từ preview hoặc ảnh khác, sửa bước ghép/truy vết trước khi dùng cặp ảnh đó làm nhãn chung.
3. Tạo `capture_id` cho mỗi lần bấm và giữ `event_key` cho phiên cân. Lưu `camera_id`, thời điểm ảnh, loại ảnh core/product, đường dẫn ảnh nguồn/ảnh ghép và phiên bản ROI/layout. Nếu chưa có thời điểm chụp thật thì ghi thời điểm nhận ảnh với tên trường đúng nghĩa.
4. Khi camera giữ góc ổn định, vẽ ROI đủ toàn màn hình số trên ảnh nguồn. Thử ROI trên các lần chụp khác, các số có nhiều chữ số hơn và các ảnh bị rung/lệch trước khi dùng tự động.
5. Chọn tập nhãn pilot đa dạng, tạo crop từ ROI và đo reader pretrained/fine-tune. Train detector ở bước 10 nếu đo được ROI không đủ ổn định hoặc để so sánh cải thiện; không cần chờ train detector mới bắt đầu reader.
6. Ở ứng dụng, kết quả `ok` hiện số + ảnh tương ứng; `review`/`unreadable` để trống số và cho kiểm tra/chụp lại. Nếu có người sửa, lưu riêng dự đoán ban đầu và số đã duyệt.

Chưa có code ứng dụng chụp trong thư mục này. Các kiểm tra tích hợp trên là việc cần thực hiện trong ứng dụng hiện có, không phải tính năng đã được triển khai.

**1. Tạo nơi lưu dữ liệu trên Google Drive**

Hai file đã được anh upload trực tiếp vào thư mục `MyDrive/tram-can/`:

```text
E:\backup-tramcan\cloudinary_backup_2026-09-27.zip
E:\backup-tramcan\cloudinary_backup_2026-09-27.zip.sha256
```

Chờ upload hoàn tất rồi kiểm tra kích thước ZIP là 3.607.981.295 byte. SHA-256 trong file checksum hiện tại:

```text
07c312b493af9bbab51ff1fec858211c203cc5aa699e865a90f3ebf84ac0c88a
```

Giữ bản ZIP này làm dữ liệu gốc. Bộ dữ liệu đã gán nhãn sẽ được đóng gói riêng trong `MyDrive/tram-can/datasets/v001/`, không ghi đè backup. Ảnh chụp Drive xác nhận tên file và kích thước hiển thị; bước SHA-256 trong notebook mới xác nhận nội dung chính xác. Khi mount Drive trong Colab, chọn tài khoản Google chứa thư mục `tram-can` (ảnh Drive hiện avatar P).

**2. Đưa code đã chuẩn bị lên Git**

Repo local nằm tại `E:\backup-tramcan\scale-ocr`. Remote public tại `https://github.com/huybitvvt/scale-ocr`. `.gitignore` đã loại ảnh, ZIP, nhãn, dữ liệu xuất và checkpoint. Sau khi sửa code, chạy PowerShell trong thư mục đó:

```powershell
Set-Location E:\backup-tramcan\scale-ocr
git status --short
git push origin main
```

Không đưa ảnh backup, nhãn hoặc token vào commit. Repo public clone trực tiếp trong Colab.

**3. Tạo notebook và chọn runtime**

Tách notebook thành chuẩn bị dữ liệu, train detector, train reader, đánh giá. Notebook nên gọi code/config trong repo; tránh chứa toàn bộ logic ở các cell khó theo dõi.

Trong Colab, chọn GPU cho notebook train. Phần chuẩn bị index/gán nhãn có thể dùng CPU. T4 nếu được cấp là điểm bắt đầu hợp lý cho model nhỏ và batch vừa; luôn đọc VRAM thực tế trước khi chọn batch. Colab không đảm bảo luôn có cùng GPU hoặc thời lượng phiên. [Thông tin tài nguyên Colab](https://research.google.com/colaboratory/faq.html).

Cell kiểm tra môi trường:

```python
import sys
import shutil
import subprocess

print(sys.version)
print(shutil.disk_usage('/content'))
subprocess.run(['nvidia-smi'], check=False)
```

Cần đĩa local đủ cho ZIP + giải nén + crop + checkpoint tạm. Với backup hiện tại, chừa khoảng 15–25 GB trước khi tạo thêm nhiều phiên bản dataset là mức dự phòng thực dụng; kiểm tra dung lượng thật thay vì giả định runtime nào cũng giống nhau.

**4. Mount Drive và đưa dữ liệu về đĩa local**

```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/tram-can')
LOCAL_ROOT = Path('/content/scale-data')
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
```

Copy archive, tính checksum rồi giải nén. Cell này đọc dữ liệu, không cần Cloudinary API key:

```python
import hashlib
import shutil
import zipfile

archive_name = 'cloudinary_backup_2026-09-27.zip'
source_zip = DRIVE_ROOT / archive_name
source_checksum = DRIVE_ROOT / (archive_name + '.sha256')
local_zip = LOCAL_ROOT / archive_name

assert source_zip.is_file(), source_zip
assert source_checksum.is_file(), source_checksum
shutil.copy2(source_zip, local_zip)

expected_sha = source_checksum.read_text().split()[0].lower()
digest = hashlib.sha256()
with local_zip.open('rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == expected_sha, 'ZIP checksum mismatch'

raw_dir = LOCAL_ROOT / 'raw'
raw_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(local_zip) as archive:
    archive.extractall(raw_dir)

RAW_ROOT = raw_dir / 'cloudinary_backup_2026-09-27'
assert (RAW_ROOT / 'manifest.json').is_file()
print('Verified archive and extracted to:', RAW_ROOT)
```

Trong những phiên sau, có thể dùng archive dataset đã xử lý thay vì giải nén toàn bộ backup lại. Cũng cần kiểm checksum của archive đó. Train đọc ảnh trong `/content`, không mở hàng nghìn ảnh trực tiếp qua thư mục Drive mount. [Hướng dẫn I/O của Colab](https://research.google.com/colaboratory/faq.html).

**5. Clone code và khóa phiên bản**

Cell này dành cho repo hiện tại. Thay `PROJECT_COMMIT` trước run chính:

```python
import subprocess
from pathlib import Path

REPO_URL = 'https://github.com/huybitvvt/scale-ocr.git'
PROJECT_COMMIT = 'REPLACE_WITH_FULL_COMMIT_SHA'
CODE_ROOT = Path('/content/scale-ocr')

assert PROJECT_COMMIT != 'REPLACE_WITH_FULL_COMMIT_SHA'
if not CODE_ROOT.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(CODE_ROOT)], check=True)
subprocess.run(['git', '-C', str(CODE_ROOT), 'fetch', 'origin'], check=True)
subprocess.run(
    ['git', '-C', str(CODE_ROOT), 'checkout', '--detach', PROJECT_COMMIT],
    check=True,
)
print(subprocess.check_output(
    ['git', '-C', str(CODE_ROOT), 'rev-parse', 'HEAD'], text=True
).strip())
```

Repo hiện public nên notebook clone qua HTTPS trực tiếp. Nếu sau này đổi về private, xem biến `GITHUB_SECRET_NAME` trong notebook và lưu token chỉ đọc repo trong Colab Secrets; không ghi token vào code.

Giữ cố định commit khi chạy một thí nghiệm. Không `git pull` tùy ý giữa lúc train rồi tiếp tục ghi cùng run ID.

**6. Sinh index, split và pilot bằng code trong repo**

Sau bước clone và giải nén, cài code rồi chạy CLI. Trong notebook [01_prepare.ipynb](../notebooks/01_prepare.ipynb), các lệnh tương đương đã được viết sẵn. Nếu chạy cell thủ công theo phần 4–5 ở trên:

```python
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(CODE_ROOT)], check=True)
WORK_ROOT = LOCAL_ROOT / 'work' / 'v001'
WORK_ROOT.mkdir(parents=True, exist_ok=True)
for command in [
    ['index', '--raw-root', str(RAW_ROOT), '--out', str(WORK_ROOT / 'index.csv')],
    ['split', '--index', str(WORK_ROOT / 'index.csv'), '--out', str(WORK_ROOT / 'split.csv')],
    ['pilot', '--index', str(WORK_ROOT / 'index.csv'), '--split', str(WORK_ROOT / 'split.csv'),
     '--out', str(WORK_ROOT / 'pilot.csv'), '--count', '400'],
]:
    subprocess.run([sys.executable, '-m', 'scale_ocr', *command], check=True)
```

Với backup hiện tại, index có 9.260 ảnh và 8.397 nội dung khác nhau. Split sơ bộ: train 5.387, val 1.954, test 1.715, exclude 204. Pilot có 400 ảnh. Rà soát ảnh và nhóm event trước khi khóa split; thời gian trong đường dẫn là thời gian đã ghi lên Cloudinary, chưa xác thực là thời điểm camera chụp. Không đưa CSV/ảnh vào Git.
**7. Làm bộ nhãn pilot rồi xuất dataset**

Trên Windows, chạy CLI từ `E:\backup-tramcan\scale-ocr` với index/split/pilot đã sinh local. Browser sẽ mở tại `http://127.0.0.1:8765/`:

```powershell
python -m scale_ocr annotate --pilot data\v001\pilot.csv --raw-root ..\cloudinary_backup_2026-09-27 --out data\v001\annotations.jsonl
```

400 nhãn đã duyệt và được chốt tại `data/v001/release/`. Lệnh đã chạy để xuất và đóng gói:

```powershell
python -m scale_ocr export --raw-root ..\cloudinary_backup_2026-09-27 --index data\v001\release\index.csv --split data\v001\release\split.csv --annotations data\v001\release\annotations.jsonl --out data\v001\prepared_release
python -m scale_ocr package --source data\v001\prepared_release --out data\v001\dataset.zip
```

Upload cả `dataset.zip` và `dataset.zip.sha256` sang Drive. ZIP `labels_reviewed_v001.zip` chứa nhãn/index/split/pilot cố định cùng manifest SHA-256; upload nó và checksum sang thư mục `labels/v001/`. Dataset chứa `evaluation_manifest.csv` cho đủ 400 ảnh, gồm cả 5 ảnh không thể tạo crop train.

Tạo khoảng 300–500 ảnh pilot đa dạng để gán nhãn. Với ảnh ghép có layout đã xác minh, tạo scene view và zoom view; ảnh chưa nhận diện được layout giữ nguyên full view. Mỗi view phải ghi lại asset cha, tọa độ trên ảnh cha và group ID.

Chưa nên viết quy tắc tự động cắt mọi ảnh ở y=900. Trước tiên xem mẫu của từng layout/kích thước, kiểm tra ảnh sau cắt, rồi lưu quy tắc vào `cameras.yaml` hoặc cấu hình layout. Cần giữ fallback toàn ảnh.

Với ảnh chụp mới, ưu tiên lấy crop từ ảnh nguồn cùng lần bấm, trước khi ghép zoom. Khi chỉ có ảnh backup, tạo crop từ view đã xác minh. Ghi `capture_id` khi có; mọi crop và lần chụp lại của cùng phiên phải giữ cùng split. Tên gateway trong manifest chưa chứng minh quan hệ một-một với camera vật lý.

Mỗi vùng số cần bbox, chuỗi text và readability. Một nhãn `readable` phải được nhập đủ cả dấu chấm và dấu âm nếu có. Vùng bị che không có text để train OCR dù người gán nhãn biết số từ ảnh khác.

Sau khi duyệt, cần có ba sản phẩm:

| Sản phẩm | Nội dung |
|---|---|
| `annotations.jsonl` | Nhãn đầy đủ, source/view/group, bbox, text, readability, người duyệt |
| `split.csv` | Mỗi asset/view có group ID và train/val/test cố định |
| Dataset xuất ra | Detection YOLO + crop recognition + danh mục ảnh âm/không đọc được |

Dataset v001 thực tế: 137 crop train, 131 crop val, 123 crop test. `evaluation_manifest.csv` có 140/135/125 ảnh theo train/val/test. `gateway-03` chỉ có 5 ảnh train và không có ảnh val/test, vì vậy chưa thể kết luận chất lượng cho gateway này.

Dataset detection sau export:

```text
/content/scale-data/prepared/v001/detector/
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
  data.yaml
```

Mỗi dòng YOLO là `class_id center_x center_y width height`, các tọa độ chia cho chiều rộng/cao của đúng view. Ví dụ công thức chuyển bbox pixel:

```python
def bbox_to_yolo(x1, y1, x2, y2, image_width, image_height):
    assert 0 <= x1 < x2 <= image_width
    assert 0 <= y1 < y2 <= image_height
    return (
        (x1 + x2) / (2 * image_width),
        (y1 + y2) / (2 * image_height),
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )
```

`data.yaml`:

```yaml
path: /content/scale-data/prepared/v001/detector
train: images/train
val: images/val
test: images/test
names:
  0: weight_display
```

Ảnh âm được xác nhận không có vùng mục tiêu dùng file nhãn rỗng. Không dùng nhãn rỗng cho ảnh có màn hình rõ nhưng chưa được gán nhãn. [Định dạng dataset YOLO](https://docs.ultralytics.com/datasets/detect).

Dataset recognition:

```text
/content/scale-data/prepared/v001/recognizer/
  crops/train/*.png
  crops/val/*.png
  crops/test/*.png
  train.txt
  val.txt
  test.txt
```

Mỗi dòng `train.txt` gồm đường dẫn crop tương đối và đáp án, ngăn bằng TAB thật. Ví dụ tên file dưới đây chỉ minh họa định dạng:

```text
crops/train/example_a.png<TAB>7.04
crops/train/example_b.png<TAB>0.00
crops/train/example_c.png<TAB>13.04
```

Trong code phải ghi `f'{crop_path}\t{text}\n'`; không ghi chuỗi ký tự `<TAB>`. Chỉ đưa crop `readable` đã duyệt vào các file nhận dạng chuỗi. Giữ mẫu khác trong bộ kiểm tra chất lượng và end-to-end. [Chuẩn bị dữ liệu nhận dạng PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_train/recognition.en.md).

**8. Kiểm tra dataset trước GPU**

Kiểm tra bằng code: file ảnh/nhãn tồn tại; bbox nằm trong ảnh; text là string; mọi dấu hợp lệ; split của crop giống ảnh cha; không có group ID hoặc ảnh thực trùng chéo tập; không có ảnh synthetic trong val/test. Với ảnh có nhiều vùng số, bảo đảm target nào cũng được gán nhãn.

Ví dụ kiểm tra nhóm trên bảng split đã tạo:

```python
split_table = pd.read_csv(WORK_ROOT / 'split.csv', dtype=str)
assert set(split_table['split']).issubset({'train', 'val', 'test'})
assert split_table['group_id'].notna().all()
assert split_table.groupby('group_id')['split'].nunique().max() == 1
```

Điều này chỉ có giá trị khi `group_id` đã được tạo đúng. Nó không tự phát hiện ảnh gần trùng chưa được liên kết.

Xem trực tiếp ít nhất một lưới bbox/crop theo từng camera và từng dạng đầu vào; ưu tiên crop số 0, chữ số cuối và dấu chấm sát biên. Tính histogram số chữ số, vị trí dấu chấm, giá trị, camera và readability. Không đoán phạm vi giá trị chỉ từ vài ảnh mẫu.

Đóng gói dataset đã xuất và nhãn/split thành `dataset.zip`, tạo SHA-256, upload vào Drive `datasets/v001/`. Những notebook train sau dùng đúng bundle này. Chỉ tăng phiên bản v002 khi nhãn/split/nội dung thay đổi.

**9. Chạy baseline trước fine-tune**

Đo OCR pretrained trên crop chuẩn của validation để biết mức xuất phát. Đo tiếp trên crop ROI/detector để biết lỗi đến từ định vị hay reader. Với bộ đọc bảy đoạn, dùng chính cùng crop và cùng split.

Ví dụ inference bằng module nhận dạng của PaddleOCR sau khi cài môi trường ở bước 12:

```python
from paddleocr import TextRecognition

reader = TextRecognition(model_name='PP-OCRv5_server_rec', device='gpu:0')
pilot_crop = '/content/scale-data/prepared/v001/recognizer/crops/val/REPLACE.png'
assert Path(pilot_crop).is_file()
for result in reader.predict(input=pilot_crop, batch_size=1):
    result.print()
```

Tên model được chỉ rõ để tránh thay đổi mặc định khi thư viện nâng cấp. In được một đáp án chỉ là smoke test; cần xuất dự đoán cho toàn validation rồi tính exact match.

**10. Train detector trên Colab**

Đây là nhánh so sánh/bổ sung cho định vị. Với camera gắn cố định và ROI đã được nghiệm thu, có thể đi thẳng từ dataset crop sang bước 12–13 để fine-tune reader trước. Chạy nhánh detector khi cần xử lý góc máy thay đổi hoặc kết quả validation cho thấy ROI làm mất số/dấu chấm. Khi bỏ qua notebook detector, notebook reader cần tự chạy các bước mount Drive, clone code, chuẩn bị dataset và ghi môi trường cho run của nó.

Nên dùng runtime riêng cho detector và reader để dễ xử lý dependency. Trong runtime detector, cài bản thư viện đã khóa trong repo. Ở lần dò môi trường đầu tiên có thể cài `ultralytics`, kiểm tra GPU và một epoch rồi ghi lại phiên bản; từ run tiếp theo dùng lock file đã kiểm chứng.

Cell bootstrap ban đầu, chưa phải lock tái lập:

```python
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], check=True)
```

Cell kiểm tra GPU và train, chỉ chạy sau khi dataset đã có nhãn:

```python
from datetime import datetime, timezone
import torch
from ultralytics import YOLO

assert torch.cuda.is_available(), 'GPU runtime is required for this training run'
print(torch.cuda.get_device_name(0))
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)

detector_yaml = LOCAL_ROOT / 'prepared/v001/detector/data.yaml'
assert detector_yaml.is_file(), 'Export the labeled dataset first'
run_id = 'det-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

model = YOLO('yolo26s.pt')
model.train(
    data=str(detector_yaml),
    imgsz=1024,
    epochs=100,
    batch=4,
    device=0,
    workers=2,
    optimizer='AdamW',
    lr0=0.001,
    patience=20,
    seed=42,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.0,
    mixup=0.0,
    degrees=5.0,
    translate=0.03,
    scale=0.1,
    hsv_h=0.0,
    hsv_s=0.2,
    hsv_v=0.2,
    project=str(DRIVE_ROOT / 'runs'),
    name=run_id,
    save=True,
    save_period=5,
    exist_ok=False,
)
```

Ảnh đọc local; run/checkpoint ghi Drive để giữ được sau khi runtime mất. Đây là cách đơn giản cho pilot. Khi pipeline lớn hơn, có thể ghi checkpoint local rồi copy định kỳ bằng file tạm và rename; vẫn phải đồng bộ trong khi train, không chỉ cuối phiên. Giữ checkpoint định kỳ để phục hồi nếu file cuối bị gián đoạn lúc ghi.

Trước run dài, dùng run ID riêng để thử 1 epoch; sau đó kiểm tra ảnh prediction/crop. Các tham số ở trên là cấu hình khởi đầu đề xuất. Không đánh giá thành công chỉ vì loss giảm hoặc mAP cao: detector có thể cắt mất dấu chấm dù IoU vẫn cao.

Khi nối lại phiên, mount Drive, đưa cùng dataset về cùng đường dẫn local, checkout cùng code/version, rồi:

```python
from ultralytics import YOLO

last_checkpoint = DRIVE_ROOT / 'runs' / 'REPLACE_RUN_ID' / 'weights/last.pt'
assert last_checkpoint.is_file()
YOLO(str(last_checkpoint)).train(resume=True)
```

Resume cần checkpoint có trạng thái train; dùng `best.pt` để suy luận/chọn model, không thay thế tùy ý `last.pt` khi mục tiêu là tiếp tục optimizer/scheduler. [Train và resume Ultralytics](https://docs.ultralytics.com/modes/train).

**11. Khóa dependency và ghi thông tin run**

Sau khi môi trường chạy được, lưu `pip freeze`, Python version, GPU/driver, commit code và config vào thư mục run. Từ đó xây lock file riêng cho detector và reader, bao gồm nguồn wheel GPU cần thiết; `pip freeze` một mình chưa mô tả mọi điều kiện hệ thống.

```python
import json
import subprocess
import sys

run_dir = DRIVE_ROOT / 'runs' / run_id
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / 'pip-freeze.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
)
(run_dir / 'nvidia-smi.txt').write_text(
    subprocess.check_output(['nvidia-smi'], text=True)
)
(run_dir / 'run_manifest.json').write_text(json.dumps({
    'project_commit': PROJECT_COMMIT,
    'dataset_version': 'v001',
    'raw_archive_sha256': expected_sha,
    'python': sys.version,
    'seed': 42,
    'note': 'Add hashes of annotations, split, dataset bundle, and config before a formal run',
}, indent=2))
```

Các hash nhãn/split/dataset/config phải được bổ sung trước run chính thức. Không dùng notebook output chứa token làm log môi trường.

**12. Cài môi trường reader và lấy source PaddleOCR**

Mở runtime GPU mới cho reader, chạy lại mount Drive và chuẩn bị dataset local. PaddleOCR có phần inference package và phần source huấn luyện; cài package inference không thay thế việc lấy source/config train.

Tài liệu PaddlePaddle hiện liệt kê wheel `paddlepaddle-gpu==3.3.0` cho CUDA 12.6. Kiểm tra GPU, driver, Python và runtime Colab thực tế; chỉ dùng cell dưới nếu wheel phù hợp. Với CUDA khác, chọn đúng wheel theo [hướng dẫn cài PaddlePaddle chính thức](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html). Sau cài đặt, chạy `paddle.utils.run_check()` như notebook.

```python
import subprocess
import sys

subprocess.run([
    sys.executable, '-m', 'pip', 'install', 'paddlepaddle-gpu==3.3.0',
    '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/',
], check=True)
```

Clone source chính thức. `release/3.5` là nhánh tham khảo trong hướng dẫn upstream hiện được kiểm tra; sau khi smoke test thành công, ghi commit cụ thể vào repo dự án và checkout đúng commit đó cho các run tiếp theo. [Cài dependency huấn luyện PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/installation.en.md).

```python
from pathlib import Path
import subprocess
import sys

PADDLE_ROOT = Path('/content/PaddleOCR')
if not PADDLE_ROOT.exists():
    subprocess.run([
        'git', 'clone', '--depth', '1', '--branch', 'release/3.5',
        'https://github.com/PaddlePaddle/PaddleOCR.git', str(PADDLE_ROOT),
    ], check=True)
print('PaddleOCR commit:', subprocess.check_output(
    ['git', '-C', str(PADDLE_ROOT), 'rev-parse', 'HEAD'], text=True
).strip())
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-r',
    str(PADDLE_ROOT / 'requirements.txt'),
], check=True)
```

Kiểm tra GPU sau cài. Nếu thay thư viện trong kernel đã import trước đó, restart runtime/kernel theo yêu cầu của Colab rồi kiểm tra lại:

```python
import paddle

print('Paddle:', paddle.__version__)
assert paddle.is_compiled_with_cuda()
assert paddle.device.cuda.device_count() > 0
paddle.set_device('gpu:0')
paddle.utils.run_check()
```

Nếu muốn dùng `TextRecognition` cho baseline, cài thêm package `paddleocr` tương thích trong môi trường đó rồi pin version sau smoke test. Dùng môi trường riêng cho inference nếu dependency xung đột; không nâng/hạ ngẫu nhiên nhiều thư viện trong cùng run.

**13. Fine-tune reader từ cấu hình gốc**

Tải pretrained **training weights**, không dùng nhầm thư mục inference model:

```python
from urllib.request import urlretrieve

PRETRAIN_ROOT = LOCAL_ROOT / 'pretrained'
PRETRAIN_ROOT.mkdir(exist_ok=True)
pretrained = PRETRAIN_ROOT / 'PP-OCRv5_server_rec_pretrained.pdparams'
urlretrieve(
    'https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/'
    'PP-OCRv5_server_rec_pretrained.pdparams',
    pretrained,
)
```

Đường dẫn training weights và lệnh train theo [hướng dẫn reader chính thức](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md). Sau khi tải, ghi SHA-256 của weights vào run manifest.

Tạo config dự án bằng cách sửa bản config gốc thay vì viết lại architecture. Giữ dictionary, head, max length và input shape của baseline tương thích pretrained. Các trường batch của sampler và loader đều cần điều chỉnh.

```python
from datetime import datetime, timezone
import yaml

rec_root = LOCAL_ROOT / 'prepared/v001/recognizer'
assert (rec_root / 'train.txt').is_file()
assert (rec_root / 'val.txt').is_file()
base_config = PADDLE_ROOT / 'configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml'
cfg = yaml.safe_load(base_config.read_text())

rec_run_id = 'rec-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
rec_run_dir = DRIVE_ROOT / 'runs' / rec_run_id
rec_run_dir.mkdir(parents=True, exist_ok=False)

cfg['Global'].update({
    'use_gpu': True,
    'distributed': False,
    'epoch_num': 50,
    'pretrained_model': str(pretrained),
    'checkpoints': None,
    'save_model_dir': str(rec_run_dir / 'checkpoints'),
    'save_epoch_step': 5,
    'eval_batch_step': [0, 100],
})
cfg['Optimizer']['lr']['learning_rate'] = 0.0001

for section, label_name in [('Train', 'train.txt'), ('Eval', 'val.txt')]:
    cfg[section]['dataset']['data_dir'] = str(rec_root)
    cfg[section]['dataset']['label_file_list'] = [str(rec_root / label_name)]
    cfg[section]['loader']['num_workers'] = 2
    cfg[section]['loader']['batch_size_per_card'] = 16

cfg['Train']['sampler']['first_bs'] = 16
cfg['Train']['sampler']['fix_bs'] = True
cfg['Train']['sampler']['scales'] = [[320, 48]]

# Baseline đầu tiên tắt RecAug; bổ sung augmentation nhẹ ở run riêng.
cfg['Train']['dataset']['transforms'] = [
    transform for transform in cfg['Train']['dataset']['transforms']
    if 'RecAug' not in transform
]

rec_config = rec_run_dir / 'recognizer.yaml'
rec_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
print(rec_config)
```

Các key trên được đối chiếu với [config PP-OCRv5 server](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml). Nếu checkout phiên bản khác làm đổi cấu trúc, kiểm tra config trước khi chạy; không bỏ qua `KeyError` và tiếp tục bằng tham số mặc định.

Lệnh train chạy trong thư mục source PaddleOCR để những đường dẫn dictionary tương đối được tìm đúng:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, 'tools/train.py', '-c', str(rec_config),
], cwd=PADDLE_ROOT, check=True)
```

Notebook `03_train_reader.ipynb` đã có cell smoke test 1 epoch với 64 crop train và 16 crop val, dùng config và checkpoint riêng trong `/content`. Chạy cell đó trước cell train chính; kiểm tra log báo pretrained được nạp đúng. Một epoch chỉ xác nhận pipeline chạy được, chưa chứng minh mô hình đã học tốt hoặc tổng quát hóa. Nếu head bị bỏ qua do đổi dictionary, đó là thay đổi cần chủ động kiểm soát.

Không tự cho rằng metric mặc định của framework khớp tiêu chí nghiệp vụ. Bộ đánh giá riêng phải giữ dấu chấm, dấu âm và số 0 cuối; phân biệt exact string với equal numeric value.

Resume reader dùng `Global.checkpoints` trỏ đến prefix checkpoint có đủ tham số, optimizer và trạng thái. Kiểm tra tên file thực tế trong `checkpoints/`; không thêm bừa suffix `.pdparams` vào prefix nếu công cụ đang yêu cầu prefix. Ví dụ mẫu sau cần thay prefix bằng checkpoint có thật:

```python
checkpoint_prefix = rec_run_dir / 'checkpoints' / 'REPLACE_ACTUAL_CHECKPOINT_PREFIX'
assert checkpoint_prefix.with_suffix('.pdparams').is_file()
assert checkpoint_prefix.with_suffix('.pdopt').is_file()
subprocess.run([
    sys.executable, 'tools/train.py', '-c', str(rec_config), '-o',
    'Global.checkpoints=' + str(checkpoint_prefix),
], cwd=PADDLE_ROOT, check=True)
```

Khôi phục biến `rec_run_dir`/`rec_config` của run cũ sau khi reconnect; đừng tạo run mới rồi tìm checkpoint ở đó. Dùng cùng code, dataset và config cho resume.

**14. Đánh giá và chọn mô hình**

Tạo `predictions.csv` có ít nhất `sample_id`, `group_id`, `gateway`, `input_kind`, `readability`, `gt_text`, `pred_text`, `status`, `reason`, `latency_ms`. Một dòng cho mỗi đầu vào được đánh giá. Không bỏ dòng dự đoán thất bại.

Cell ví dụ tính ba chỉ tiêu cơ bản từ file dự đoán; file này phải được pipeline inference tạo ra, hiện chưa có:

```python
import pandas as pd

predictions_path = rec_run_dir / 'predictions.csv'
pred = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
assert pred['sample_id'].is_unique
readable = pred['readability'].eq('readable')
accepted = pred['status'].eq('ok')
correct = readable & pred['pred_text'].eq(pred['gt_text'])

def ratio(numerator, denominator):
    return float(numerator) / int(denominator) if int(denominator) else None

print('Readable exact match:', ratio((correct & accepted).sum(), readable.sum()))
print('Accepted accuracy:', ratio((correct & accepted).sum(), accepted.sum()))
print('Coverage all inputs:', ratio(accepted.sum(), len(pred)))
print('False accepts on unreadable:', ratio((accepted & ~readable).sum(), (~readable).sum()))
```

Ở đây `Readable exact match` yêu cầu kết quả cuối cùng được chấp nhận và đúng. Có thể báo thêm exact match thô của reader trước khi áp ngưỡng để chẩn đoán; phải đặt tên khác và ghi rõ mẫu số.

Xuất cùng các chỉ tiêu theo camera và input kind; báo số mẫu mỗi nhóm. Tính CER, lỗi dấu chấm/dấu âm và numeric error bằng evaluator đã kiểm chứng. Với `Decimal`, hãy truyền string gốc đã được xác nhận định dạng; không chuyển float rồi mới tạo Decimal.

So sánh ba chế độ trên cùng parent test đã giữ riêng: chỉ dùng scene/raw, dùng composite toàn pipeline, và chỉ dùng zoom để chẩn đoán. Các kết quả này phụ thuộc nhau vì chung ảnh cha; không cộng số mẫu của ba chế độ để tăng cỡ mẫu thống kê.

Chọn checkpoint và ngưỡng trên validation/calibration. Khi mọi cấu hình đã khóa mới chạy test 26–27/09 và dữ liệu mới. Chưa đủ mẫu thì báo kết quả quan sát và giới hạn bằng chứng, không làm tròn lên “đạt 99,9%”.

**15. Export, lưu kết quả và kiểm tra sau export**

Export reader theo công cụ của đúng phiên bản PaddleOCR. Ví dụ cho checkpoint `best_accuracy` khi checkpoint đó đã tồn tại:

```python
best_prefix = rec_run_dir / 'checkpoints' / 'best_accuracy'
assert best_prefix.with_suffix('.pdparams').is_file()
export_dir = DRIVE_ROOT / 'exports' / rec_run_id
subprocess.run([
    sys.executable, 'tools/export_model.py', '-c', str(rec_config), '-o',
    'Global.pretrained_model=' + str(best_prefix),
    'Global.save_inference_dir=' + str(export_dir),
], cwd=PADDLE_ROOT, check=True)
```

Sau export, dùng API inference tương thích với model xuất để chạy lại một bộ ảnh cố định. So sánh chuỗi từng ảnh với model gốc và với nhãn; kiểm tra quy tắc resize/pad, kênh màu, dictionary và decoding giống nhau. Nếu đổi FP16/INT8 hoặc engine chạy, đánh giá lại trước khi thay phiên bản đang dùng.

Mỗi phiên bản model cần đủ: detector, reader, dictionary, cấu hình preprocessing/layout, ngưỡng chấp nhận, version metadata, báo cáo test và danh sách camera đã đánh giá. Chỉ có file weights là chưa đủ để tái tạo pipeline.

**16. Trạng thái code và phần cần triển khai tiếp**

| Phần | Trạng thái | Bước kiểm chứng/triển khai tiếp |
|---|---|---|
| `inventory.py`, `splits.py`, `pilot.py` | Đã có; chạy trên backup thật | Xác nhận nhóm event và 204 ảnh exclude trước khi mở rộng dữ liệu |
| `annotations.py`, `export.py`, `package.py` | 400 nhãn đã duyệt; v001 đã xuất trên ảnh thật | Upload hai ZIP và checksum; kiểm tra crop trên Colab trước train |
| `evaluate.py` | Đã có evaluator cho CSV dự đoán | Tạo pipeline inference sinh CSV, kiểm tra thêm báo cáo theo camera/layout |
| `notebooks/01_prepare.ipynb` | Đã tạo | Chỉ cần khi muốn tạo lại index/pilot từ backup gốc |
| `notebooks/02_train_detector.ipynb`, `03_train_reader.ipynb` | Đã tạo | Upload dataset rồi chạy smoke test reader trước run chính |
| ROI theo camera, baseline OCR, pipeline inference và kiểm tra chất lượng | Chưa có | Triển khai sau khi rà soát nhãn/crop; đo với validation |
| Adapter trong ứng dụng chụp | Chưa có code ứng dụng trong workspace | Xác minh `capture_id`, frame nguồn, ảnh ghép và thứ tự phản hồi |

Các notebook là mã khởi đầu có điều kiện rõ ràng; việc tạo file chưa xác nhận rằng GPU/wheel, preprocessing hoặc pretrained đã chạy đúng trong Colab. Chỉ bắt đầu run dài sau smoke test và kiểm tra nhãn.

**17. Lỗi thường gặp và cách xử lý**

| Hiện tượng | Việc cần kiểm tra trước |
|---|---|
| Số nhận dạng khác ảnh vừa bấm | Frame cũ trong luồng chụp, crop/zoom khác frame, kết quả bất đồng bộ ghép sai capture ID |
| Số ảnh đọc rõ nhưng chưa đúng cân thực tế | Cân còn dao động tại thời điểm chụp; OCR chỉ đọc frame, chưa xác nhận ổn định |
| Accuracy rất cao ngay lập tức | Ảnh draft/final cùng event, duplicate, crop cùng cha có lọt sang val/test không |
| mAP cao nhưng số sai | Crop có giữ dấu chấm/chữ số cuối không; đánh giá reader trên predicted crop |
| Đọc được zoom nhưng sai ảnh gốc | Chênh lệch kích thước số; thiếu dữ liệu raw thật; train chỉ dựa vào zoom |
| Hay nhầm 0/8, 1/7, 5/6 | Ảnh LED thiếu đoạn/lóa, nhãn mơ hồ, lệch miền font; bổ sung ảnh thật tương ứng |
| Hay mất dấu chấm | Crop, resize quá nhỏ, threshold/morphology xóa chấm, decoding/head/dictionary |
| Trả số cho ảnh đen | Thiếu nhánh kiểm tra chất lượng, thiếu negative, ngưỡng chưa hiệu chỉnh |
| Out of memory | Giảm batch trước, sau đó cân nhắc kích thước/model; kiểm tra sampler batch |
| Paddle không thấy GPU | Runtime chưa có GPU, cài CPU wheel, driver/Python/wheel không tương thích |
| Load weights báo shape mismatch | Đổi dictionary/head/architecture hoặc dùng nhầm inference weights |
| Train chạy rất chậm | Ảnh đang đọc trực tiếp từ Drive, decode lặp lại, CPU/worker hoặc batch chưa phù hợp |
| Runtime mất, mất tiến độ | Checkpoint chỉ ở `/content`; chưa đồng bộ trong khi train |
| Resume không tiếp tục đúng | Sai checkpoint/state, đổi dataset/version/path hoặc tạo nhầm run mới |
| Chỉ camera 01 tốt | Dữ liệu/lấy mẫu mất cân bằng, ROI cứng, font/góc camera khác |
| Kết quả ONNX khác model gốc | Preprocessing, shape, dictionary/decoder, precision và engine |

Nút thắt hiện tại là bộ nhãn đúng và phép đánh giá không rò rỉ. Backup đủ để bắt đầu xây baseline có cơ sở; mức độ chính xác cuối cùng chỉ xác định được sau khi hoàn thành các bước trên và đánh giá trên dữ liệu chưa dùng để phát triển.
