# Scale OCR

Code chuẩn bị dữ liệu và huấn luyện đọc số cân từ ảnh camera. Kế hoạch đầy đủ ở [TRAINING_PLAN.md](docs/TRAINING_PLAN.md) và hướng dẫn Colab ở [COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

Repo local nằm trong `E:\backup-tramcan\scale-ocr`; remote public: `https://github.com/huybitvvt/scale-ocr`. Backup gốc ở thư mục cha. Git chỉ chứa code/notebook/tài liệu. Dữ liệu ảnh, nhãn, split, checkpoint nằm ngoài Git hoặc trong `data/` đã ignore. Đã có 400 nhãn pilot được duyệt và dataset v001 đã đóng gói; chưa có mô hình đã train hoặc kết quả accuracy.

Nếu Colab báo lỗi 400/403 khi gắn Google Drive, mở [04_train_reader_upload.ipynb](notebooks/04_train_reader_upload.ipynb). Notebook này không gọi `drive.mount`. Ở cell upload, chọn đồng thời bốn file trong `E:\backup-tramcan\colab_upload_v001`. Sau smoke test mới chạy cell train chính; tải ZIP checkpoint về máy trước khi ngắt runtime. Xem hướng dẫn cụ thể ở [COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

## Chạy giai đoạn chuẩn bị trên Windows

Mở PowerShell trong thư mục `scale-ocr`:

```powershell
python -m pip install -e .
python -m scale_ocr index --raw-root ..\cloudinary_backup_2026-09-27 --out data\v001\index.csv
python -m scale_ocr split --index data\v001\index.csv --out data\v001\split.csv
python -m scale_ocr pilot --index data\v001\index.csv --split data\v001\split.csv --out data\v001\pilot.csv --count 400
python -m scale_ocr annotate --pilot data\v001\pilot.csv --raw-root ..\cloudinary_backup_2026-09-27 --out data\v001\annotations.jsonl
```

Mở `http://127.0.0.1:8765/`. Kéo chuột để vẽ khung **dãy số cân**, nhập nguyên văn số hiển thị, rồi thêm vùng. Với ảnh ghép, gán nhãn vùng ở cảnh gốc và vùng zoom nếu cả hai đọc được. Nếu hai vùng hiển thị hai số khác nhau, công cụ không nhận nhãn `readable`: cần kiểm tra ảnh/luồng ghép. Ảnh đen, mờ, bị che hoặc không thấy màn hình cần chọn đúng trạng thái; không đoán số. Nhãn được lưu liên tục vào `data\v001\annotations.jsonl` và có thể mở lại để sửa. Công cụ chỉ lắng nghe máy cục bộ.

400 ảnh pilot đã được người dùng duyệt tại `data/v001/annotations.jsonl`: 389 đọc được, 6 không thấy số, 3 không đọc được, 1 chỉ hiện một phần và 1 không phải số cân. Bản nhãn cố định nằm trong `data/v001/release/`, có SHA-256 cho từng file. Nếu sửa nhãn sau thời điểm này, tạo dataset v002 thay vì ghi đè gói v001. Giao diện gán nhãn vẫn cho sửa số trực tiếp, xem vùng cắt phóng to và duyệt từng ảnh.

Để tạo lại nhãn nháp trên một bản sao dữ liệu, cài thêm dependencies `python -m pip install -e ".[prelabel]"`, rồi chạy:

```powershell
python tools/prelabel_pilot.py --pilot data/v001/pilot.csv --raw-root ../cloudinary_backup_2026-09-27 --out-dir data/v001/assist
# Rà contact sheets và ghi audit_decisions.json, manual_boxes.json trong data/v001/assist/
python tools/import_pilot_drafts.py --pilot data/v001/pilot.csv --candidates data/v001/assist/candidates.json --audit data/v001/assist/audit_decisions.json --manual-boxes data/v001/assist/manual_boxes.json --out data/v001/annotations.jsonl
```

Lệnh import giữ lại mọi nhãn đã có; dừng máy chủ gán nhãn trước khi chạy để tránh bộ nhớ máy chủ ghi đè file. Các file JSON và ảnh contact sheet nằm trong `data/` đã ignore, không đưa lên repo public.

Dataset v001 đã được xuất từ bản nhãn cố định. Để tạo lại trong một thư mục trống:

```powershell
python -m scale_ocr export --raw-root ..\cloudinary_backup_2026-09-27 --index data\v001\release\index.csv --split data\v001\release\split.csv --annotations data\v001\release\annotations.jsonl --out data\v001\prepared_release
python -m scale_ocr package --source data\v001\prepared_release --out data\v001\dataset.zip
```

`prepared_release/detector` có ảnh/nhãn YOLO và `data.yaml`; `prepared_release/recognizer` có crop số và các file `train.txt`, `val.txt`, `test.txt`. `evaluation_manifest.csv` giữ đủ 400 ảnh đã duyệt; 5 ảnh không đủ điều kiện tạo crop train được giữ trong `evaluation/other/`. `dataset_card.json` ghi hash nguồn, nhãn và split.

Đã chạy `index`, `split`, `pilot` trên backup hiện tại: 9.260 ảnh, 8.397 nội dung SHA-256 khác nhau; train 5.387, val 1.954, test 1.715, exclude 204; pilot 400 ảnh. File CSV sinh ra ở `data/v001/` trên máy hiện tại. Đây là split sơ bộ theo đường dẫn ngày, event và hash; cần rà soát cùng nhãn trước khi chốt. Các số này mô tả **ảnh**, chưa phải 9.260 nhãn số cân.

## Git → Drive → Colab

1. Repo GitHub public ở `https://github.com/huybitvvt/scale-ocr`. Khi sửa code, commit và push từ thư mục này. Không đưa ảnh, nhãn hoặc checkpoint lên Git.
2. ZIP gốc và checksum đã được upload vào `MyDrive/tram-can/` theo ảnh Drive anh gửi. Upload `data/v001/dataset.zip` và `dataset.zip.sha256` sang `MyDrive/tram-can/datasets/v001/`. Upload `labels_reviewed_v001.zip` và checksum sang `MyDrive/tram-can/labels/v001/` để lưu bằng chứng phiên bản. Notebook sẽ xác minh SHA-256 của dataset sau khi mount Drive bằng đúng tài khoản sở hữu thư mục này.
3. Mở [03_train_reader.ipynb](notebooks/03_train_reader.ipynb) trên Colab để train reader baseline; dùng [02_train_detector.ipynb](notebooks/02_train_detector.ipynb) khi ROI theo camera không ổn định hoặc để so sánh. [01_prepare.ipynb](notebooks/01_prepare.ipynb) chỉ cần nếu tạo lại index/pilot từ backup. Các notebook copy archive về `/content`, xác minh SHA-256, giải nén và train trên đĩa local. Checkpoint ghi vào Drive. Repo public clone trực tiếp, không cần GitHub token. Trước run chính, điền `PROJECT_COMMIT` đã push để khóa code.

ZIP backup gốc khoảng 3,61 GB. Kiểm tra dung lượng trống trong Colab trước khi copy và giải nén. Không đọc hàng nghìn ảnh trực tiếp từ Drive mount trong vòng lặp train.

## Đánh giá

Sau khi model tạo `predictions.csv` với `sample_id,gateway,readability,status,pred_text,gt_text`, chạy:

```powershell
python -m scale_ocr evaluate --predictions data\v001\predictions.csv --out data\v001\metrics.json
```

Metric chính: đúng toàn bộ chuỗi số trên ảnh có thể đọc, độ đúng trong số ảnh tự động chấp nhận và tỷ lệ từ chối. Ảnh khó hoặc không có màn hình phải xuất hiện trong báo cáo, không được bỏ khỏi mẫu số.

## Kiểm tra local

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

Giới hạn hiện tại: 137 crop train là tập pilot nhỏ, gateway-03 chỉ có 5 ảnh train và không có val/test. Chưa chạy Colab hoặc đo accuracy. Cần đánh giá baseline trước khi quyết định số ảnh cần gán thêm.
