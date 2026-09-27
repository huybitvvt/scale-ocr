# Scale OCR

Code chuẩn bị dữ liệu và huấn luyện đọc số cân từ ảnh camera. Kế hoạch đầy đủ ở [TRAINING_PLAN.md](docs/TRAINING_PLAN.md) và hướng dẫn Colab ở [COLAB_GUIDE.md](docs/COLAB_GUIDE.md).

Repo này nằm trong `E:\backup-tramcan\scale-ocr`; backup gốc ở thư mục cha. Git chỉ chứa code/notebook/tài liệu. Dữ liệu ảnh, nhãn, split, checkpoint nằm ngoài Git hoặc trong `data/` đã ignore. Hiện chưa có Git remote, chưa có nhãn đã duyệt và chưa có mô hình đã train.

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

Trước khi train, reviewer cần duyệt nhãn. Tập val/test yêu cầu `reviewer` khác `annotator`. Chỉ nhãn `reviewed` được xuất. Sau khi đủ nhãn:

```powershell
python -m scale_ocr export --raw-root ..\cloudinary_backup_2026-09-27 --index data\v001\index.csv --split data\v001\split.csv --annotations data\v001\annotations.jsonl --out data\v001\prepared
python -m scale_ocr package --source data\v001\prepared --out data\v001\dataset.zip
```

`prepared/detector` có ảnh/nhãn YOLO và `data.yaml`; `prepared/recognizer` có crop số và các file `train.txt`, `val.txt`, `test.txt`. `dataset_card.json` ghi hash nguồn, nhãn và split. Khi sửa nhãn, tạo v002 trong thư mục mới thay vì ghi đè v001 đã dùng để train.

Đã chạy `index`, `split`, `pilot` trên backup hiện tại: 9.260 ảnh, 8.397 nội dung SHA-256 khác nhau; train 5.387, val 1.954, test 1.715, exclude 204; pilot 400 ảnh. File CSV sinh ra ở `data/v001/` trên máy hiện tại. Đây là split sơ bộ theo đường dẫn ngày, event và hash; cần rà soát cùng nhãn trước khi chốt. Các số này mô tả **ảnh**, chưa phải 9.260 nhãn số cân.

## Git → Drive → Colab

1. Tạo Git remote **private** trên dịch vụ Git anh chọn. Repo local đã sẵn sàng; thêm remote và push sau khi có URL:

   ```powershell
   git remote add origin <URL_REPO_PRIVATE>
   git push -u origin main
   ```

2. Upload `cloudinary_backup_2026-09-27.zip` và `.sha256` lên `MyDrive/scale-ocr/raw/`. Sau khi export, upload `data/v001/dataset.zip` và checksum sang `MyDrive/scale-ocr/datasets/v001/`. Giữ bản nhãn và split riêng trong Drive để có thể chỉnh và kiểm toán.
3. Mở [01_prepare.ipynb](notebooks/01_prepare.ipynb) trên Colab để kiểm tra backup và tạo index/pilot nếu cần. Sau khi đủ nhãn đã duyệt và upload ZIP đã export, chạy [03_train_reader.ipynb](notebooks/03_train_reader.ipynb); dùng [02_train_detector.ipynb](notebooks/02_train_detector.ipynb) khi ROI theo camera không ổn định hoặc để so sánh. Các notebook copy archive về `/content`, xác minh SHA-256, giải nén và train trên đĩa local. Checkpoint ghi vào Drive. Điền `REPO_URL` và `PROJECT_COMMIT` đã push. Với GitHub private, lưu token chỉ đọc repo trong Colab Secrets và đặt `GITHUB_SECRET_NAME='GITHUB_TOKEN'`; không ghi token vào notebook.

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

Giới hạn hiện tại: metadata Cloudinary không chứa giá trị cân đã được người duyệt. Không thể train hoặc đo accuracy có ý nghĩa trước khi hoàn thành bộ nhãn.
