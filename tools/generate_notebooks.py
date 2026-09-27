"""Generate the three Colab notebooks without storing notebook outputs."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": dedent(source).strip() + "\n"}


def write(name, cells):
    OUT.mkdir(parents=True, exist_ok=True)
    book = {"cells": cells, "metadata": {"colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}
    (OUT / name).write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


BOOTSTRAP = code("""
    from pathlib import Path
    import hashlib, json, os, shutil, subprocess, sys, zipfile
    from google.colab import drive

    # Điền URL sau khi tạo và push repo Git. Để trống sẽ dừng rõ ràng.
    REPO_URL = ''
    PROJECT_COMMIT = ''  # Commit SHA của phiên train; để trống chỉ khi thăm dò.
    GITHUB_SECRET_NAME = ''  # Repo private: đặt 'GITHUB_TOKEN' và lưu token trong Colab Secrets.
    VERSION = 'v001'
    DRIVE_ROOT = Path('/content/drive/MyDrive/scale-ocr')
    LOCAL_ROOT = Path('/content/scale-data')
    REPO_ROOT = Path('/content/scale-ocr-code')
    drive.mount('/content/drive')
    assert REPO_URL.startswith(('https://', 'git@')), 'Điền REPO_URL sau khi push Git'
    if not REPO_ROOT.exists():
        git_env = os.environ.copy()
        askpass = Path('/content/scale-ocr-askpass.sh')
        try:
            if GITHUB_SECRET_NAME:
                from google.colab import userdata
                assert REPO_URL.startswith('https://github.com/'), 'Secret này chỉ dùng với github.com'
                git_env['SCALE_GIT_TOKEN'] = userdata.get(GITHUB_SECRET_NAME)
                git_env['GIT_TERMINAL_PROMPT'] = '0'
                git_env['GIT_ASKPASS'] = str(askpass)
                askpass.write_text(chr(10).join([
                    '#!/bin/sh', 'case "$1" in',
                    '  *Username*) printf "%s\\n" "x-access-token" ;;',
                    '  *) printf "%s\\n" "$SCALE_GIT_TOKEN" ;;',
                    'esac', ''
                ]), encoding='utf-8')
                askpass.chmod(0o700)
            subprocess.run(['git', 'clone', REPO_URL, str(REPO_ROOT)],
                           env=git_env, check=True)
        finally:
            git_env.pop('SCALE_GIT_TOKEN', None)
            askpass.unlink(missing_ok=True)
    if PROJECT_COMMIT:
        subprocess.run(['git', '-C', str(REPO_ROOT), 'checkout', '--detach', PROJECT_COMMIT], check=True)
    PROJECT_COMMIT = subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(REPO_ROOT)], check=True)
    print('Code commit:', PROJECT_COMMIT)
    print('Disk GiB free:', round(shutil.disk_usage('/content').free / 2**30, 2))
    subprocess.run(['nvidia-smi'], check=False)
""")

COPY_DATASET = code("""
    # Dataset đã duyệt nằm trên Drive; vòng lặp train chỉ đọc từ /content.
    bundle = DRIVE_ROOT / 'datasets' / VERSION / 'dataset.zip'
    checksum_file = bundle.with_name(bundle.name + '.sha256')
    assert bundle.is_file() and checksum_file.is_file(), 'Upload dataset.zip và .sha256 sau khi gán/duyệt nhãn'
    expected = checksum_file.read_text(encoding='ascii').split()[0].lower()
    assert len(expected) == 64
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    local_bundle = LOCAL_ROOT / bundle.name
    def sha256_stream(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    if not local_bundle.exists() or sha256_stream(local_bundle) != expected:
        shutil.copyfile(bundle, local_bundle)
    actual = sha256_stream(local_bundle)
    assert actual == expected, f'Dataset SHA-256 mismatch: {actual}'
    prepared = LOCAL_ROOT / 'prepared' / VERSION
    if not (prepared / 'dataset_card.json').is_file():
        prepared.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(local_bundle) as archive:
            for item in archive.infolist():
                target = (prepared / item.filename).resolve()
                assert target.is_relative_to(prepared.resolve()), 'Unsafe ZIP path'
            archive.extractall(prepared)
    card = json.loads((prepared / 'dataset_card.json').read_text(encoding='utf-8'))
    assert card['counts'].get('recognizer_crops_train', 0) > 0, 'Dataset thiếu crop train'
    print('Dataset SHA-256:', actual)
    print('Counts:', card['counts'])
""")

write("01_prepare.ipynb", [
    md("""# 01 — Chuẩn bị backup trên Colab

    Chạy sau khi upload ZIP gốc và checksum vào `MyDrive/scale-ocr/raw/`, và push code lên Git. Notebook copy ZIP về `/content`, kiểm tra SHA-256, giải nén rồi tạo index/split/pilot. **Gán nhãn bằng công cụ local trong README**; notebook này chưa tạo ground truth.
    """),
    BOOTSTRAP,
    code("""
        source = DRIVE_ROOT / 'raw' / 'cloudinary_backup_2026-09-27.zip'
        sha_file = source.with_name(source.name + '.sha256')
        assert source.is_file() and sha_file.is_file(), 'Upload ZIP backup và .sha256 vào Drive/raw'
        expected = sha_file.read_text(encoding='ascii').split()[0].lower()
        assert len(expected) == 64
        assert shutil.disk_usage('/content').free > source.stat().st_size * 2, 'Thiếu dung lượng local'
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        local_zip = LOCAL_ROOT / source.name
        if not local_zip.exists():
            shutil.copyfile(source, local_zip)
        def sha256(path):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()
        assert sha256(local_zip) == expected, 'Backup SHA-256 mismatch; copy lại ZIP'
        raw_root = LOCAL_ROOT / 'cloudinary_backup_2026-09-27'
        if not (raw_root / 'manifest.json').is_file():
            with zipfile.ZipFile(local_zip) as archive:
                for item in archive.infolist():
                    target = (LOCAL_ROOT / item.filename).resolve()
                    assert target.is_relative_to(LOCAL_ROOT.resolve()), 'Unsafe ZIP path'
                archive.extractall(LOCAL_ROOT)
        assert (raw_root / 'manifest.json').is_file()
        print('Raw archive verified:', expected)
    """),
    code("""
        work = LOCAL_ROOT / 'work' / VERSION
        work.mkdir(parents=True, exist_ok=True)
        def run(*args):
            subprocess.run([sys.executable, '-m', 'scale_ocr', *map(str, args)], check=True)
        run('index', '--raw-root', raw_root, '--out', work / 'index.csv')
        run('split', '--index', work / 'index.csv', '--out', work / 'split.csv')
        run('pilot', '--index', work / 'index.csv', '--split', work / 'split.csv',
            '--out', work / 'pilot.csv', '--count', 400)
        print('Pilot:', work / 'pilot.csv')
        print('Để gán nhãn local: xem README. Lưu annotations.jsonl và split.csv trên Drive.')
    """),
])

write("02_train_detector.ipynb", [
    md("""# 02 — Train detector YOLO

    Chạy khi đã có dataset ZIP gán nhãn và cần so sánh detector với ROI theo camera. Mở runtime GPU. Điền URL Git, chạy các cell theo thứ tự. Checkpoint nằm trên Drive; ảnh train nằm ở `/content`. Với camera cố định, có thể bắt đầu reader trước rồi quay lại detector nếu ROI không ổn định.
    """),
    BOOTSTRAP,
    COPY_DATASET,
    code("""
        import torch, yaml
        from datetime import datetime, timezone
        assert torch.cuda.is_available(), 'Bật GPU trong Colab Runtime settings'
        print(torch.cuda.get_device_name(0))
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics'], check=True)
        from ultralytics import YOLO
        detector_root = prepared / 'detector'
        config = yaml.safe_load((detector_root / 'data.yaml').read_text(encoding='utf-8'))
        config['path'] = str(detector_root)
        local_yaml = detector_root / 'data.colab.yaml'
        local_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        assert (detector_root / 'images' / 'train').is_dir()
        RUN_ID = 'det-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_dir = DRIVE_ROOT / 'runs' / RUN_ID
        assert not run_dir.exists(), 'Run ID đã tồn tại'
        run_dir.mkdir(parents=True)
        (run_dir / 'run_manifest.json').write_text(json.dumps({
            'project_commit': PROJECT_COMMIT, 'dataset_sha256': actual,
            'dataset_version': VERSION, 'model': 'yolo26s.pt',
            'yaml': config, 'seed': 42, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Run:', RUN_ID)
    """),
    code("""
        # Thử 1 epoch bằng RUN_ID riêng trước run dài. Chỉ dùng ảnh train/val ở đây.
        model = YOLO('yolo26s.pt')
        model.train(data=str(local_yaml), imgsz=1024, epochs=100, batch=4,
            device=0, workers=2, optimizer='AdamW', lr0=0.001,
            patience=20, seed=42, fliplr=0, flipud=0, mosaic=0, mixup=0,
            degrees=5, translate=0.03, scale=0.1, hsv_h=0, hsv_s=0.2, hsv_v=0.2,
            project=str(DRIVE_ROOT / 'runs'), name=RUN_ID,
            save=True, save_period=5, exist_ok=True)
    """),
    md("""## Resume khi Colab ngắt

    Chạy lại bootstrap và copy **cùng version dataset** vào `/content`. Điền ID run cũ. Chỉ dùng `last.pt` để tiếp tục optimizer; `best.pt` dùng để chọn/suy luận.
    """),
    code("""
        RESUME_RUN_ID = ''  # Chỉ điền khi cần resume; không chạy cell này cho run mới.
        if RESUME_RUN_ID:
            old_run = DRIVE_ROOT / 'runs' / RESUME_RUN_ID
            old = json.loads((old_run / 'run_manifest.json').read_text(encoding='utf-8'))
            assert old['project_commit'] == PROJECT_COMMIT and old['dataset_sha256'] == actual
            last = old_run / 'weights' / 'last.pt'
            assert last.is_file(), 'Không có last.pt để resume'
            YOLO(str(last)).train(resume=True)
    """),
])

write("03_train_reader.ipynb", [
    md("""# 03 — Fine-tune PP-OCRv5 reader

    Mở runtime GPU mới. Dataset recognizer đã gồm crop từ ảnh camera gốc và vùng zoom; chỉ dùng crop có nhãn được duyệt. Notebook lấy source PaddleOCR và training weights chính thức, tạo config dựa trên upstream, lưu checkpoint trên Drive. Kiểm tra wheel PaddlePaddle hợp với CUDA/driver thực tế trước khi chạy cell cài đặt.
    """),
    BOOTSTRAP,
    COPY_DATASET,
    code("""
        # Ví dụ wheel CUDA 12.6 theo hướng dẫn Paddle; sửa theo runtime hiện được cấp.
        PADDLE_WHEEL_INDEX = 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'
        PADDLE_WHEEL = 'paddlepaddle-gpu==3.2.0'
        subprocess.run([sys.executable, '-m', 'pip', 'install', PADDLE_WHEEL,
                        '-i', PADDLE_WHEEL_INDEX], check=True)
        PADDLE_ROOT = Path('/content/PaddleOCR')
        PADDLE_REF = 'release/3.5'  # Sau smoke test, thay bằng commit SHA cố định.
        if not PADDLE_ROOT.exists():
            subprocess.run(['git', 'clone', '--depth', '1', '--branch', PADDLE_REF,
                'https://github.com/PaddlePaddle/PaddleOCR.git', str(PADDLE_ROOT)], check=True)
        paddle_commit = subprocess.check_output(
            ['git', '-C', str(PADDLE_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                        str(PADDLE_ROOT / 'requirements.txt')], check=True)
        import paddle, yaml
        assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        paddle.set_device('gpu:0')
        paddle.utils.run_check()
        print('PaddleOCR commit:', paddle_commit)
    """),
    code("""
        from urllib.request import urlretrieve
        from datetime import datetime, timezone
        pretrained = LOCAL_ROOT / 'pretrained' / 'PP-OCRv5_server_rec_pretrained.pdparams'
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        if not pretrained.exists():
            urlretrieve('https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/'
                        'PP-OCRv5_server_rec_pretrained.pdparams', pretrained)
        def sha256(path):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()
        rec_root = prepared / 'recognizer'
        for split in ('train', 'val'):
            assert (rec_root / (split + '.txt')).stat().st_size > 0, f'Thiếu nhãn {split}'
        base_config = PADDLE_ROOT / 'configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml'
        cfg = yaml.safe_load(base_config.read_text(encoding='utf-8'))
        REC_RUN_ID = 'rec-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        rec_run = DRIVE_ROOT / 'runs' / REC_RUN_ID
        rec_run.mkdir(parents=True, exist_ok=False)
        cfg['Global'].update({'use_gpu': True, 'distributed': False, 'epoch_num': 50,
            'pretrained_model': str(pretrained), 'checkpoints': None,
            'save_model_dir': str(rec_run / 'checkpoints'), 'save_epoch_step': 5,
            'eval_batch_step': [0, 100]})
        cfg['Optimizer']['lr']['learning_rate'] = 0.0001
        for section, label in [('Train', 'train.txt'), ('Eval', 'val.txt')]:
            cfg[section]['dataset']['data_dir'] = str(rec_root)
            cfg[section]['dataset']['label_file_list'] = [str(rec_root / label)]
            cfg[section]['loader']['num_workers'] = 2
            cfg[section]['loader']['batch_size_per_card'] = 16
        cfg['Train']['sampler']['first_bs'] = 16
        cfg['Train']['sampler']['fix_bs'] = True
        cfg['Train']['sampler']['scales'] = [[320, 48]]
        cfg['Train']['dataset']['transforms'] = [t for t in cfg['Train']['dataset']['transforms']
            if 'RecAug' not in t]
        rec_config = rec_run / 'recognizer.yaml'
        rec_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        (rec_run / 'run_manifest.json').write_text(json.dumps({
            'project_commit': PROJECT_COMMIT, 'dataset_sha256': actual,
            'dataset_version': VERSION, 'paddleocr_commit': paddle_commit,
            'pretrained_sha256': sha256(pretrained), 'paddle': paddle.__version__,
            'config_sha256': sha256(rec_config), 'seed': 42,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Run:', REC_RUN_ID)
    """),
    code("""
        # Trước run chính, thử 32–64 crop và kiểm tra log pretrained/learning.
        subprocess.run([sys.executable, 'tools/train.py', '-c', str(rec_config)],
                       cwd=PADDLE_ROOT, check=True)
    """),
    md("""## Resume khi Colab ngắt

    Chạy lại bootstrap, copy cùng dataset, cài PaddleOCR rồi chọn prefix checkpoint cũ có cả `.pdparams` và `.pdopt`. Giữ nguyên config cũ; `Global.checkpoints` phục hồi trạng thái train.
    """),
    code("""
        RESUME_RUN_ID = ''
        CHECKPOINT_NAME = ''  # Ví dụ: latest hoặc epoch_10, theo tên thực tế trong Drive.
        if RESUME_RUN_ID:
            old_run = DRIVE_ROOT / 'runs' / RESUME_RUN_ID
            old = json.loads((old_run / 'run_manifest.json').read_text(encoding='utf-8'))
            assert old['project_commit'] == PROJECT_COMMIT and old['dataset_sha256'] == actual
            old_cfg = old_run / 'recognizer.yaml'
            assert sha256(old_cfg) == old['config_sha256']
            prefix = old_run / 'checkpoints' / CHECKPOINT_NAME
            assert prefix.with_suffix('.pdparams').is_file()
            assert prefix.with_suffix('.pdopt').is_file()
            subprocess.run([sys.executable, 'tools/train.py', '-c', str(old_cfg),
                '-o', 'Global.checkpoints=' + str(prefix)], cwd=PADDLE_ROOT, check=True)
    """),
])
