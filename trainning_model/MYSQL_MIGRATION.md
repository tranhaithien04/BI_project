# Huong Dan Chay MySQL-Only

## 1) Cai dependency

```bash
d:/BIAPP/.venv/Scripts/python.exe -m pip install -r BI/trainning_model/requirements.txt
```

## 2) Cau hinh MySQL

Project da duoc setup theo schema va tai khoan:

- Schema: `jobmatch`
- User: `root`
- Password: `root`

Dat bien moi truong trong PowerShell:

```bash
$env:APP_DATABASE_URL="mysql+pymysql://root:root@127.0.0.1:3306/jobmatch?charset=utf8mb4"
$env:APP_SESSION_SECRET="replace-with-strong-secret"
```

Hoac copy file `.env.example` va ap dung theo cach cua ban.

## 3) Chay app

```bash
d:/BIAPP/.venv/Scripts/python.exe BI/trainning_model/main.py
```

Hoac uvicorn:

```bash
cd BI/trainning_model
..\..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## 4) Kiem tra nhanh

Mo endpoint:

- `http://127.0.0.1:8000/health`

Neu ket noi MySQL thanh cong, app se tu dong tao database/tables neu chua ton tai.

## Luu y

- He thong da bo SQLite khoi luong runtime.
- Neu `APP_DATABASE_URL` khong phai MySQL, app se bao loi ngay luc startup.
- Neu artifacts AI chua san sang, app van khoi dong cho auth/profile/recruitment web flow.
