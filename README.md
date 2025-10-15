# 📈 Streamlit 線性迴歸互動應用

這是一個使用 **Streamlit** 製作的互動式線性回歸展示專案。  
可自由調整參數 `a`, `b`, `noise`, `n_points`，即時觀察模型擬合效果。

---

## 🚀 本地執行方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 部署到 Streamlit Cloud

1. 登入 [Streamlit Cloud](https://streamlit.io/cloud)
2. 新建應用，連結到此專案的 GitHub repository
3. 在執行指令中輸入：
   ```
   streamlit run app.py
   ```
4. 點擊 **Deploy**

完成後即可取得網址，例如：
```
https://你的帳號.streamlit.app
```

---

## 📂 專案結構
```
streamlit_regression_project/
├── app.py
├── regression.py
├── requirements.txt
└── .streamlit/config.toml
```
