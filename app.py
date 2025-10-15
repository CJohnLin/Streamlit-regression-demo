import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="線性迴歸互動示範", page_icon="📊", layout="centered")

st.title("📈 線性迴歸互動應用（Linear Regression Demo）")
st.markdown("這個應用示範了 **線性迴歸模型** 如何根據不同參數變化進行擬合。")

# Sidebar 參數設定
st.sidebar.header("參數設定")
a = st.sidebar.slider("斜率 a", -10.0, 10.0, 2.0)
b = st.sidebar.slider("截距 b", -10.0, 10.0, 1.0)
noise = st.sidebar.slider("噪音程度", 0.0, 10.0, 1.0)
n_points = st.sidebar.slider("資料筆數", 10, 200, 50)

# 產生資料
np.random.seed(42)
X = np.random.rand(n_points, 1) * 10
y = a * X + b + np.random.randn(n_points, 1) * noise

# 分割訓練與測試
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 評估指標
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 模型評估結果")
st.write(f"**R²:** {r2:.3f}")
st.write(f"**MSE:** {mse:.3f}")
st.write(f"**MAE:** {mae:.3f}")

# 圖表
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='資料點')
ax.plot(X, a*X + b, color='green', linewidth=2, label='真實線')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='預測線')
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("線性回歸擬合圖")
st.pyplot(fig)

st.caption("作者：你的名字 | 使用 Streamlit 製作")
