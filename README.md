# 環境
`ubuntu 18.04`

**安裝virtualenv**
```
pip install virtualenv
cd  進你要的資料夾
python3 -m venv venv
```
**使用虛擬環境**
```
source ./venv/bin/activate
```
# python 套件
```
pip install flwr
pip install scikit-learn
```
# Run Federated Learning 
start server
```
python server.py
```
Start client 1 
```
python client.py
```
start client2 in second terminal
```
python client.py 
```