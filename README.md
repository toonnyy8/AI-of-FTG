# AI-of-FTG
以tensorflow.js實作基於深度學習的FTG AI

## 安裝
```
git clone https://github.com/toonnyy8/AI-of-FTG.git

cd AI-of-FTG

npm i

npm run dev
```

然後在瀏覽器開啟`http://127.0.0.1:8080`

## 建置
```
npm run build
```
開啟./build/index.html

## 線上測試

### 版本1(channel通訊版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn/index.html)  
[agent](https://toonnyy8.github.io/AI-of-FTG/build/dddqn/agent.html)

### 版本1(web worker版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn_webworker/index.html)

### 版本2(channel通訊版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn2/index.html)  
[agent](https://toonnyy8.github.io/AI-of-FTG/build/dddqn2/agent.html)

### 版本3(channel通訊版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn3/index.html)  
[agent](https://toonnyy8.github.io/AI-of-FTG/build/dddqn3/agent.html)

### 版本5(channel通訊版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn5/index.html)  
[agent](https://toonnyy8.github.io/AI-of-FTG/build/dddqn5/agent.html)
```
使用了softmax與logSoftmax取出相對優劣勢
```

### 版本6(channel通訊版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn6/index.html)  
[agent](https://toonnyy8.github.io/AI-of-FTG/build/dddqn6/agent.html)
```
使用softmax並乘上正負來取出相對優劣勢，
取出後與原Q值相加
```

### 版本7(channel通訊版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn7/index.html)  
[agent](https://toonnyy8.github.io/AI-of-FTG/build/dddqn7/agent.html)
```
使用softmax並乘上正負來取出相對優劣勢，
相對優劣勢減去mean後與原Q值相加
```

### 版本8(channel通訊版)
>[environment](https://toonnyy8.github.io/AI-of-FTG/build/dddqn8/index.html)  
[agent](https://toonnyy8.github.io/AI-of-FTG/build/dddqn8/agent.html)
```
使用max-min正規化取出相對優劣勢
```