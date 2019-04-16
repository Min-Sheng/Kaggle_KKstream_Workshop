# Kaggle_KKstream_Workshop
Kaggle Competition- KKstream Deep Learning Workshop:
<https://www.kaggle.com/c/kkstream-deep-learning-workshop/>

## Goal
由 KKTV 提供的去識別化用戶看劇的歷史資料，來預測之後指定某段時間，用戶是否會看劇。

若能準確掌握用戶看劇習慣，便能適時地推播用戶可能有興趣的內容與廣告，如此一來應可提高用戶對產品的喜愛與依賴；而避免在用戶不看劇的時段推播，也能減少打擾用戶的機會。

## Feature Engineering
除了主辦單位提供的 preprocessed data 之外，我加入了從 raw data 中另外蒐集而來的 5 種特徵:

1. Total duration in the slot: 在每個 time slot 中，用戶總共的播劇時間長度。
2. Average episode rate in the slot: 在每個 time slot 中，將每部劇所屬的集數總和除以每部劇的影集集數總和。
3. Platform kinds in the slot: 在每個 time slot 中，用戶所使用的播放平台。
4. Action types in the slot: 在每個 time slot 中，導致播放器停止播放的原因。
5. Trailer flags in the slot: 在每個 time slot 中，每部用戶觀看的影片使否為預告片。

我認為這幾項特徵能進一步描述個用戶的看劇情形，對預測效果應該有所助益。

## Model Architecture
最終採用的架構如下圖：

<img width="80%" src="https://github.com/Min-Sheng/Kaggle_KKstream_Workshop/raw/master/kkstream_model.png"/>

先利用 GRU 抽取時序資訊，接著將萃取出的 latant variables 餵進由 1D Convolution 組成的 ResNet 中，進一步將時序資訊濃縮，最後過一層 global average pooling 與 fully connected layer ，並通過 Sigmoid function 產生 binary sequence output 。

## Loss Function
我將此問題視為一個 multi-label binary classification task ，故使用 Binary Cross-Entropy loss (BCE) 來訓練我的 model 。

## Training detail
我使用 Tesla K80 來訓練，將 training 與 validation 比例為 4:1 ，並採用 Adam optimizer ， batch size 為 128 ， initial learning rate 設為 0.001 ，並每 10 epoch 降 0.1 倍。

此外，我使用 batch normalization 加速收斂與避免 gradient vanishing/exploding problem ，並改用 ELU activation function 取代 ReLU 。

## Experiment
1. 一開始以 DNN (Dense Neural Network) 作為 baseline ，在 testing data 的 AUROC 表現為: 0.87084 。
2. 改用 1D Convolution ResNet 的架構，在 testing data 的表現提升至: 0.88487 。
3. 改用 ELU 取代 ReLU ， AUROC 略為提高至: 0.88522。
4. 加入 1 層 RNN (採用Bidirectional GRU)，並使用更多從 raw data 萃取的 features ， AUROC 升高至 0.88742 。
5. 增加為 2 層 GRU，得到最高 performance 為 0.88810 。

## Result

## Conclusion
1. 考慮時序資訊的因素很重要。
2. 除了 time slot 之外，加入用戶的其他看劇資訊對 performance 有很大幫助。

## Reference
我使用 PyTorch Template: <https://github.com/victoresque/pytorch-template> ，快速部屬我的 Deep Learning 系統。
