This algorithm is based on LOAM
https://github.com/wh200720041/floam

This version is to check plane & line eigen

檢查方法
1.將用來估計線特徵的5個點使用SVD來得到最小平方法的向量

再跟線特徵的curr_point算距離，最後將這結果和PCA的距離比較

2. 將平面點特徵取特徵矩陣，然後再將數值最小的維度當作法向量

，最後將原本QR分解出的法向量正規化跟特徵法向量做比較。
