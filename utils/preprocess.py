import pandas as pd
import numpy as np


_file_name = './static/data/NPA_TMA'
_files = [pd.read_csv(f'{_file_name}{i}.csv') for i in range(1, 4)]

_files[0].drop('死亡受傷人數', axis=1, inplace=True)
_files[1]['受傷人數'] = _files[1]['死亡受傷人數'].apply(lambda s: int(s[-1]))
_files[1].drop('死亡受傷人數', axis=1, inplace=True)

_files[0].drop(['經度', '緯度'], axis=1, inplace=True)
_files[1].drop(['經度', '緯度'], axis=1, inplace=True)

_files[1]['縣市'] = _files[1]['發生地點'].str[:3]
_files[2]['縣市'] = _files[2]['發生地點'].str[:3]

for file in _files:
    # 添加事故主要車種
    file['事故主要車種'] = '機車/腳踏車'
    file.loc[
        file['車種'].str.contains(
            '公車|客運|貨車|曳引車|火車|拖車|拼裝車|遊覽車|大客車', na=False), '事故主要車種'] = '大型車'
    file.loc[file['車種'].str.contains('小客車|計程車|救護車|其他車', na=False), '事故主要車種'] = '小型車'
    
    # 添加城市規模
    file['城市規模'] = '一般縣市'
    file.loc[
        file['縣市'].str.contains(
            '臺北市|新北市|桃園市|臺中市|臺南市|高雄市', na=False), '城市規模'] = '直轄市'
    
    # 添加事故發生時段
    periods = file['發生時間'].apply(lambda s: int(s[11:13]))
    file['發生時段'] = '晚間'
    file.loc[periods < 7, '發生時段'] = '夜間'
    file.loc[periods < 19, '發生時段'] = '日間'
    
    # 添加事故發生月份
    file['發生月份'] = file['發生時間'].apply(lambda s: s[4:7].replace('^0', ''))


TMA1, TMA2, TMA3 = _files
__all__ = ['TMA1', 'TMA2', 'TMA3']