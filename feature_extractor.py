import numpy as np
import pandas as pd
import datetime as dt
import glob
from tqdm import tqdm

# Training data:
# data-001.csv ~  data-045.csv : 訓練組使用者在 2017/01/01-00:00:00 
#                                           ~ 2017/08/21-01:00:00 間的使用記錄。
# label-001.csv ~ label-045.csv : 訓練組使用者在 2017/08/14-01:00:00 
#                                           ~ 2017/08/21-01:00:00 間的使用記錄的標籤。
# Testing data:
# data-046.csv ~  data-075.csv : 測試組使用者在 2017/01/01-00:00:00 
#                                           ~ 2017/08/14-01:00:00 間的使用記錄。

# Label
# time_slot_0 : 使用者在 2017/08/14-01:00:00 ~ 2017/08/14-09:00:00 間有無活動（機率）。
# time_slot_1 : 使用者在 2017/08/14-09:00:00 ~ 2017/08/14-17:00:00 間有無活動（機率）。
# time_slot_2 : 使用者在 2017/08/14-17:00:00 ~ 2017/08/14-21:00:00 間有無活動（機率）。
# time_slot_3 : 使用者在 2017/08/14-21:00:00 ~ 2017/08/15-01:00:00 間有無活動（機率）。
# time_slot_4 ~ time_slot_27 : 以此類推。

data_files = glob.iglob(r'public/data-*.csv')
data_files = list(sorted(data_files))
label_files = glob.iglob(r'public/label-*.csv')
label_files = list(sorted(label_files))

# Platform
def what_platform(x):
    
    if x == "Web":
        platform = 0
    elif x == "Android":
        platform = 1
    elif x == "iOS":
        platform = 2
    return platform

# Action (action_trigger, episode_number, series_total_episodes_count, is_trailer)
def which_action_type(x):
    if 'network error': # 網路出錯
        action = 0
    elif 'pause': # 用戶暫停播放
        action = 1
    elif 'cast': # 用戶啟動 chromecast 或 airplay
        action = 2
    elif 'api error': # api 禁止播放
        action = 3
    elif 'limit single device playing': # 同帳號不能跨平台同步播放影片，被另一個平台搶走播放權的時候，停止播放
        action = 4
    elif 'leave': # 用戶主動離開播放器
        action = 5
    elif 'next episode': # 用戶點擊「下一集」的按鈕
        action = 6
    elif 'interrupt': # 被系統中斷
        action = 7
    elif 'player error': # 播放器出現錯誤
        action = 8
    elif 'limit playzone countdown': # playzone 每日有時間限制（免費時限），播放中遭遇倒數終止，而停止播放
        action = 9
    elif 'seek': # 用戶 seek video
        action = 10
    elif 'video ended': # 影片播到底，最後一秒
        action = 11
    elif 'program stopped or enlarged-reduced': # 節目即將播畢
        action = 12
    elif 'program stopped': # 節目即將播畢
        action = 13
    elif 'error': # 未定義，就是個錯誤
        action = 14
    return action

# Concaatenate all training labels
new_label = []
for file in label_files:
    label = pd.read_csv(file)
    new_label.append(label)
train_label = pd.concat(new_label, ignore_index=True)
np.save("train_labels.npy", train_label)

# Extract training features
new_feature = []
#2017/01/01-00:00:00 ~ 2017/08/21-01:00:00 間的使用記錄。
start_date = dt.date(2017, 1, 1)
end_date = dt.date(2017, 8, 21)
length = (end_date - start_date).days * 4

for file in data_files[:45]:
    data = pd.read_csv(file)
    # Time
    data['event_time'] = pd.to_datetime(data['event_time'])
    data['date'] = pd.to_datetime(data['event_time']).dt.date
    data['time'] = pd.to_datetime(data['event_time']).dt.time
    data.index=pd.to_datetime(data['event_time'])

    for i in tqdm(np.unique(data['user_id'])):

        feature = np.zeros((1+1+3+15+2, length), dtype=np.float32)

        slot1_df = data[data['user_id']==i].between_time("{}:0:0".format(1),"{}:0:0".format(9))
        slot2_df = data[data['user_id']==i].between_time("{}:0:0".format(9),"{}:0:0".format(17))
        slot3_df = data[data['user_id']==i].between_time("{}:0:0".format(17),"{}:0:0".format(21))
        slot4_1_df = data[data['user_id']==i].between_time("{}:0:0".format(21),"{}:0:0".format(0))
        slot4_2_df = data[data['user_id']==i].between_time("{}:0:0".format(0),"{}:0:0".format(1))
        
        for j, slot_df in enumerate([slot1_df, slot2_df, slot3_df, slot4_1_df, slot4_2_df]):
            date_group = slot_df.groupby(['date'])
            date_group_sum = date_group.sum()
            dates = date_group_sum.index
            for date in dates:
                if j == 4:
                    slot_id = ((date - start_date).days-1) * 4 + (j-1)
                else:
                    slot_id = (date - start_date).days * 4 + j
                
                total_count = date_group.size().loc[date]
                total_duration = date_group_sum['played_duration'].loc[date]
                total_episode_number = date_group_sum['episode_number'].loc[date]
                total_episodes_count = date_group_sum['series_total_episodes_count'].loc[date]
                
                # feature 0: total duration in the slot
                feature[0, slot_id] = total_duration
                # feature 1: average episode rate in the slot
                feature[1, slot_id] = total_episode_number/total_episodes_count
                
                # feature 2~4: platform kinds in the slot
                for ptf, cnt in slot_df.groupby(['date', 'platform']).size().loc[date].iteritems():
                    feature[what_platform(ptf)+2, slot_id] = cnt/total_count
                    
                # feature 5~19: action types in the slot
                for act, cnt in slot_df.groupby(['date', 'action_trigger']).size().loc[date].iteritems():
                    feature[which_action_type(act)+5, slot_id] = cnt/total_count
                
                # feature 20~21: trailer flags in the slot
                for tfg, cnt in slot_df.groupby(['date', 'is_trailer']).size().loc[date].iteritems():
                    feature[int(tfg)+20, slot_id] = cnt/total_count
        
        new_feature.append(feature)
        tqdm.write("extract user #{}".format(i))


train_feature = np.stack(new_feature)
np.save("train_features.npy", train_feature)


# Extract testing features
new_feature = []
# 2017/01/01-00:00:00 ~ 2017/08/14-01:00:00 間的使用記錄。
start_date = dt.date(2017, 1, 1)
end_date = dt.date(2017, 8, 14)
length = (end_date - start_date).days * 4

for file in data_files[45:]:
    data = pd.read_csv(file)
    # Time
    data['event_time'] = pd.to_datetime(data['event_time'])
    data['date'] = pd.to_datetime(data['event_time']).dt.date
    data['time'] = pd.to_datetime(data['event_time']).dt.time
    data.index=pd.to_datetime(data['event_time'])

    for i in tqdm(np.unique(data['user_id'])):

        feature = np.zeros((1+1+3+15+2, length), dtype=np.float32)

        slot1_df = data[data['user_id']==i].between_time("{}:0:0".format(1),"{}:0:0".format(9))
        slot2_df = data[data['user_id']==i].between_time("{}:0:0".format(9),"{}:0:0".format(17))
        slot3_df = data[data['user_id']==i].between_time("{}:0:0".format(17),"{}:0:0".format(21))
        slot4_1_df = data[data['user_id']==i].between_time("{}:0:0".format(21),"{}:0:0".format(0))
        slot4_2_df = data[data['user_id']==i].between_time("{}:0:0".format(0),"{}:0:0".format(1))
        
        for j, slot_df in enumerate([slot1_df, slot2_df, slot3_df, slot4_1_df, slot4_2_df]):
            date_group = slot_df.groupby(['date'])
            date_group_sum = date_group.sum()
            dates = date_group_sum.index
            for date in dates:
                if j == 4:
                    slot_id = ((date - start_date).days-1) * 4 + (j-1)
                else:
                    slot_id = (date - start_date).days * 4 + j
                
                total_count = date_group.size().loc[date]
                total_duration = date_group_sum['played_duration'].loc[date]
                total_episode_number = date_group_sum['episode_number'].loc[date]
                total_episodes_count = date_group_sum['series_total_episodes_count'].loc[date]
                
                # feature 0: total duration in the slot
                feature[0, slot_id] = total_duration
                # feature 1: average episode rate in the slot
                feature[1, slot_id] = total_episode_number/total_episodes_count
                
                # feature 2~4: platform kinds in the slot
                for ptf, cnt in slot_df.groupby(['date', 'platform']).size().loc[date].iteritems():
                    feature[what_platform(ptf)+2, slot_id] = cnt/total_count
                    
                # feature 5~19: action types in the slot
                for act, cnt in slot_df.groupby(['date', 'action_trigger']).size().loc[date].iteritems():
                    feature[which_action_type(act)+5, slot_id] = cnt/total_count
                
                # feature 20~21: trailer flags in the slot
                for tfg, cnt in slot_df.groupby(['date', 'is_trailer']).size().loc[date].iteritems():
                    feature[int(tfg)+20, slot_id] = cnt/total_count
        
        new_feature.append(feature)
        tqdm.write("extract user #{}".format(i))

test_feature = np.stack(new_feature)
np.save("test_features.npy", test_feature)

del new_feature, train_feature, train_label, test_feature

dataset = np.load('./datasets/v0_eigens.npz')
train_f = np.load('train_features.npy')
#train_l= np.load('train_labels.npy')
test_f = np.load('test_features.npy')

ori_train_f = dataset['train_eigens'][:, :-28]
ori_train_l = dataset['train_eigens'][:, -28:]
ori_test_f = dataset['issue_eigens'][:, :-28]
ori_test_l = dataset['issue_eigens'][:, -28:]

train_f[:,0,:] = (train_f[:,0,:] - train_f[:,0,:].min())/ (train_f[:,0,:].max() - train_f[:,0,:].min())
train_f[:,1,:] = (train_f[:,1,:] - train_f[:,1,:].min())/ (train_f[:,1,:].max() - train_f[:,1,:].min())

test_f[:,0,:] = (test_f[:,0,:] - test_f[:,0,:].min())/ (test_f[:,0,:].max() - test_f[:,0,:].min())
test_f[:,1,:] = (test_f[:,1,:] - test_f[:,1,:].min())/ (test_f[:,1,:].max() - test_f[:,1,:].min())

train_f = np.hstack((ori_train_f[:,None], train_f[:, :, 4:-28]))
train_l = ori_train_l
test_f = np.hstack((ori_test_f[:,None], test_f[:, :, 4:]))
test_l = ori_test_l

np.savez('kkstream_data.npz', train_eigens=train_f, train_labels=train_l, test_eigens=test_f, test_labels=test_l)