from tqdm import tqdm
from scipy import signal
import wfdb
import pandas as pd
import tsaug
import numpy as np

import torch
from enum import Enum

from env import PhysioEnv4Classes
from env import PhysioEnv3Classes







class Actions4(Enum):
    Relax = 1
    EmotionalStress = 2
    PhysicalStress = 3
    CognitiveStress = 4

class Actions3(Enum):
    Relax = 1
    EmotionalStress = 2
    PhysicalStress = 3



def calculate_rew_threshold(num_classes=4):
    
    action_list=np.zeros(num_classes)
    action_list[0]=1
    sm = torch.nn.Softmax(dim=0)
    action_list = torch.FloatTensor(action_list)
    actions_prob = sm(action_list)
    rew_threshold = max(actions_prob) / 2
    
    return rew_threshold

def calculate_rew_mean_threshold(num_classes=4):
    
    action_list=np.zeros(num_classes)
    action_list[0]=1
    sm = torch.nn.Softmax(dim=0)
    action_list = torch.FloatTensor(action_list)
    actions_prob = sm(action_list)
    rew_mean_threshold = max(actions_prob) * 0.95
    
    return rew_mean_threshold


def DataDownloader(test_index_subject=0):
    
    print('\n')
    print('###### data downlading  #######')
    print('\n')
    
    dataset = {}

    train_index_list = [i for i in range(1,21)]
    test_index = train_index_list.pop(test_index_subject)

    cont = 21

    for i in tqdm(train_index_list):
        r = wfdb.rdrecord(record_name='Subject{}_AccTempEDA'.format(i), pn_dir='noneeg/1.0.0')
        r1 = wfdb.rdrecord(record_name='Subject{}_SpO2HR'.format(i), pn_dir='noneeg/1.0.0')
        label = wfdb.io.rdann(record_name='Subject{}_AccTempEDA'.format(i), extension='atr', pn_dir='noneeg/1.0.0')
        label = label.sample
        sig = r.p_signal
        sig1 = r1.p_signal

        df = pd.DataFrame(sig)
        df1 = pd.DataFrame(sig1)

        f = signal.resample(df1, len(df))
        df1 = pd.DataFrame(f)

        spo2 = df1[0]
        hr = df1[1]
        df[5] = spo2
        df[6] = hr
        df.columns = ['ACCx', 'ACCy', 'ACCz', 'TEMP', 'EDA', 'SpO2', 'HR']
        df['ACCx'] = (df['ACCx'] + 2) / 4
        df['ACCy'] = (df['ACCy'] + 2) / 4
        df['ACCz'] = (df['ACCz'] + 2) / 4
        df['TEMP'] = df['TEMP'] / 115
        df['EDA'] = df['EDA'] / 10
        df['SpO2'] = df['SpO2'] / 100
        df['HR'] = df['HR'] / 220


        df_relax0 = df[label[0]:label[1]][:2380]

        df_relax1 = df[label[2]:label[3]][:2380]

        df_relax2 = df[label[5]:label[6]][:2380]

        df_relax3 = df[label[7]:][:2380]

        my_augmenter = (
                tsaug.TimeWarp(n_speed_change=7, max_speed_ratio=(1.02, 1.08)) @ 0.5
                + tsaug.Drift(max_drift=(0.02, 0.05), n_drift_points=5) @ 0.5
                + tsaug.AddNoise(scale=(0.005, 0.02)) @ 0.8
                + tsaug.Pool(size=(3, 7), per_channel=True) @ 0.5
                + tsaug.Quantize(n_levels=[30, 35, 40, 45, 50]) @ 0.5
        )

        df_PhysicalStress = df[label[1]:label[2]]
        cols = df_PhysicalStress.columns
        arr = np.array(df_PhysicalStress[df_PhysicalStress.columns])
        X = np.reshape(arr, (1, len(df_PhysicalStress), 7))
        X_aug1 = my_augmenter.augment(X)
        df_PhysicalStress1 = pd.DataFrame(X_aug1[0], columns=cols)
        X_aug2 = my_augmenter.augment(X)
        df_PhysicalStress2 = pd.DataFrame(X_aug2[0], columns=cols)
        X_aug3 = my_augmenter.augment(X)
        df_PhysicalStress3 = pd.DataFrame(X_aug3[0], columns=cols)

        df_EmotionalStress = df[label[3]:label[5]]

        cols = df_EmotionalStress.columns
        arr = np.array(df_EmotionalStress[df_EmotionalStress.columns])
        X = np.reshape(arr, (1, len(df_EmotionalStress), 7))
        X_aug1 = my_augmenter.augment(X)
        df_EmotionalStress1 = pd.DataFrame(X_aug1[0], columns=cols)
        X_aug2 = my_augmenter.augment(X)
        df_EmotionalStress2 = pd.DataFrame(X_aug2[0], columns=cols)
        X_aug3 = my_augmenter.augment(X)
        df_EmotionalStress3 = pd.DataFrame(X_aug3[0], columns=cols)

        df_CognitiveStress = df[label[6]:label[7]]

        cols = df_CognitiveStress.columns
        arr = np.array(df_CognitiveStress[df_CognitiveStress.columns])
        X = np.reshape(arr, (1, len(df_CognitiveStress), 7))
        X_aug1 = my_augmenter.augment(X)
        df_CognitiveStress1 = pd.DataFrame(X_aug1[0], columns=cols)
        X_aug2 = my_augmenter.augment(X)
        df_CognitiveStress2 = pd.DataFrame(X_aug2[0], columns=cols)
        X_aug3 = my_augmenter.augment(X)
        df_CognitiveStress3 = pd.DataFrame(X_aug3[0], columns=cols)

        dataset['Sogetto{}'.format(i)] = {

            'Relax': pd.concat([df_relax0, df_relax1, df_relax2, df_relax3], ignore_index=True),

            'PhysicalStress': pd.concat(
                [df_PhysicalStress, df_PhysicalStress1, df_PhysicalStress2, df_PhysicalStress3], ignore_index=True)[
                              :9520],

            'EmotionalStress': pd.concat(
                [df_EmotionalStress, df_EmotionalStress1[:1600], df_EmotionalStress2, df_EmotionalStress3],
                ignore_index=True)[:9520],
            
            'CognitiveStress': pd.concat(
                [df_CognitiveStress, df_CognitiveStress1[:1600], df_CognitiveStress2, df_CognitiveStress3],
                ignore_index=True)[:9520]
            
        }

        arr = np.array(df)
        # print(cont)
        X = np.reshape(arr, (1, len(df), 7))

        n = np.random.randint(cont + 1, cont + 6)
        cols = df.columns

        for j in range(cont, n):
            X_aug = my_augmenter.augment(X)
            df = pd.DataFrame(X_aug[0], columns=cols)

            df_relax0 = df[label[0]:label[1]][:2380]

            df_relax1 = df[label[2]:label[3]][:2380]

            df_relax2 = df[label[5]:label[6]][:2380]

            df_relax3 = df[label[7]:][:2380]

            df_PhysicalStress = df[label[1]:label[2]]
            cols = df_PhysicalStress.columns
            arr = np.array(df_PhysicalStress[df_PhysicalStress.columns])
            X1 = np.reshape(arr, (1, len(df_PhysicalStress), 7))
            X_aug1 = my_augmenter.augment(X1)
            df_PhysicalStress1 = pd.DataFrame(X_aug1[0], columns=cols)
            X_aug2 = my_augmenter.augment(X1)
            df_PhysicalStress2 = pd.DataFrame(X_aug2[0], columns=cols)
            X_aug3 = my_augmenter.augment(X1)
            df_PhysicalStress3 = pd.DataFrame(X_aug3[0], columns=cols)

            df_EmotionalStress = df[label[3]:label[5]]
            cols = df_EmotionalStress.columns
            arr = np.array(df_EmotionalStress[df_EmotionalStress.columns])
            X2 = np.reshape(arr, (1, len(df_EmotionalStress), 7))
            X_aug1 = my_augmenter.augment(X2)
            df_EmotionalStress1 = pd.DataFrame(X_aug1[0], columns=cols)
            X_aug2 = my_augmenter.augment(X2)
            df_EmotionalStress2 = pd.DataFrame(X_aug2[0], columns=cols)
            X_aug3 = my_augmenter.augment(X2)
            df_EmotionalStress3 = pd.DataFrame(X_aug3[0], columns=cols)

            df_CognitiveStress = df[label[6]:label[7]]
            cols = df_CognitiveStress.columns
            arr = np.array(df_CognitiveStress[df_CognitiveStress.columns])
            X3 = np.reshape(arr, (1, len(df_CognitiveStress), 7))
            X_aug1 = my_augmenter.augment(X3)
            df_CognitiveStress1 = pd.DataFrame(X_aug1[0], columns=cols)
            X_aug2 = my_augmenter.augment(X3)
            df_CognitiveStress2 = pd.DataFrame(X_aug2[0], columns=cols)
            X_aug3 = my_augmenter.augment(X3)
            df_CognitiveStress3 = pd.DataFrame(X_aug3[0], columns=cols)

            dataset['Sogetto{}'.format(j)] = {

                'Relax': pd.concat([df_relax0, df_relax1, df_relax2, df_relax3], ignore_index=True),

                'PhysicalStress': pd.concat(
                    [df_PhysicalStress, df_PhysicalStress1, df_PhysicalStress2, df_PhysicalStress3], ignore_index=True)[
                                  :9520],

                'EmotionalStress': pd.concat(
                    [df_EmotionalStress, df_EmotionalStress1, df_EmotionalStress2, df_EmotionalStress3],
                    ignore_index=True)[:9520],
                
                'CognitiveStress': pd.concat(
                    [df_CognitiveStress, df_CognitiveStress1, df_CognitiveStress2, df_CognitiveStress3],
                    ignore_index=True)[:9520]

            }

        cont += (n - cont)

    train = dataset

    dataset = {}

    r = wfdb.rdrecord(record_name='Subject{}_AccTempEDA'.format(test_index), pn_dir='noneeg/1.0.0')
    r1 = wfdb.rdrecord(record_name='Subject{}_SpO2HR'.format(test_index), pn_dir='noneeg/1.0.0')
    label = wfdb.io.rdann(record_name='Subject{}_AccTempEDA'.format(test_index), extension='atr', pn_dir='noneeg/1.0.0')
    label = label.sample
    sig = r.p_signal
    sig1 = r1.p_signal
    df = pd.DataFrame(sig)
    df1 = pd.DataFrame(sig1)
    f = signal.resample(df1, len(df))
    df1 = pd.DataFrame(f)
    spo2 = df1[0]
    hr = df1[1]
    df[5] = spo2
    df[6] = hr
    df.columns = ['ACCx', 'ACCy', 'ACCz', 'TEMP', 'EDA', 'SpO2', 'HR']
    df['ACCx'] = (df['ACCx'] + 2) / 4
    df['ACCy'] = (df['ACCy'] + 2) / 4
    df['ACCz'] = (df['ACCz'] + 2) / 4
    df['TEMP'] = df['TEMP'] / 115
    df['EDA'] = df['EDA'] / 10
    df['SpO2'] = df['SpO2'] / 100
    df['HR'] = df['HR'] / 220

    df_relax0 = df[label[0]:label[1]]
    df_relax1 = df[label[2]:label[3]]
    df_relax2 = df[label[5]:label[6]]
    df_relax3 = df[label[7]:]
    df_PhysicalStress = df[label[1]:label[2]]
    df_EmotionalStress = df[label[3]:label[5]]
    df_CognitiveStress = df[label[6]:label[7]]

    dataset['Sogetto{}'.format(test_index)] = {

        'Relax': pd.concat([df_relax0, df_relax1, df_relax2, df_relax3], ignore_index=True),
        'PhysicalStress': df_PhysicalStress,
        'EmotionalStress': df_EmotionalStress,
        'CognitiveStress': df_CognitiveStress
        
    }

    test = dataset

    print('\n')
    print('####  Downlad --> completed  ####')
    print('\n')
    print('####  Preprocessing --> completed  ####')
    print('\n')
    
    return train, test


#train, test = DataDownloader(test_index_subject=3)  # download and create dataset






def Evaluate4Classes(test,model,episode_max_len=300,window_size=8,subject=1):

    print('\n')
    print('Test...')
    print('\n')
    
    cont = 0
    y_test = []
    y_pred = []
    non_classified = 0
    classification_time = []
    s=subject
    
    for sogg in test.keys():
    
        for classe in test[sogg].keys():
            print('sogg {}'.format(sogg))
            print('classe {}'.format(classe))
            class_len = len(test[sogg][classe])
            num_episode_class = class_len // episode_max_len
            print(num_episode_class)
            episode_steps=0
            while (episode_steps + episode_max_len) < class_len-1:
    
                episode_data = test[sogg][classe][episode_steps:episode_steps + episode_max_len]
    
                print('\n')
                print('Episode test {}'.format(cont))
                print('\n')
                test_env = PhysioEnv4Classes(data=episode_data, window_size=window_size, is_train=False,
                                     episode_max_len=episode_max_len, sogg=sogg,
                                     classe=classe)  # crea un istanza dell'ambiente di trading di test
                obs = test_env.reset()
   
                total_reward = 0

                done = False
    
                while not done:
    
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = test_env.step(action)
                    # episode_starts = done
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward}")
                    print(info)
    
                episode_steps += (test_env._time + 1)
    
                sm = torch.nn.Softmax(dim=0)
                a = torch.FloatTensor(action)
                actions_prob = sm(a)
    
                pred = max(actions_prob)
    
                if pred == actions_prob[0]:
                    action = 1
                elif pred == actions_prob[1]:
                    action = 2
                elif pred == actions_prob[2]:
                    action = 3
                elif pred == actions_prob[3]:
                    action = 4
                else:
                    action = 5
    
                if classe == 'Relax':
                    label = Actions4.Relax.value
                elif classe == 'PhysicalStress':
                    label = Actions4.PhisicalStress.value
                elif classe == 'EmotionalStress':
                    label = Actions4.EmotionalStress.value
                elif classe == 'CognitiveStress':
                    label = Actions4.CognitiveStress.value
    
    
                if action <= 4:
                    y_test.append(label)
                    y_pred.append(action)
                    classification_time.append(test_env._time)
                else:
                    non_classified += 1
    
    
    
                cont += 1
    
    test_results = pd.DataFrame()
    test_results['y_true'] = y_test
    test_results['y_pred'] = y_pred
    test_results['classification_time'] = classification_time
    test_results.to_csv('./StressSAC_Train4Classes_mlp_Test_Result_Sogg{}.csv'.format(s))
    model.save('./StressSAC_Train4Classes_mlp_TunedModel_Sogg{}'.format(s))



def Evaluate3Classes(test,model,episode_max_len=300,window_size=8,subject=1):

    print('\n')
    print('Test...')
    print('\n')
    
    cont = 0
    y_test = []
    y_pred = []
    non_classified = 0
    classification_time = []
    s=subject
    
    for sogg in test.keys():
    
        for classe in test[sogg].keys():
            print('sogg {}'.format(sogg))
            print('classe {}'.format(classe))
            class_len = len(test[sogg][classe])
            num_episode_class = class_len // episode_max_len
            print(num_episode_class)
            episode_steps=0
            while (episode_steps + episode_max_len) < class_len-1:
    
                episode_data = test[sogg][classe][episode_steps:episode_steps + episode_max_len]
    
                print('\n')
                print('Episode test {}'.format(cont))
                print('\n')
                test_env = PhysioEnv3Classes(data=episode_data, window_size=window_size, is_train=False,
                                     episode_max_len=episode_max_len, sogg=sogg,
                                     classe=classe)  # crea un istanza dell'ambiente di trading di test
                obs = test_env.reset()

                total_reward = 0
                done = False
    
                while not done:
    
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = test_env.step(action)
                    # episode_starts = done
                    total_reward += reward
                    print(f"Action: {action}, Reward: {reward}")
                    print(info)
    
                episode_steps += (test_env._time + 1)
    
                sm = torch.nn.Softmax(dim=0)
                a = torch.FloatTensor(action)
                actions_prob = sm(a)
    
                pred = max(actions_prob)
    
                if pred == actions_prob[0]:
                    action = 1
                elif pred == actions_prob[1]:
                    action = 2
                elif pred == actions_prob[2]:
                    action = 3
                else:
                    action = 4
    
                if classe == 'Relax':
                    label = Actions3.Relax.value
                elif classe == 'PhysicalStress':
                    label = Actions3.PhisicalStress.value
                elif classe == 'EmotionalStress':
                    label = Actions3.EmotionalStress.value

    
    
                if action <= 3:
                    y_test.append(label)
                    y_pred.append(action)
                    classification_time.append(test_env._time)
                else:
                    non_classified += 1
    
    
    
                cont += 1
    
    test_results = pd.DataFrame()
    test_results['y_true'] = y_test
    test_results['y_pred'] = y_pred
    test_results['classification_time'] = classification_time
    test_results.to_csv('./StressSAC_Train3Classe_mlp_Test_Result_Sogg{}.csv'.format(s))
    model.save('./StressSAC_Train3Classe_mlp_TunedModel_Sogg{}'.format(s))
