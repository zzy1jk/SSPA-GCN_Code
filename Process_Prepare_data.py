import torch
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import xlrd
import os
import re
from natsort import os_sorted
import math
import numpy as np
from scipy.fftpack import fft, ifft

class TwoDataset:
    """
    MODMA和PRED+CT数据集基类,
    后续所有类均需要继承该父类
    """
    def __init__(self, Dataset):
        self.Dataset = Dataset
        self.class_num = 2
        self.sub_num = 53 if self.Dataset == "MODMA" else 86  # 数据集中的受试者数量
        self.sample_rate = 250 if self.Dataset == "MODMA" else 500  # 采样频率，Hz
        self.window_len = 2  # 时间窗口长度：2s
        self.chan_num = 128 if self.Dataset == "MODMA" else 66
        self.win_num = 150
        self.FileNameFirstChar = 'M' if self.Dataset == "MODMA" else 'P'
        self.EEGrawdata_path = "E:\MyCode_zzy\Depression_cly_idea_1\MODMA _230208\MODMA_DataSet/" \
            if self.Dataset == "MODMA" else "E:\EEG_Depression\Dataset_MODMA_PREDICT\OneDrive_2022-03-14\Depression Rest\Matlab Files/"
        self.excelpath = "E:\MyCode_zzy\Depression_cly_idea_1\MODMA _230208\subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx" \
            if self.Dataset == "MODMA" \
            else "E:\EEG_Depression\Dataset_MODMA_PREDICT\OneDrive_2022-03-14\Depression Rest\Data_4_Import_REST.xlsx"

        self.stftn = 500
        self.fStart = [0.5, 4, 8, 13, 30]
        self.fEnd = [4, 8, 13, 30, 100]


class ProcessRawData(TwoDataset):
    """
    P:PREDICT数据集
    M：MODMA数据集
    通过该类获得的数据内容：
    1.标签列表；
    2.2s时间窗的EEG信号片段，共150段,数据形式[sub_num,150,chan_num,sample_rate*2s]
    """

    def __init__(self, Dataset):
        super().__init__(Dataset)

    def assist_site_lan_country(self, site_lan_country_str):
        re_pattern = r"_"
        re_compile = re.compile(pattern=re_pattern)
        split_str_list = re_compile.split(site_lan_country_str)
        return split_str_list

    def ReadRawDataSet(self, file_name, subjectname):
        read_mat = sio.loadmat(self.EEGrawdata_path + file_name)  # 数据类型，<dictionary>
        RawDataSet = read_mat[subjectname]
        return RawDataSet

    def get_subject_name(self):
        """
        获得每一个受试者样本的抑郁状态
        :param file_path: 文件夹路径
        :return: list:file_name_list, list:subject_name, list:subject_Label
                1: depress
                0: normal
        """
        subject_name_list = []
        file_name_list = os_sorted(os.listdir(self.EEGrawdata_path))
        for file_name in file_name_list:
            divide = re.split(r'[.\s+]', file_name)
            subject_name = 'a' + divide[0] + '_' + divide[1] + '_' + divide[2] + 'mat'
            subject_name_list.append(subject_name)
        return file_name_list, subject_name_list

    # 归一化
    def normalization(self, data, norm_dim):
        max_data = np.max(data, norm_dim, keepdims=True)
        min_data = np.min(data, norm_dim, keepdims=True)
        mean_data = np.mean(data, norm_dim, keepdims=True)
        sigma = np.std(data, norm_dim, keepdims=True)
        base = max_data - min_data
        normalized_data = (data - min_data) / base
        return normalized_data

    ## 微分熵特征计算
    def feature_de_construct(self, data):
        '''
        计算一个2s时间窗内所有通道的DE数据
        '''
        fStartNum = np.zeros([len(self.fStart)], dtype=int)
        fEndNum = np.zeros([len(self.fEnd)], dtype=int)
        for i in range(0, len(self.fStart)):
            fStartNum[i] = int(self.fStart[i] / self.sample_rate * self.stftn)
            fEndNum[i] = int(self.fEnd[i] / self.sample_rate * self.stftn)
        de = np.zeros([self.chan_num, len(self.fStart)])
        Hlength = self.window_len * self.sample_rate
        Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])

        dataNow = data[0:self.chan_num]
        for j in range(self.chan_num):
            temp = dataNow[j]
            Hdata = temp * Hwindow
            FFTdata = fft(Hdata, self.stftn)
            magFFTdata = abs(FFTdata[0:int(self.stftn / 2)])
            for p in range(0, len(self.fStart)):
                E = 0
                for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                    E = E + magFFTdata[p0] * magFFTdata[p0]
                E = E / (fEndNum[p] - fStartNum[p] + 1)
                de[j][p] = 100 * math.log(100 * E, 2)
        DE_one_2s_window = np.expand_dims(de, axis=0)
        return DE_one_2s_window

    def DE_conduction(self):
        file_name_list, subject_name = self.get_subject_name()
        DE_all_time_windows = np.zeros([self.win_num, self.chan_num, len(self.fStart)], dtype=float)  # 150*128*5
        Out_DE_allsub = np.zeros([self.sub_num, self.win_num, self.chan_num, len(self.fStart)], dtype=float)
        i = 0
        for filename in file_name_list:
            data = self.ReadRawDataSet(filename, subject_name[i])
            for time_step in range(self.win_num):
                data_time = data[0:self.chan_num,
                            self.stftn * time_step:self.stftn * (time_step + 1)]
                DE_all_time_windows[time_step] = self.feature_de_construct(data_time)
            Out_DE_allsub[i] = np.expand_dims(DE_all_time_windows, axis=0)  # 第i个受试者的所有DE数据
            i += 1
        print('DE返回形状:', Out_DE_allsub.shape)
        return Out_DE_allsub

    def get_graph_data(self):
        Processed_Data_NoNor = np.load('./NpyFile/' + Dataset[0] + "_DE_feature.npy", allow_pickle=True)
        # 特征数据归一化后再计算相关性
        Processed_Data = self.normalization(Processed_Data_NoNor, 2)
        Graph = np.zeros([self.sub_num, self.win_num, self.chan_num, self.chan_num], dtype=float)
        a = np.zeros([self.chan_num, self.chan_num], dtype=float)
        for m in range(self.sub_num):
            for t in range(self.win_num):
                for i in range(self.chan_num):
                    for j in range(self.chan_num):
                        x1 = Processed_Data[m][t]
                        if i == j:
                            a[i][j] = 1
                        else:
                            corr = np.corrcoef(x1[i], x1[j])
                            a[i][j] = 1 if np.all(abs(corr) >= 0.3) else 0
                Graph[m][t] = np.expand_dims(a, axis=0)
            print("生成图进度：", m)
        print(Graph.shape)
        return Graph


class PrepareData(TwoDataset):
    def __init__(self, Dataset, domain_num, use_soft_label, Band_index):
        super().__init__(Dataset)
        self.domain_num = domain_num
        self.use_soft_label = use_soft_label
        self.Band_index = Band_index

    def get_subject_PHQ9_score(self):
        excel = xlrd.open_workbook(self.excelpath)
        sheet = excel.sheet_by_name("Sheet1")
        PHQ9_data = np.array(sheet.col_values(5)[1:54])
        # print(type(PHQ9_data),PHQ9_data.shape,PHQ9_data)
        return PHQ9_data

    def get_subject_GAD7_score(self):
        excel = xlrd.open_workbook(self.excelpath)
        sheet = excel.sheet_by_name("Sheet1")
        GAD7_data = np.array(sheet.col_values(9)[1:54])
        # print(type(PHQ9_data),PHQ9_data.shape,PHQ9_data)
        return GAD7_data

    def get_subject_BDI_score(self):
        excel = xlrd.open_workbook(self.excelpath)
        sheet = excel.sheet_by_name("Depression Rest")
        BDI_data = np.array(sheet.col_values(6)[1:87])
        # print(type(BDI_data),BDI_data.shape,BDI_data)
        return BDI_data

    def read_npy_Nor(self):
        DE_Data_NoNor = np.load("./NpyFile/" + self.FileNameFirstChar + "_DE_feature.npy")
        DE_Data = ProcessRawData(Dataset).normalization(DE_Data_NoNor, norm_dim=2)
        graph_data = np.load("./NpyFile/" + self.FileNameFirstChar + "_graph_2s_window.npy")
        domain_label = np.load("./NpyFile/" + self.FileNameFirstChar + "_domain_label_" + str(self.domain_num) + ".npy")

        # 使用单频带数据实验
        if self.Band_index != 5:
            DE_Data = DE_Data[:, :, :, self.Band_index].reshape(self.sub_num, self.win_num, self.chan_num, 1)

        return DE_Data, graph_data, domain_label

    def soft_label_con(self):
        label_all_timecut = np.empty((0, self.class_num))
        if self.Dataset == "MODMA":
            if self.use_soft_label:
                # 制作软标签并返回
                PHQ9_data = self.get_subject_PHQ9_score()
                GAD7_data = self.get_subject_GAD7_score()
                for PHQ9_score, GAD7_score in zip(PHQ9_data, GAD7_data):
                    # label_first_num = 1.2*(PHQ9_score / 27.) - 0.2*(GAD7_score / 21.)
                    label_first_num = PHQ9_score / 27.
                    label_second_num = 1. - label_first_num
                    for ii in range(self.win_num):
                        label_all_timecut = np.append(label_all_timecut,
                                                      np.array([[label_first_num, label_second_num]]),
                                                      axis=0)
                # print(label_all_timecut.shape,label_all_timecut[26])
            else:
                # 制作传统1,0标签并返回
                label_dep = np.tile([1, 0], (24, 150, 1))
                label_nor = np.tile([0, 1], (29, 150, 1))
                label_all_timecut = np.concatenate((label_dep, label_nor), 0)
                # print(label_all_timecut.shape)
        else:
            Label_86 = np.load("./NpyFile/DE_feature_Predict.npz")["Label_86"]
            if self.use_soft_label:
                # 根据BDI制作软标签
                BDI_data = self.get_subject_BDI_score()
                for subject_BDI in BDI_data:
                    label_first_num = subject_BDI / 30.
                    label_second_num = 1 - label_first_num
                    for ii in range(self.win_num):
                        label_all_timecut = np.append(label_all_timecut,
                                                      np.array([[label_first_num, label_second_num]]), axis=0)
            else:
                # 制作传统1,0标签并返回
                label_all_timecut = np.empty((0, self.class_num))
                for label_subject in Label_86:
                    if label_subject == 1:
                        for ii in range(self.win_num):
                            label_all_timecut = np.append(label_all_timecut, np.array([[1, 0]]), axis=0)
                    if label_subject == 0:
                        for ii in range(self.win_num):
                            label_all_timecut = np.append(label_all_timecut, np.array([[0, 1]]), axis=0)

        label_all_timecut = label_all_timecut.reshape(self.sub_num, self.win_num, self.class_num)
        return label_all_timecut

    def one_test_data_construct(self, test_subject_index):
        DE_Data, graph_data, domain_label = self.read_npy_Nor()
        label_all_timecut = self.soft_label_con()
        # 将一个受试者的测试数据放到数据的最后一行
        DE_Data_del_one_subject = np.delete(DE_Data, test_subject_index, 0)
        subject_test_data = DE_Data[test_subject_index].reshape(1, self.win_num, self.chan_num, DE_Data.shape[-1])
        DE_Data_ = np.concatenate((DE_Data_del_one_subject, subject_test_data), 0)

        graph_data_del_one_subject = np.delete(graph_data, test_subject_index, 0)
        graph_data_one_subject = graph_data[test_subject_index].reshape(1, self.win_num, self.chan_num, self.chan_num)
        graph_data_ = np.concatenate((graph_data_del_one_subject, graph_data_one_subject), 0)

        label_all_timecut_del_one_subject = np.delete(label_all_timecut, test_subject_index, 0)
        label_all_timecut_one_subject = label_all_timecut[test_subject_index].reshape(1, self.win_num, self.class_num)
        label_all_timecut_ = np.concatenate((label_all_timecut_del_one_subject, label_all_timecut_one_subject), 0)

        domain_label_del_one_subject = np.delete(domain_label, test_subject_index, 0)
        domain_label_one_subject = domain_label[test_subject_index].reshape(1, self.win_num,
                                                                            self.domain_num)  # [1,150,domain_num]
        domain_label_ = np.concatenate((domain_label_del_one_subject, domain_label_one_subject), 0)

        # 变换数据以时间片段为最小计算单元
        DE_Data_reshape = DE_Data_.reshape(self.sub_num * self.win_num, self.chan_num, DE_Data.shape[-1])
        graph_data_reshape = graph_data_.reshape(self.sub_num * self.win_num, self.chan_num, self.chan_num)
        label_all_timecut_reshape = label_all_timecut_.reshape(self.sub_num * self.win_num, self.class_num)
        domain_label_reshape = domain_label_.reshape(self.sub_num * self.win_num, self.domain_num)

        return DE_Data_reshape, graph_data_reshape, label_all_timecut_reshape, domain_label_reshape


class MyDataSet(Dataset):
    def __init__(self, Dataset, DE_Data_reshape, graph_data_reshape, label_all_timecut_reshape, domain_label_reshape,
                 train_mode):
        self.Dataset = Dataset
        self.DE_Data_reshape = DE_Data_reshape  # [53*150,128,5]
        self.label_all_timecut_reshape = label_all_timecut_reshape
        self.graph_data_reshape = graph_data_reshape
        self.domain_label_reshape = domain_label_reshape
        self.train_mode = train_mode
        self.train_num = 52 * 150 if self.Dataset == "MODMA" else 85 * 150
        self.test_num = 150

    def __len__(self):
        return self.train_num if self.train_mode == "train" else self.test_num

    def __getitem__(self, index):  # (x, y)
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_num

        DE_data_net = torch.as_tensor(self.DE_Data_reshape[index, :, :], dtype=torch.float)
        graph_data_net = torch.as_tensor(self.graph_data_reshape[index, :, :], dtype=torch.float)
        class_label_net = torch.as_tensor(self.label_all_timecut_reshape[index, :], dtype=torch.float)
        domain_label_net = torch.as_tensor(self.domain_label_reshape[index, :], dtype=torch.float)

        return {"DE_data_net": DE_data_net, "graph_data_net": graph_data_net, "class_label_net": class_label_net,
                "domain_label_net": domain_label_net}


def train_loader_test_loader(Dataset, DE_Data_reshape, graph_data_reshape, label_all_timecut_reshape,
                             domain_label_reshape):
    train_data = MyDataSet(Dataset, DE_Data_reshape, graph_data_reshape, label_all_timecut_reshape,
                           domain_label_reshape,
                           train_mode="train")
    train_loader = DataLoader(train_data, batch_size=150 * 5 if Dataset == "MODMA" else 150 * 10,
                              shuffle=True)

    test_data = MyDataSet(Dataset, DE_Data_reshape, graph_data_reshape, label_all_timecut_reshape, domain_label_reshape,
                          train_mode="test")
    test_loader = DataLoader(test_data, batch_size=150, shuffle=False)
    return train_data, train_loader, test_data, test_loader


if __name__ == '__main__':
    # 文件路径
    Dataset = "PRED+CT"
    example = ProcessRawData(Dataset)
    DE_feature = example.DE_conduction()
    np.save('./NpyFile/' + Dataset[0] + "_DE_feature", DE_feature)
    print(Dataset + '微分熵特征矩阵计算完成！')
