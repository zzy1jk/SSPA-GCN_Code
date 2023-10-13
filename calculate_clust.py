import torch
import numpy as np
from dif_similarity_measure_method_init import mmd_rbf,KL_calculate,L2_calculate

def MMD_matrix_construct(Feature_data,del_same_result,all_sub_MMD_matrix):
    N, T, C, F = Feature_data.shape
    #加快运行速度，只更新改变的值；
    flag = 0 #使初始为空时也能运行
    for sub_num_1 in del_same_result:
        if flag == 0: #相当于只取del_same_result列表中的第一个值进行计算
            flag = 1
            for sub_num_2 in range(0,N):
                #优化代码，加快计算速度
                if sub_num_2 not in del_same_result:
                    MMD_list = []
                    for time_cut in range(T):
                        source_1 = Feature_data[sub_num_1,time_cut,:,:]
                        source_2 = Feature_data[sub_num_2,time_cut,:,:]
                        X = torch.Tensor(source_1)
                        Y = torch.Tensor(source_2)
                        MMD_temp = mmd_rbf(X,Y).numpy()
                        MMD_list.append(MMD_temp)
                    all_time_cut_average_MMD = sum(MMD_list)/len(MMD_list)
                    all_sub_MMD_matrix[del_same_result,sub_num_2] = all_time_cut_average_MMD
                    all_sub_MMD_matrix[sub_num_2, del_same_result] = all_time_cut_average_MMD
                else:
                    all_sub_MMD_matrix[del_same_result,sub_num_2] = 0
                    all_sub_MMD_matrix[sub_num_2, del_same_result] = 0
    MMD_matrix = np.around(all_sub_MMD_matrix, 5)
    #将下三角置为0,i>=j,MMD_Matrix = 0
    N = MMD_matrix.shape[0]
    for ii in range(N):
        for jj in range(0,ii):
            MMD_matrix[ii,jj] = 0
    return MMD_matrix

def average_feature(Feature_matrix,min_value_pos):
    """
    根据相似性矩阵得到最相近的多个个体下标i和j，求i,j个体的特征平均值，
    并将特征矩阵中i和j的位置替换为计算后的平均值
    return:Feature_matrix(计算均值之后的)
    """
    N,T,C,F = Feature_matrix.shape
    need_average_index = (np.unique(min_value_pos.reshape(-1))).tolist()
    length = len(need_average_index)
    sub_sum_data = np.zeros((T,C,F))
    for sub_index in need_average_index:
        sub_sum_data += Feature_matrix[sub_index, :, :, :] #[150,128,9]
        sub_mean_data = (sub_sum_data/length ).reshape(1,T,C,F)#[1,150,128,9]
    Feature_matrix[need_average_index]  = sub_mean_data
    return Feature_matrix

if __name__ == '__main__':
    Feature_matrix_next_in = np.load("../DE_feature_53_Nor.npy")
    clust_result = []
    del_same_result = []
    N, T, C, F = Feature_matrix_next_in.shape
    MMD_matrix_next_in = np.load("MODMA_DataSet_all_sub_MMD_matrix.npy") #初始化MMD矩阵由init_MMD_matrix.py计算
    filename = open('clust_result.txt', 'w')
    while(1):
        MMD_matrix_out = MMD_matrix_construct(Feature_matrix_next_in,del_same_result,MMD_matrix_next_in)
        if np.all(MMD_matrix_out == 0):  #所有样本聚成一类,终止循环
            break
        MMD_matrix_next_in = MMD_matrix_out
        #寻找除了0之外的最小值
        MMD_matrix = MMD_matrix_out.reshape(-1)
        MMD_matrix = MMD_matrix[MMD_matrix != 0]
        MMD_matrix = np.sort(MMD_matrix)
        MMD_min = MMD_matrix[0]
        print(MMD_min)

        #定位最小值的下标
        min_value_pos = np.argwhere(MMD_matrix_out == MMD_min)
        # print("当前MMD最小值对应的下标(min_value_pos)：", min_value_pos)

        Feature_matrix = average_feature(Feature_matrix_next_in, min_value_pos)
        Feature_matrix_next_in = Feature_matrix
        del_same_result = (np.unique(min_value_pos.reshape(-1)).tolist()) #删除重复元素并排序
        filename.write(str(del_same_result) + "\n")
        clust_result.append(del_same_result)
        print("层次聚类结果：",del_same_result)
        # print("----------------------------------------")
    filename.close()




