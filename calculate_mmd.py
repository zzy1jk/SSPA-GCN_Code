import torch
import numpy as np

def MMD_matrix_construct(Feature_data,del_same_result,all_sub_MMD_matrix):
    N, T, C, F = Feature_data.shape
    for sub_num_1 in del_same_result:
        if flag == 0:
            flag = 1
            for sub_num_2 in range(0,N):
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
    N = MMD_matrix.shape[0]
    for ii in range(N):
        for jj in range(0,ii):
            MMD_matrix[ii,jj] = 0
    return MMD_matrix

def average_feature(Feature_matrix,min_value_pos):
    N,T,C,F = Feature_matrix.shape
    need_average_index = (np.unique(min_value_pos.reshape(-1))).tolist()
    length = len(need_average_index)
    sub_sum_data = np.zeros((T,C,F))
    for sub_index in need_average_index:
        sub_sum_data += Feature_matrix[sub_index, :, :, :]
        sub_mean_data = (sub_sum_data/length).reshape(1,T,C,F)
    Feature_matrix[need_average_index]  = sub_mean_data
    return Feature_matrix

if __name__ == '__main__':
    Feature_matrix_next_in = np.load("../DE_feature_53_Nor.npy")
    clust_result = []
    del_same_result = []
    N, T, C, F = Feature_matrix_next_in.shape
    MMD_matrix_next_in = np.load("MODMA_DataSet_all_sub_MMD_matrix.npy")
    filename = open('clust_result.txt', 'w')
    while(1):
        MMD_matrix_out = MMD_matrix_construct(Feature_matrix_next_in,del_same_result,MMD_matrix_next_in)
        if np.all(MMD_matrix_out == 0):
            break
        MMD_matrix_next_in = MMD_matrix_out
        MMD_matrix = MMD_matrix_out.reshape(-1)
        MMD_matrix = MMD_matrix[MMD_matrix != 0]
        MMD_matrix = np.sort(MMD_matrix)
        MMD_min = MMD_matrix[0]
        min_value_pos = np.argwhere(MMD_matrix_out == MMD_min)

        Feature_matrix = average_feature(Feature_matrix_next_in, min_value_pos)
        Feature_matrix_next_in = Feature_matrix
        del_same_result = (np.unique(min_value_pos.reshape(-1)).tolist()) 
        filename.write(str(del_same_result) + "\n")
        clust_result.append(del_same_result)
    filename.close()
