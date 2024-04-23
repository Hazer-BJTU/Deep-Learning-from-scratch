import torch
import numpy as np

NUMBER = 1000

if __name__ == '__main__':
    height_normal = np.random.normal(loc=165, scale=7.5, size=[NUMBER*2])
    height_obese = np.random.normal(loc=165, scale=7.5, size=[NUMBER*2])
    bmi_normal = np.random.normal(loc=21.25, scale=1.375, size=[NUMBER*2])
    bmi_obese = np.random.normal(loc=26, scale=1, size=[NUMBER*2])
    weight_normal = np.zeros(shape=[NUMBER*2], dtype=float)
    weight_obese = np.zeros(shape=[NUMBER*2], dtype=float)
    for i in range(NUMBER*2):
        weight_normal[i] = bmi_normal[i] * (height_normal[i] / 100) ** 2
        weight_obese[i] = bmi_obese[i] * (height_obese[i] / 100) ** 2
    print(f'Height normal(cm): {height_normal}')
    print(f'Weight normal(kg): {weight_normal}')
    print(f'Height obese(cm): {height_obese}')
    print(f'Weight obese(kg): {weight_obese}')

    normal = np.concatenate((np.array([height_normal[0:NUMBER]]), np.array([weight_normal[0:NUMBER]])))
    obese = np.concatenate((np.array([height_obese[0:NUMBER]]), np.array([weight_obese[0:NUMBER]])))

    normal_test = np.concatenate((np.array([height_normal[NUMBER:NUMBER*2]]), np.array([weight_normal[NUMBER:NUMBER*2]])))
    obese_test = np.concatenate((np.array([height_obese[NUMBER:NUMBER*2]]), np.array([weight_obese[NUMBER:NUMBER*2]])))

    u_normal = np.mean(normal, axis=1)
    u_normal = np.array([u_normal]).T
    u_obese = np.mean(obese, axis=1)
    u_obese = np.array([u_obese]).T
    print(f'u_normal: {u_normal}')
    print(f'u_obese: {u_obese}')

    sigma_normal = np.cov(normal)
    sigma_obese = np.cov(obese)
    print(f'sigma_normal: {sigma_normal}')
    print(f'sigma_obese: {sigma_obese}')

    S_w = sigma_normal + sigma_obese
    S_w_inv = np.linalg.pinv(S_w)
    w = S_w_inv @ (u_normal - u_obese)
    print(f'w: {w}')

    ans_normal = w.T @ normal_test
    ans_obese = w.T @ obese_test
    ans_data_normal = w.T @ normal
    ans_data_obese = w.T @ obese

    num_true, num_false, num_data_true, num_data_false = 0, 0, 0, 0
    judge_normal, judge_obese = w.T @ u_normal, w.T @ u_obese
    for i, j in zip(ans_normal[0], ans_data_normal[0]):
        if abs(i-judge_normal) < abs(i-judge_obese):
            num_true += 1
        else:
            num_false += 1
        if abs(j-judge_normal) < abs(j-judge_obese):
            num_data_true += 1
        else:
            num_data_false += 1
    for i, j in zip(ans_obese[0], ans_data_obese[0]):
        if abs(i-judge_normal) > abs(i-judge_obese):
            num_true += 1
        else:
            num_false += 1
        if abs(j-judge_normal) > abs(j-judge_obese):
            num_data_true += 1
        else:
            num_data_false += 1
    acc = num_true / (num_true + num_false)
    acc_data = num_data_true / (num_data_true + num_data_false)
    print(f'Train accuracy: {acc_data}')
    print(f'Test accuracy1: {acc}')

    num_true2, num_false2 = 0, 0
    for i in normal_test.T:
        x = np.array([i]).T
        if np.linalg.norm(x-u_normal) < np.linalg.norm(x-u_obese):
            num_true2 += 1
        else:
            num_false2 += 1
    for i in obese_test.T:
        x = np.array([i]).T
        if np.linalg.norm(x-u_normal) < np.linalg.norm(x-u_obese):
            num_false2 += 1
        else:
            num_true2 += 1
    acc2 = num_true2 / (num_true2 + num_false2)
    print(f'Test accuracy2: {acc2}')
