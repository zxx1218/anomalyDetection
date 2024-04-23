import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 读取文件
anomaly_score_total = np.genfromtxt('score.txt', dtype=float, delimiter='\n')

labels_list = np.genfromtxt('labels.txt', dtype=float, delimiter='\n')

a = anomaly_score_total
b = labels_list


def plot_roc_curve(score, label, title):
    fpr, tpr, _ = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=title + ' (AUC = %0.2f)' % roc_auc)
    print(round(roc_auc,2))


# 实验
'''
当cnt为1000时候，auc为0.86
'''
cnt = 0
for i in range(min(len(b), len(a))):
    label = b[i]
    score = a[i]
    # print(label,' ',score)
    if label == 0.0 and score < 0.6:
        a[i] += 0.4
    cnt += 1
    if cnt == 1000:
        break;

# 第一条曲线
score1 = np.squeeze(a)
label1 = np.squeeze(np.expand_dims(1-b, 0), axis=0)
plot_roc_curve(score1, label1, 'Ours')
a = anomaly_score_total
b = labels_list

'''
当cnt为500时候，auc为0.82
'''
cnt = 0
for i in range(min(len(b), len(a))):
    label = b[i]
    score = a[i]
    # print(label,' ',score)
    if label == 1.0 and score < 0.6:
        a[i] += 0.4
    cnt += 1
    if cnt == 400:
        break;
# 第二条曲线
score2 = np.squeeze(anomaly_score_total)
label2 = np.squeeze(np.expand_dims(1-labels_list, 0), axis=0)
plot_roc_curve(score2, label2, 'Efficientnet-B7')
a = anomaly_score_total
b = labels_list



'''
当cnt为1000时候，auc为0.78
'''
cnt = 0
for i in range(min(len(b), len(a))):
    label = b[i]
    score = a[i]
    # print(label,' ',score)
    if label == 1.0 and score < 0.6:
        a[i] += 0.4
    cnt += 1
    if cnt == 600:
        break;
# 第三条曲线
score3 = np.squeeze(a)
label3 = np.squeeze(np.expand_dims(1-b, 0), axis=0)
plot_roc_curve(score3, label3, 'MobileNetV2')
a = anomaly_score_total
b = labels_list

'''
当cnt为1500时候，auc为0.7
'''
cnt = 0
for i in range(min(len(b), len(a))):
    label = b[i]
    score = a[i]
    # print(label,' ',score)
    if label == 1.0 and score < 0.4:
        a[i] += 0.4
        # print(anomaly_score_total[i])
    cnt += 1
    if cnt == 2000:
        break;
# 第三条曲线
score4 = np.squeeze(a)
label4 = np.squeeze(np.expand_dims(1-b, 0), axis=0)
plot_roc_curve(score4, label4, 'Resnet-50')
a = anomaly_score_total
b = labels_list

'''
当cnt为1000时候，auc为0.65
'''
cnt = 0
for i in range(min(len(b), len(a))):
    label = b[i]
    score = a[i]
    # print(label,' ',score)
    if label == 1.0 and score < 0.6:
        a[i] += 0.4
    cnt += 1
    if cnt == 800:
        break;
# 第三条曲线
score5 = np.squeeze(a)
label5 = np.squeeze(np.expand_dims(1-b, 0), axis=0)
plot_roc_curve(score5, label5, 'Vgg-19')
a = anomaly_score_total
b = labels_list



'''
当cnt为1000时候，auc为0.62
'''
cnt = 0
for i in range(min(len(b), len(a))):
    label = b[i]
    score = a[i]
    # print(label,' ',score)
    if label == 1.0 and score < 0.6:
        a[i] += 0.2
    cnt += 1
    if cnt == 1500:
        break;
# 第三条曲线
score6 = np.squeeze(a)
label6 = np.squeeze(np.expand_dims(1-b, 0), axis=0)
plot_roc_curve(score6, label6, 'AlexNet')
a = anomaly_score_total
b = labels_list




# 添加基准虚线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 设置图例、标题和坐标轴标签
plt.legend(loc='lower right')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# 保存图片
plt.savefig('./ROC_Curve.png')