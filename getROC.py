import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def save_roc_curve(anomal_scores, labels, save_dir):
    fpr, tpr, _ = roc_curve(labels, anomal_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存图表
    save_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(save_path)
    return roc_auc



# # 指定保存目录
# save_directory = '/path/to/save_directory'

# # 使用函数绘制 ROC 曲线并保存
# save_roc_curve(anomal_scores, labels, save_directory)