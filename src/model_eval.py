import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve

def evaluate_auc(model, X, y, model_name):
    plt.figure(figsize=(10,6))

    y_pred = model.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    plt.plot(fpr, tpr, color='red', lw=2, label=f'{model_name} (area = %0.4f)' % auc(fpr, tpr))

    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle= '-', label = 'Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=17)
    plt.legend(loc='lower right', fontsize=13)
    plt.show()