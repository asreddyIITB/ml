import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, roc_curve

def plot_errors(y_obs, y_pred):
    """
    Plot errors to evaluate predictions

    Arguments:
        y_obs: observed values for y
        y_pred: predicted values for y


    Returns:
        Plots (a) scatter plot of residual = (y_obs-y_pred) 
              (b) KDE plot of residuals
    
    """

    errors = y_obs - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    axes[0].scatter(np.arange(errors.shape[0]), errors)
    axes[0].set(xlabel='Observation Num', ylabel='Error')

    errors.plot(kind='kde', ax=axes[1])
    axes[1].set_xlabel('Error')

    plt.suptitle('Error Plot')
    plt.show()
    
    return axes



def plot_corr(df):
    """
    Plot correlation matrix for each pair of columns in the dataframe.

    Arguments:
        df: pandas DataFrame

    Returns:
        Plots correlation matrix df.corr()
        returns correlation matrix
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.matshow(corr)
    tick_marks = [i for i in range(len(corr.columns))]
    plt.xticks(tick_marks, corr.columns, rotation='vertical')
    plt.yticks(tick_marks, corr.columns)
    plt.show()

    return corr


def plot_confusion_matrix(y_obs, y_pred, class_labels):
    """
    Compute/Plot confusion matrix for classification problem.

    Arguments:
        y_obs: observed values for y
        y_pred: predicted values for y
        class_labels: class names to which observations are mapped

    Returns:
        Plots confusion matrix
        returns confusion matrix
    """
    
    cm = confusion_matrix(y_obs, y_pred)
    df_cm = pd.DataFrame(cm)

    ax = plt.axes()
    sns.set(font_scale=1.25)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="g", ax=ax, cmap="magma")

    ax.set_title('Confusion Matrix')
    ax.set_xlabel("Predicted labels", fontsize =15)
    ax.set_xticklabels(['']+class_labels)
    ax.set_ylabel("Observed labels", fontsize=15)
    ax.set_yticklabels(list(class_labels), rotation = 0)
    plt.show()

    return cm


def plot_roc(y_obs, y_pred):
    """
    Plot ROC curve to evaluate classification.
    Arguments:
        y_obs: Observed values for y
        y_pred: Predicted values for y as probabilities
        ax: The `Axes` object to plot on
    Returns:
        A matplotlib `Axes` object.
    """
    
    fig, ax = plt.subplots(1, 1)

    fpr, tpr, _ = roc_curve(y_obs, y_pred)
    auc_score = auc(fpr, tpr)

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='baseline')
    ax.plot(fpr, tpr, color='red', lw=2, label=f'AUC: {auc_score:.2}')

    ax.legend()
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.show()

    return ax



def plot_multiclass_roc(y_obs, y_pred):
    """
    Plot ROC curve to evaluate classification.
    Arguments:
        y_obs: Observed values for y
        y_pred: Predicted values for y as probabilities
    Returns:
        A matplotlib `Axes` object.
    """
    
    fig, ax = plt.subplots(1, 1)

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='baseline')
    class_labels = np.sort(y_obs.unique())
    
    for i, class_label in enumerate(class_labels):
        actuals = np.where(y_obs == class_label, 1, 0)
        predicted_probabilities = y_pred[:,i]

        fpr, tpr, _ = roc_curve(actuals, predicted_probabilities)
        auc_score = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"""class {class_label}; AUC: {auc_score:.2}""")

    ax.legend()
    ax.set_title('Multiclass ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.show()

    return ax


def plot_pca(data, labels):
    """
    Scatter plot of data along first two principle components
    Arguments:
        data: data for PCA analysis
        labels: labels for the data
    """

    pca = PCA(2)
    projected = pca.fit_transform(data)

    plt.scatter(projected[:, 0], projected[:, 1],
            c=labels, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', np.unique(labels).shape[0]))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()


def plot_tf_model_history(history):
    """
    Plot tensorflow model training history
    Arguments:
        history: TensorFlow model History object
    """

    epochs = range(len(history.history['loss']))

    # summarize history for accuracy
    plt.plot(epochs, history.history['accuracy'])
    plt.plot(epochs, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_target_value_counts(target_column, orientation="vertical"):
    """
    Make bar plot of target distribution/count
    Arguments:
        target_column: target column of pandas dataframe
    """
    if orientation != "vertical":
        target_column.value_counts().sort_index(ascending=False).plot(kind='barh')
    else:
        target_column.value_counts().sort_index(ascending=False).plot(kind='bar')


