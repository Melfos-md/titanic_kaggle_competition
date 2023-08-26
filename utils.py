import matplotlib.pyplot as plt

def plot_loss_metrics(history, to_file=False):
    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    accuracy_values = history.history['accuracy']
    val_accuracy_values = history.history['val_accuracy']

    epochs = range(1, len(loss_values) + 1)

    # Tracer la perte
    plt.figure()
    plt.plot(epochs, loss_values, 'r', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if to_file:
        plt.savefig('loss.png')
    else:
        plt.show()

    # Tracer la précision
    plt.figure()
    plt.plot(epochs, accuracy_values, 'r', label='Training accuracy')
    plt.plot(epochs, val_accuracy_values, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if to_file:
        plt.savefig('metrics.png')
    else:
        plt.show()