import matplotlib.pyplot as plt


def plot_loss_history(train_loss_history, eval_loss_history, save_path):
    """
    Plots the loss history
    """
    plt.figure()
    ep = range(len(train_loss_history))

    plt.plot(ep, train_loss_history, "-b", label="training")
    # plt.plot(ep, eval_loss_history, "-r", label="validation")
    plt.title("Loss history")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_path)
    plt.show()
