import matplotlib.pyplot as plt


def re_order(names):
    """
    Re-orders the names of the files to match the order of the tasks

    Parameters
    ----------
    names : list The list of names to re-order

    Returns
    -------
    list The re-ordered list of names

    """
    order = ["rest", "task_motor", "task_story_math", "task_working_memory"]
    new_names = []
    while len(names) > 0:
        for name in order:
            for n in names:
                if name in n:
                    new_names.append(n)
                    names.remove(n)
                    break

    return new_names


def plot(history, save=False, show=True, name=""):
    """
    Plots the history of a model

    Parameters
    ----------
    history : dict The history of the model
    save : bool, optional (default=False) Whether to save the plot
    show : bool, optional (default=True) Whether to show the plot
    name : str, optional (default="") The name of the plot

    Returns
    -------
    None

    """
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    if save:
        plt.savefig(f"{name}_loss.png")
    if show:
        plt.show()

    plt.close()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    if save:
        plt.savefig(f"{name}_accuracy.png")
    if show:
        plt.show()


def show_evaluation(evaluation_report):
    """
    Prints the evaluation report

    Parameters
    ----------
    evaluation_report : list The evaluation report

    Returns
    -------
    None

    """

    print("----" * 30)
    print(f"Loss: {evaluation_report[0]}")
    print(f"Accuracy: {evaluation_report[1]}")
    print("----" * 30)
