from enum import Enum
import h5py
import numpy as np
import os
try:
    import pretty_errors
except ImportError:
    pass

DATA_PREFIX = "data/Final Project data"


class Goal(Enum):
    """
    Goal enum class

    Purpose: Keep the labeling consistent and easy to read/change

    Attributes:
        REST (list): rest goal
        MATH_AND_STORY (list): math and story goal
        WORKING_MEMORY (list): working memory goal
        MOTOR (list): motor goal 

    Examples:
        >>> Goal.REST.value
        >>> Output: [1, 0, 0, 0]
    """
    REST = [1, 0, 0, 0]
    MATH_AND_STORY = [0, 1, 0, 0]
    WORKING_MEMORY = [0, 0, 1, 0]
    MOTOR = [0, 0, 0, 1]


class DataFile:
    """
    Data file class

    Purpose: Represent a data file

    Attributes:
        subject_id (str): subject id
        chunk_id (str): chunk id
        goal (Goal): goal
        matrix (np.ndarray): matrix

    Methods:
        __init__(self, filename): constructor
        __str__(self): to string
        get_matrix(self): get matrix
        get_goals(self): get goal value
        get_goal_name(self): get goal name
        remove(self): remove the object from memory
        downsample(self, rate): downsample the matrix, select every rate-th row

    Examples:
        >>> DataFile("rest_105923_1.h5")
        >>> Output: Subject: 105923, Chunk: 1, Goal: Goal.REST, Goal ID: 1, Matrix: (248, 35624)
    """

    def __init__(self, filename, root_dir, scaler=None, downsample_rate=1):
        """
        Constructor

        Args:
            filename (str): filename
            root_dir (str): root directory
            scaler (StandardScaler): scaler (default: None)
            downsample_rate (int): downsample rate (default: 1) -> Use every row

        """
        items = filename.split('_')
        self.subject_id = items[-2]
        self.chunk_id = items[-1].split('.')[0]
        label = ("_").join(items[0:len(items)-2])
        self.goal = decode_task_to_goal(label)
        self.matrix = get_dataset_values(root_dir + filename).T
        self.matrix = self.downsample(downsample_rate)
        if scaler is not None:
            scaler.partial_fit(self.matrix)

    def __str__(self):
        """
        To string

        Returns:
            str: string representation of the object

        Examples:
            >>> DataFile("rest_105923_1.h5", root_dir="data/Final Project data/Intra/train/")
            >>> Subject: 105923, Chunk: 1, Goal: Goal.REST, Goal Value: [1, 0, 0, 0], Matrix: (35624, 248)
        """
        return f"Subject: {self.subject_id}, Chunk: {self.chunk_id}, Goal: {self.goal}, Goal Value: {self.goal.value}, Matrix: {self.matrix.shape}"

    def get_matrix(self):
        """
        Get the values of the dataset file

        Returns:
            np.ndarray: matrix

        """
        return self.matrix

    def get_goal(self):
        """
        Get the goal/label value of the dataset file

        Returns:
            Goal: goal value
        """
        return self.goal.value

    def get_goal_name(self):
        """
        Get the goal/label name of the dataset file

        Returns:
            Goal: goal name
        """
        return self.goal.name

    def remove(self):
        """
        Remove the object from memory

        """
        del self

    def downsample(self, rate):
        """
        Downsample the matrix, select every rate-th row

        Args:
            rate (int): rate

        Returns:
            np.ndarray: downsampled matrix

        """
        temp = [line[::rate] for line in self.matrix.T]

        return np.array(temp).T


def decode_task_to_goal(string):
    """
    Decode task to goal

    Args:
        string (str): task string

    Returns:
        goal (Goal): goal

    Examples:
        >>> decode_task_to_goal("rest")
        >>> Output: Goal.REST
    """
    if string == "rest":
        return Goal.REST
    elif string == "task_story_math":
        return Goal.MATH_AND_STORY
    elif string == "task_working_memory":
        return Goal.WORKING_MEMORY
    elif string == "task_motor":
        return Goal.MOTOR
    else:
        raise ValueError("Invalid task type")


def get_dataset_name(file_name_with_dir):
    """
    Get dataset name from file name with dir 

    Args:
        file_name_with_dir (str): file name with dir

    Returns:
        dataset_name (str): dataset name

    Examples:
        >>> get_dataset_name("data/Final Project data/Intra/train/rest_105923_1.h5")
        >>> Output: rest_105923
    """
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]

    dataset_name = "_".join(temp)
    return dataset_name


def get_dataset_values(file_name_with_dir):
    """
    Get dataset values from file name with dir 

    Args:
        file_name_with_dir (str): file name with dir

    Returns:
        matrix (np.ndarray): dataset values

    Examples:
        >>> get_dataset_values("data/Final Project data/Intra/train/rest_105923_1.h5")
        >>> Output: np.ndarray
    """

    dataset_name = get_dataset_name(file_name_with_dir)

    with h5py.File(file_name_with_dir, 'r') as f:
        matrix = f.get(dataset_name)[()]

    return matrix


def print_dataset_help():
    """
    Print dataset help, information about the file structure and what dataset contain.

    """

    INDICES = 40

    print("----" * INDICES)

    print("Dataset information:")
    print("----" * INDICES)

    print('''    C:.
    └───data
        ├───Final Project data
        │   ├───Cross
        │   │   ├───test1
        │   │   ├───test2
        │   │   ├───test3
        │   │   └───train
        │   └───Intra
        │       ├───test
        │       └───train
        └───__MACOSX
            └───Final Project data
                ├───Cross
                │   ├───test1
                │   ├───test2
                │   ├───test3
                │   └───train
                └───Intra
                    ├───test
                    └───train''')
    print("----" * INDICES)

    print('The  files  have  the  following  format: “taskTypesubjectIdentifiernumber.h5” where taskType can be rest, taskmotor, ')
    print('taskstorymath, and taskworkingmemory. In practice, these tasks correspond to the activities performed by the subjects:')
    print('\n\t•Resting Task:  Recording the subjects’ brain while in a relaxed restingstate.')
    print('\t•Math & Story Task: Subject performs mental calculation and languageprocessing task.')
    print('\t•Working Memory task:  Subject performs a memorization task.')
    print('\t•Motor Task:  Subject performs a motor task, typically moving fingersor feets\n')
    print("----" * INDICES)

    print('The subject identifier is made of 6 numbers,  and the number at the end corresponds to a chunk  part.')
    print('This number has no particular meaning (splitted files) are easier to handle in terms of memory management). The folder “Intra” contains the files of 1 subject only.')
    print('In the folder “Cross”, 2 subjects are contained in the train folder while the 3 test folders contain different subjects')
    print('from the ones contained in the train folder.  As seen in the section above, each file is represented by a matrix of shape 248 x 35624.')
    print('The number of rows, 248,corresponds to the number of magnetometer sensors placed on the human scalp. The number of columns, 35624, corresponds to the time steps of a recording.')
    print("----" * INDICES)

    print('In order to better understand and visualize the data you can use https://myhdf5.hdfgroup.org/')
    print('Each file contains a part of the same dataset, for example rest_105923.')
    print('You can imagine the .h5 file as a dictionary and all same-named files share the same key value.')
    print("----" * INDICES)


def get_all_filenames(directory):
    """
    Get all filenames from directory

    Args:
        directory (str): directory path

    Returns:
        all_files (list): list of all filenames in a directory

    Examples:
        >>> get_all_filenames("data/Final Project data/Intra/train")
        >>> ['rest_105923_1.h5', 'rest_105923_2.h5', 'rest_105923_3.h5', 'rest_105923_4.h5', 'rest_105923_5.h5', 'rest_105923_6.h5', 'rest_105923_7.h5', 'rest_105923_8.h5', 'task_motor_105923_1.h5', 'task_motor_105923_2.h5', 'task_motor_105923_3.h5', 'task_motor_105923_4.h5', 'task_motor_105923_5.h5', 'task_motor_105923_6.h5', 'task_motor_105923_7.h5', 'task_motor_105923_8.h5', 'task_story_math_105923_1.h5', 'task_story_math_105923_2.h5', 'task_story_math_105923_3.h5', 'task_story_math_105923_4.h5', 'task_story_math_105923_5.h5', 'task_story_math_105923_6.h5', 'task_story_math_105923_7.h5', 'task_story_math_105923_8.h5', 'task_working_memory_105923_1.h5', 'task_working_memory_105923_2.h5', 'task_working_memory_105923_3.h5', 'task_working_memory_105923_4.h5', 'task_working_memory_105923_5.h5', 'task_working_memory_105923_6.h5', 'task_working_memory_105923_7.h5', 'task_working_memory_105923_8.h5']   
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
