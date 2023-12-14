﻿Deep Learning Project (INFOMPRDL)

Dr. Siamak Mehrkanoon

Dec 2023

1  Magnetoencephalography (MEG) data

MEG data comes from a neuroimaging technique that allows to scan the brain's magnetic eld. Multiple sensors (eg magnetometers) are placed on the human scalp and their recordings can be of major importance in neuroscience research. One can for instance infer from brain data the state of a patient that has mental disorders [\[1\].](#_page3_x133.77_y301.56)

![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.001.png)![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.002.png)

Figure 1: MEG data from a subject. The unit is in Tesla and the order of magnitude is in fT (femtoTesla = 10e-15 Tesla).

2  Data access and reading

You can download the data in the following link (password 123): [dataset](https://surfdrive.surf.nl/files/index.php/s/3bDWFzLx3smTNTn)

Once downloaded and uncompressed, you should end up with 2 folders : \Intra" and \Cross". The folder \Intra" contains 2 folders : train and test. The folder \Cross" contains 4 folders: train, test1, test2, and test3.

1. Reading of the data

The les contained in each of those folders have the \h5" extension. In order to read them, you need to use the h5py library ( that you can install using \pip install h5py" if you don't have it already ). This type of les can contain datasets identied by a name. For simplicity, each le contains only 1 dataset. The following code snippet can read the le "Intra=train=rest~~ 105923~~ 1:h5":

import h5py

def get dataset name ( file ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.003.png)name with ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.004.png)dir ):![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.005.png)![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.006.png)![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.007.png)

filename![ref1]without dir = file ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.009.png)name with ![ref1]dir . split ( '/ ' )[  1] ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.010.png)![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.011.png)temp = filename ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.012.png)without dir . split ( ' ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.013.png)' )[:  1]![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.014.png)

dataset name = " ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.015.png)" . join (temp)![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.016.png)

return dataset name![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.017.png)

filename path="Intra/train/rest 105923 1 .h5"![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.018.png)![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.019.png)![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.020.png)

with h5py . File ( filename path , ' r ' ) as f :![ref2]

dataset name = get dataset ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.022.png)name ( filename path ) ![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.023.png)![ref2]![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.024.png)matrix = f . get (dataset name ) [ ( ) ]![](Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.025.png)

print (type (matrix ))

print (matrix . shape)

After executing the code above, "matrix" variable will be a numpy array with shape 248 x 35624

2. Explanation of the les

The les have the following format: \taskType~~ subjectIdentier~~ number.h5" where taskType can be rest, task~~ motor, task~~ story~~ math, and task~~ working~~ memory. In practice, these tasks correspond to the activities performed by the subjects:

- Resting Task: Recording the subjects' brain while in a relaxed resting state.
- Math & Story Task : Subject performs mental calculation and language processing task.
- Working Memory task: Subject performs a memorization task.
- Motor Task : Subject performs a motor task, typically moving ngers or feets.

The subject identier is made of 6 numbers, and the number at the end cor- responds to a chunk part. This number has no particular meaning (splitted les are easier to handle in terms of memory management). The folder \In- tra" contains the les of 1 subject only. In the folder \Cross", 2 subjects are contained in the train folder while the 3 test folders contain dierent subjects from the ones contained in the train folder. As seen in the section above, each

le is represented by a matrix of shape 248 x 35624. The number of rows, 248, corresponds to the number of magnetometer sensors placed on the human scalp. The number of columns, 35624, corresponds to the time steps of a recording.

3  Investigation and questions

In brain decoding, 2 types of classications are performed. The rst one is intra-subject classication, where deep learning is used to train and test models using the same subject(s). The second type, called cross-subject classication, happens when we train a model with a set of subjects, but test the model on new, unseen subjects. In this assignment, you are asked to perform both intra- subject and cross-subject classication. The goal will be to accurately classify whether the subject is in one of the following states: rest,

math, memory, motor.

Tasks:

- (a) Choose a suitable deep learning model for the involved classication tasks. Justify your choice.
- (b) Compare the accuracy of the 2 types of classication, i.e. intra-subject and cross subject data using your model. Explain your results.
- (c) Explain the choices of hyper-parameters of your model architecture and analyze their inuence on the results (for both 2 types of classication). How they are selected?
- (d) If there is a signicant dierence in training and testing accuracies, what could be a possible reason? What are the alternative models or approaches you would select? Select one and implement to further improve your results. Justify your choice.

3\.1 Hints

1. Data preprocessing

As you have seen in gure 1, the order of magnitude of this data is 10e-15, which might not be adapted for deep learning tasks. A common approach to tackle this problem is to do min-max scaling, making all the data scale to values between 0 and 1. Another common approach is Z-score normalization. More specically, a time wise scaling/normalization is more suitable.

2. Data downsampling

The machine that made the recording of this data used a sample rate of 2034 Hz, meaning that every second corresponds to 2034 samples, or data points. Therefore every le corresponds to a duration of approximately 17.5 seconds. A common approach in neuroscience research is to consider that not every sam- ples are signicant, and to perform downsampling. A major advantage of this technique is that it makes deep learning training faster, while not necessarily having a negative impact on the accuracy.

3. Memory management during training

Since the train folder of the "Cross" directory contains 64 les, it might be dicult to load everything in the memory for the training. A simple workaround is to use a loop. For instance, the rst iteration would load a small subpart of all the les (eg: 8 les), to t the model to this data. The second iteration would load the next subpart (the next 8 les), to t it etc ...

References

<a name="_page3_x133.77_y301.56"></a>[1] Stefan Kloppel, Cynthia M Stonnington, Carlton Chu, Bogdan Draganski, Rachael I Scahill, Jonathan D Rohrer, Nick C Fox, Cliord R Jack Jr, John Ashburner, and Richard SJ Frackowiak. Automatic classication of mr scans in alzheimer's disease. Brain, 131(3):681{689, 2008.
4

[ref1]: Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.008.png
[ref2]: Aspose.Words.ddc7060c-2bac-46e3-ab71-6f09f3445078.021.png
