# Computer Vision ML OpenHack Resource List

## Setting up your WorkSpace

**Ubuntu Data Science Virtual Machine (DSVM)**

* Introduction to the Azure Data Science Virtual Machine [Ref](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/overview)
* Create a Linux Data Science Virtual Machine (DSVM) and use JupyterHub to code with a team - [Video](https://www.youtube.com/watch?v=4b1G9pQC3KM) or [Doc](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/linux-dsvm-walkthrough#jupyterhub)

**Local Setup**

* Install Anaconda if you don't have it for your system [Installation information](https://docs.anaconda.com/anaconda/install/)
* Getting started with conda <a href="https://conda.io/docs/user-guide/getting-started.html" target="_blank">Doc</a>
* Creating and activating a conda environment <a href="https://conda.io/docs/user-guide/tasks/manage-environments.html" target="_blank">Ref</a>
* Connecting a Jupyter Notebook to a specific conda environment  <a href="http://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments" target="_blank">Ref</a>

## Custom Vision

* Custom Vision [Ref](https://customvision.ai)
* Machine Learning Demystified <a href="https://youtu.be/k-K3g4FKS_c" target="_blank">Video</a>
* Definitions of some common Machine Learning terms <a href="https://docs.microsoft.com/en-us/azure/machine-learning/studio/what-is-machine-learning#key-machine-learning-terms-and-concepts" target="_blank">Ref</a>
* Classification description <a href="https://docs.microsoft.com/en-us/azure/machine-learning/studio/data-science-for-beginners-the-5-questions-data-science-answers#question-1-is-this-a-or-b-uses-classification-algorithms" target="_blank">Ref</a>
* Overview diagram of the Machine Learning process <a href="https://blogs.msdn.microsoft.com/continuous_learning/2014/11/15/end-to-end-predictive-model-in-azureml-using-linear-regression/" target="_blank">Docs</a>

**Jupyter and Managing Packages**

* `jupyter` <a href="https://jupyter.readthedocs.io/en/latest/running.html" target="_blank">Ref</a>
* On using `conda` or `pip` to install Python packages <a href="https://conda.io/docs/user-guide/tasks/manage-pkgs.html" target="_blank">Ref</a>
* Jupyter notebook usage <a href="http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Notebook%20Basics.html" target="_blank">Ref</a>

**Calling the Prediction API**

* Requests is one of the most popular Python libraries to make API calls and is easy to use <a href="http://docs.python-requests.org/en/master/" target="_blank">Ref</a> with an example at <a href="https://github.com/michhar/python-jupyter-notebooks/blob/master/cognitive_services/Computer_Vision_API.ipynb" target="_blank">Code Sample</a>
* Alternatively, the Custom Vision Service gives an example of calling the service with Python's `urllib` library and Python3 - <a href="https://southcentralus.dev.cognitive.microsoft.com/docs/services/57982f59b5964e36841e22dfbfe78fc1/operations/5a3044f608fa5e06b890f164" target="_blank">Docs</a>

**Custom Vision Service**

* Custom Vision Service <a href="https://customvision.ai" target="_blank">Ref</a>
* Custom Vision Service Docs <a href="https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/home" target="_blank">Docs</a>
* Custom Vision Python SDK (Linux) <a href="https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/python-tutorial" target="_blank">Ref</a>

## Data Manipulation

* Is your data ready for data science? <a href="https://docs.microsoft.com/en-us/azure/machine-learning/studio/data-science-for-beginners-is-your-data-ready-for-data-science" target="_blank">Doc</a>

**Useful Packages**

* `matplotlib` on dealing with images (I/O, plotting) <a href="https://matplotlib.org/2.0.2/users/image_tutorial.html" target="_blank">Ref</a>
* `numpy` for image manipulation/processing/visualization <a href="http://www.scipy-lectures.org/advanced/image_processing/" target="_blank">Ref</a>
* `PIL` Image module for I/O and more <a href="http://pillow.readthedocs.io/en/4.2.x/reference/Image.html" target="_blank">Ref</a>
* `PIL` ImageOps module which has the ability to flip, rotate, equalize and perform other operations. <a href="http://pillow.readthedocs.io/en/4.2.x/reference/ImageOps.html" target="_blank">Ref</a>

**Concepts**

* Feature scaling (normalization) <a href="https://en.wikipedia.org/wiki/Feature_scaling" target="_blank">Ref</a>

**Code samples**

* Pixel intensity normalization example <a href="https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues" target="_blank">Ref</a>

## First Custom ML

* `scikit-learn` algorithm cheatsheet <a href="http://scikit-learn.org/stable/index.html" target="_blank">Ref</a>
* Non-parametric and parametric algorithm differences <a href="https://sebastianraschka.com/faq/docs/parametric_vs_nonparametric.html" target="_blank">Ref</a>
* `scikit-learn` Machine Learning guide with vocabulary <a href="http://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction" target="_blank">Ref</a>
* `scikit-learn` Supervised Learning <a href="http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html" target="_blank">Ref</a>
* `scikit-learn` General User Guide <a href="http://scikit-learn.org/stable/user_guide.html" target="_blank">Ref</a>

## Classification with Deep Learning

* What is a convolutional neural net <a href="https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/" target="_blank">Ref</a> or <a href="https://www.youtube.com/watch?v=FmpDIaiMIeA" target="_blank">Video</a>
* High level overview of Machine Learning and CNNs <a href="https://youtu.be/k-K3g4FKS_c" target="_blank">Video</a>

Deep Learning Frameworks

* Keras (an abstraction layer that uses TensorFlow or CNTK on the backend)
    * <a href="https://keras.io/" target="_blank">Docs</a> And <a href="https://github.com/fchollet/keras-resources" target="_blank">Tutorials</a>
* TensorFlow
    * <a href="https://www.tensorflow.org/" target="_blank">Docs</a> And <a href="https://www.tensorflow.org/tutorials/" target="_blank">Tutorials</a>
    * Suggested starting point is a CNN from a Tutorial with the Layers API
* CNTK
    * <a href="https://www.microsoft.com/en-us/cognitive-toolkit/" target="_blank">Docs</a> And <a href="https://cntk.ai/pythondocs/tutorials.html" target="_blank">Tutorials</a>
    * Suggested starting point is a CNN from a Tutorial with the Layer API

## Object Detection with Deep Learning

* Visual Object Tagging Tool `VoTT`. Works for TensorFlow and CNTK <a href="https://github.com/Microsoft/VoTT" target="_blank">Ref</a>
* When on Linux, the Tensorflow Object Detection API <a href="https://github.com/tensorflow/models/tree/master/research/object_detection" target="_blank">Ref</a>
* CNTK Documentation <a href="https://www.microsoft.com/en-us/cognitive-toolkit/" target="_blank">Ref</a>

## Deployment

* Overview of Azure ML model management <a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview" target="_blank">Doc</a>
* Deployment walkthrough <a href="https://michhar.github.io/deploy-with-azureml-cli-boldly/" target="_blank">Ref</a>

**More on Deployment**

* Microsoft Blog on deploying from Azure ML Workbench and the Azure ML CLI <a href="https://blogs.technet.microsoft.com/machinelearning/2017/09/25/deploying-machine-learning-models-using-azure-machine-learning/" target="_blank">Ref</a>
* Setting up with the Azure ML CLI for deployment 
<a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration" target="_blank">Doc</a>
* Non-CLI deployment methods (AML alternative) <a href="https://github.com/Azure/ACS-Deployment-Tutorial" target="_blank">Ref</a>

**Scoring File and Schema Creation References**

* Example of schema generation <a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#2-create-a-schemajson-file" target="_blank">Doc</a>
* Example of the scoring file showing a CNTK model and serializing an image as a `PANDAS` data type for input data to service <a href="https://github.com/Azure/MachineLearningSamples-ImageClassificationUsingCntk/blob/master/scripts/deploymain.py" target="_blank">Ref</a>
* Example of the scoring file showing a `scikit-learn` model and a `STANDARD` data type (json) for input data to service <a href="https://github.com/Azure/Machine-Learning-Operationalization/blob/master/samples/python/code/newsgroup/score.py" target="_blank">Ref</a>
* After creating a `run` and `init` methods as in the links above, plus a schema file, begin with "Register a model" found in this <a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#4-register-a-model">Doc</a>
  * Note one change required:  there must be, if using certain frameworks, a `pip` requirements file (use `-p` flag) when creating the manifest

**Docker**

* Docker Docs <a href="https://docs.docker.com/get-started/" target="_blank">Ref</a>

## General Resources

**Longer Courses**

* Coursera courses from Andrew Ng on [traditional](https://www.coursera.org/learn/machine-learning) and [deep learning](https://www.coursera.org/specializations/deep-learning)
* Microsoft Professional Program for [Data Science](https://academy.microsoft.com/en-us/professional-program/tracks/data-science/)
* Microsoft Professional Program for [Artificial Intelligence](https://academy.microsoft.com/en-us/professional-program/tracks/artificial-intelligence/)

**Videos**

* Deep Learning Simplified - no math, very basic intro series [Link](https://www.youtube.com/channel/UC9OeZkIwhzfv-_Cb7fCikLQ/videos)
* Channel 9 AI Show [Link](https://channel9.msdn.com/Shows/AI-Show)
* Brandon Rhoher's YouTube channel with ML/DL introductory videos [Link](https://www.youtube.com/user/BrandonRohrer/videos)

**Tutorials/Examples**

* Microsoft AI school [Link](https://aischool.microsoft.com/learning-paths)
* AI Developer on Azure resources [Link](https://learnanalytics.microsoft.com/learningpaths/developing-advanced-ai-applications)

**Books**

* Jake VanderPlas - The Data Science Handbook (Free on Azure Notebooks) [Link](https://notebooks.azure.com/jakevdp/libraries/pythondatasciencehandbook)

**Competitions**

* Kaggle has competitions and can be a source of datasets and interesting notebooks [Link](https://www.kaggle.com/)


## Links from Tech Talks

| Talk | Links |
| --- | --- |
| Scikit-Learn Liner Regression | [Notebooks](https://notebooks.azure.com/DaveVoyles/libraries/DVTestLib/tree/Open%20Hack%20Talks)
| Team Data Science Process | [Algorithm Cheatsheet](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-cheat-sheet), [Team Data Science Process Lifecycle](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/lifecycle), [TDSP Template](https://github.com/Azure/Azure-TDSP-ProjectTemplate), [Handrawn data science diagram](https://github.com/PythonWorkshop/intro-to-sklearn/blob/master/imgs/ml_process_by_micheleenharris.png) |

## Misc

* [Understanding what's going on in a CNN diagram](http://scs.ryerson.ca/~aharley/vis/conv/)
* [Great book for ML Beginner's on AI and Python](https://www.packtpub.com/big-data-and-business-intelligence/artificial-intelligence-python)