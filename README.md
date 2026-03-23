## Deep Learning for Satellite-Based Wildfire Detection
**Project Title:** Deep Learning for Satellite-Based Wildfire Detection

**Project Category (in the form of Method + Application, e.g. CNN for Face Recognition)**:
Convolutional Neural Networks (CNNs) for Wildfire Detection

**Names with UVA IDs:** Isaac Tabor (zds3st), Michael Dunlap (upw4ys), Jarrett Markman (gzt8na)

**Proposal:**

**1. Motivation: What problem are we tackling? Is this an application or a theoretical result?**
The goal of our project is to help address the problem of wildfire detection by building a deep learning
model capable of identifying wildfire-affected regions from satellite imagery, by classifying satellite
image patches as either wildfire or no wildfire. This is primarily an application result.
According to NASA, extreme wildfire activity has more than doubled worldwide, and has been amplified
due to Earth’s warming climate, particularly in northern and temperate forests, and satellite data is rich in
helping detect and track them, by helping communities and land managers prepare for and respond to
fires and understand this growing risk. NASA’s Terra and Aqua satellites detect active wildfires twice
each day. In 2023, Canada’s warmest and driest conditions since 1980 stoked extreme fires that lasted for
five months. NASA researchers found that these fires released about 640 million metric tons of carbon.
Isaac, being from Minnesota, has also experienced many summer days in Minnesota with unhealthy air
quality due to Canadian wildfires.
Training a CNN to classify image patches as “fire” or “no fire” could enhance early detection of wildfires,
that could be timelier and more scalable for environmental management and disaster response than
manually analyzing large volumes of satellite imagery.

**2. Dataset: URL + Description of the dataset with some basic stats**

We are using the Wildfire Prediction Dataset (Satellite Images) from Kaggle.

*Kaggle URL:* https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
The creator generated these images by extracting satellite imagery using the Map Box API at latitude and
longitude coordinates of wildfire events (>0.01 acres burned) from the Canadian government wildfire
dataset. All images are colored. Original wildfire records come from the Government of Canada Open

*Data Portal URL:* https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003

Description with some basic stats:

Dataset contains satellite images (350px x 350 px) in 2 classes: Wildfire (22710 images) and No wildfire
(20140 images). File names are coordinates, so a stretch goal could be to aggregate predicted probabilities
and map it. The data was divided into Train (~70%), Test (~15%), Validation (~15%).

**3. Related Work: At least 2 examples of prior methodology on the topic is a valuable addition**

Spiller, D., Carbone, A., Amici, S., Thangavel, K., Sabatini, R., & Laneve, G. (2023). Wildfire Detection
Using Convolutional Neural Networks and PRISMA Hyperspectral Imagery: A Spatial-Spectral Analysis.
Remote Sensing, 15(19), 4855. https://doi.org/10.3390/rs15194855

• This research paper examined the generalization capability of four neural network models: the
fully connected (FC), one-dimensional (1D) CNN, two-dimensional (2D) CNN, and threedimensional (3D) CNN model
• They used training data from Australia and Sicily and evaluated the performances of the trained
model on test dataset from Oregon.
• Their results were that their FC architecture generalized best, while their 3D CNN model offered
“more refined and less distorted classifications”
• However, they had a persistent issue with false fire detection and confusion between smoke and
shadows
M. S. Khan, R. Patil and S. Ali Haider, "Application of Convolutional Neural Networks For Wild Fire
Detection," 2020 SoutheastCon, Raleigh, NC, USA, 2020, pp. 1-5, doi:
10.1109/SoutheastCon44009.2020.9368283.
• These results were very promising as a CNN with one convolution layer, 64 classification nodes,
and 2 fully connected layers gave the best accuracy

**4. Technical Plan: What are the inputs and outputs of our task? Which deep learning models
and loss functions do we plan to use?**

The inputs to our tasks are satellite images split into test, train, and validation sets containing images
labeled either 0 for no wildfire or 1 for wildfire. This is determined based on a threshold of 0.01 acres
burned. The output for our task will therefore be a model that will predict within the test set either
wildfire or no wildfire. In this sense, we hope to create a model that can look at an image and classify it
into one of these two groups. The deep learning model we intend to use is a CNN, because of the one-toone basis of our dataset and task. In our plan, we believe that false positives and false negatives should be weighed equally, with maybe a slight preference towards giving false positives (i.e. model predicting
wildfire when it doesn’t meet the threshold). Therefore, we intend to use BCEWithLogitsLoss as our loss
function. This is designed for binary classification and includes a parameter that we can adjust to define
which incorrect decision we prefer (PyTorch Documentation).

**5. Evaluation Plan: What experiments are we planning to run? How do we plan to evaluate our ML algorithm?**

Our main experiment is to measure how well we can predict wildfires or not, utilizing train/validation/test
splits to maximize metrics such as accuracy (as well as precision, recall, etc.). Utilizing our results, we
plan to visualize model predictions across Canada to identify what areas are more at risk, and how these
wildfires are coming in based on the geography and ultimately use that to help prevent and protect the
country and areas from wildfires. Lastly, we plan to compare and expand our CNN model, architecture,
and results based on prior works, such as Spiller et al. (2023) and Khan et al. (2020), to serve as a
benchmark for our model’s performance.

**6. References**

• https://science.nasa.gov/earth/explore/wildfires-and-climate-change/
• https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
