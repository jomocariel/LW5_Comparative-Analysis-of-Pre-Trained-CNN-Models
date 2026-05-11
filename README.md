# LW5_Comparative-Analysis-of-Pre-Trained-CNN-Models

https://colab.research.google.com/drive/1TP5nKoZJcBHh_-47Mai6rR09mQH8UAaX#scrollTo=_NC8439aMxeQ

GUIDE QUESTIONS (FINAL REFLECTION)

A. Model Performance

1. Which pre-trained model achieved the highest accuracy? Why?
2. Which model had the lowest performance? What could be the reason?
3. How did loss values compare across models?

B. Evaluation Metrics

4. Why is accuracy not enough to evaluate a model?
5. Which model had the best F1-score? What does it indicate?
6. How did Precision and Recall differ across models?

C. Confusion Matrix Analysis

7. Which classes were frequently misclassified?
8. What patterns did you observe in the confusion matrix?

D. ROC and AUC

9. Which model had the highest AUC score?
10. What does AUC tell us about model performance?

E. Explainability (Grad-CAM)

11. What did Grad-CAM reveal about model decision-making?
12. Did the model focus on relevant image regions?
13. Which model produced the most meaningful heatmaps?
F. Model Comparison & Improvement
14. Which model would you recommend for deployment? Why?
15. How can you further improve your best-performing model?

G. Real-World Application

16. How can your model be applied in real-world scenarios?
17. What are the risks of deploying an inaccurate model?
18. How can this system be integrated into a mobile/web app?


# A. Model Performance

1. Which pre-trained model achieved the highest accuracy? Why?

The result table shows that the test accuracy for ResNet50 was the highest, at **80.18%**. This is because ResNet50's residual (skip) connections enable it to be trained deeper without suffering from the vanishing gradient problem and be able to learn richer, more discriminative features from your custom image dataset. The layer depth (50 layers vs. lightweight MobileNetV2) allows it to have a higher representational capacity when fine-tuned with transfer learning from your 20+ class data set.

2. Which model had the lowest performance? What could be the reason?

The test accuracy of MobileNetV2 is the lowest at **75.04%**. In order to prioritize speed and efficiency, MobileNetV2 was designed for mobile/edge deployment, employing depthwise separable convolutions, which were developed for this purpose. This makes it less powerful to distinguish between similarly-looking classes in a complex 20+ class dataset, particularly with just 10 epochs of training and frozen base layers.

3. How did loss values compare across models?

| Model | Train Loss | Test Loss |
|---|---|---|
| ResNet50 | 0.8009 | 0.7352 |
| EfficientNetB0 | 1.0334 | 0.8875 |
| MobileNetV2 | 1.0592 | 0.9571 |

ResNet50 showed the minimum loss in both training and testing phase that suggests in better convergence and generalization. The loss values were the largest for MobileNetV2. However, it is important to note that, as there is still no serious overfitting, for all three models the test loss is lower than or close to the training loss, presumably because the base model weights were not updated (trainable=False) during training.

# B. Evaluation Metrics

4. Why is accuracy not enough to evaluate a model?

In the case of imbalanced classes, accuracy may be misleading. For instance, a model that always predicts the majority class may be accurate overall, and yet tell lousy stories on the minority classes. Precision, Recall, F1-score, and AUC provide a comprehensive description — they show if the model performance is the same for all classes, or if it is a "cheating" model, that is, if it shows a higher accuracy in the dominant classes. It is particularly important in your lab with 20+ classes.

5. Which model had the best F1-score? What does it indicate?

**ResNet50** achieved the best macro F1-score at **0.7938**. The F1 score is the harmonic mean of Precision and Recall, and a higher F1 score means high precision (not many false positives) and high recall (not many false negatives) among classes. ResNet50 is effective on all the 20+ categories, not just the easy ones, as you can see in the result.

6. How did Precision and Recall differ across models?
   
| Model | Precision | Recall |
|---|---|---|
| ResNet50 | 0.7988 | 0.7946 |
| EfficientNetB0 | 0.7800 | 0.7673 |
| MobileNetV2 | 0.7461 | 0.7444 |

The Precision/Recall values for all three models were very similar, indicating no extreme bias towards false positives or false negatives. ResNet50 was the top scorer on both counts. The results showed that MobileNetV2 performed poorly on both measures. The difference between Precision and Recall between models was not too great to consider that no model over-predicted or under-predicted any class on average.

# C. Confusion Matrix Analysis

7. Which classes were frequently misclassified?

The **5 weakest classes per model** were selected based on the sort of per-class recall from the confusion matrix in the lab. As you can see in all three models, the classes that are most likely to be confused have similarities in their appearance in the image, whether it's a texture of the food item, a color of an animal, or anything else. The specific classes are dependent on the dataset (in this case stored in class_names). A decrease in the number of images given to a class also results in a decrease in per-class recall.

8. What patterns did you observe in the confusion matrix?

There are several common trends in a classification confusion matrix of this sort with 20 classes:

- Diagonal dominance: Most predictions are on the diagonal (correct classifications) as would be expected as there was 75-80% overall accuracy.
Off-diagonal clusters: Misclassifications tend to be grouped within similar looking classes, indicating that the model is mixing classes with similar texture, colour or shape features.
- Misclassifications are fairly uniform across all three models, suggesting that the difficulty is due not to any one model's shortcomings, but to the difficulty of the data itself (either visual ambiguity or lack of data samples).
Asymmetric errors: some classes are more over-predicted than others (a class which is over-predicted will be represented as a column), which means that the most common features of these classes tend to be more prevalent than the features of similar classes.

# D. ROC and AUC

9. Which model had the highest AUC score?

ResNet50 performed best with an Overall AUC score of 0.9837, while EfficientNetB0 and MobileNetV2 had scores of 0.9787 and 0.9729 respectively.

10. What does AUC tell us about model performance?
    
AUC (Area Under the ROC Curve) indicates how well a model can separate the classes, no matter what threshold is used for classification. The value of 1.0 represents complete separation and 0.5 represents random guessing. They all performed with greater than 0.97, which is excellent (very good), although the absolute accuracy is moderate (~75-80%). The value of AUC is especially important in multi-class settings where it measures the model's ability to discriminate between all class boundaries (one-vs-rest curves).

# E. Explainability (Grad-CAM)

11. What did Grad-CAM reveal about model decision-making?
    
The first step of Grad-CAM is to calculate the gradient of the output score for a given class for the feature maps generated by the last convolutional layer of the model. This shows the spatial areas that contributed most to the prediction. Grad-CAM was also shown to detect the most relevant parts of an image for the model in the lab, the central parts in the image being semantically relevant (such as the main object).

12. Did the model focus on relevant image regions?
    
Typically, yes, especially if the sample was correctly classified; the lab is looking for images that are classified correctly by all three models. The Grad-CAM heatmaps usually focus on the object of interest rather than the background, when the model is confident and correct. Unfortunately, for misclassified images, the heatmap will often also highlight activation in the background textures as well as in the wrong edges, which is a sign of the model's failure.

13. Which model produced the most meaningful heatmaps?
    
The heat maps of ResNet50 were most focused and meaningful, partly because they were generated by a deeper network than the other models (more Conv2D layers, the latter one captures highly abstract, class discriminative features). EfficientNetB0 is also often found to yield clear heatmaps due to its compound scaling of depth, width and resolution. Lightweight depthwise separable convolutions, which are used in MobileNetV2 to capture more general features, may make the heatmaps in MobileNetV2 less localized.

# F. Model Comparison & Improvement

14. Which model would you recommend for deployment? Why?
    
ResNet50 is the best among all models in terms of all the metrics — test accuracy (80.18%), F1 score (0.7938%), AUC (0.9837%) and test loss (0.7352%). It generalizes well and has the least misclassification in its confusion matrix. For deployment on a resource constrained device (mobile/embedded), ResNet50 is the obvious winner.

15. How can you further improve your best-performing model?

There are several strategies that can get ResNet50 to perform better:

Fine tuning: Unfreeze the last 20-30 layers of ResNet50 base (or even all the layers if you want) and train with a very small learning rate (e.g., 1e-5) — this allows pre-trained weights to be adjusted to your own data.
Data augmentation: Random flip, rotation, zoom, brightness changes, and cutmix/mixup augmentation.
Use cosine decay or ReduceLROnPlateau to increase epochs when learning rate should be lower to let model converge properly.
Class balancing: Oversampling or class weights in the loss function - that is if some classes have fewer images.
Greater dense head: Try using a wider dense head, such as 256 or 512 units, or use an additional dense layer.
Ensemble methods: Take an average of the predictions of all three models, to lower variance and increase prediction accuracy beyond individual models.

# G. Real-World Application

16. How can your model be applied in real-world scenarios?
With your dataset's 20+ classes, your model can enable:

Food recognition applications, such as calorie counter or recipe applications based on images of food that are being photographed
Monitoring of wildlife/biodiversity (using camera trap images to identify animal species)
Knowing about the plant disease types from field photos (plant disease detection)
Retail product recognition (identifying products for shops in stocks)
Triage in medical imaging (classification of skin conditions/pathology slides)
This transfer learning approach makes the model particularly suitable for specialized domains, where the amount of labelled data is limited.

17. What are the risks of deploying an inaccurate model?

Potential false positive/negative with real-world consequences: When using a medical or safety-critical system, a classification error may result in misdiagnosis, missed hazards or erroneous treatment.
Bias amplification: If your training data is biased towards certain classes or demographics, your model will be biased towards those underrepresented classes or demographics.
Loss of user trust: When consumer apps include repeated errors, users lose trust, and adoption is hindered.
Security risk: Critical systems could be vulnerable to adversarial vulnerability; CNN models can be tricked by small changes in the pixels.
Data drift: Data in the real world can change over time, without monitoring (due to different lighting conditions, angles, camera quality, etc.).

18. How can this system be integrated into a mobile/web app?

Web deployment: Export the trained ResNet50 model with model.save(), transform it into a model in the TensorFlow.js format with tensorflowjs_converter, and deploy the model directly to the client-side of an app without requiring a server to do inference.
Mobile deployment: Prototype to tflite for Android/ iOS deployment. The models run efficiently on-device with low latency.
The backend is a REST API with a Flask or FastAPI server that includes a /predict endpoint that will receive uploaded images, and will be linked to any front-end (React, Vue, etc.) or mobile app using HTTP requests.
Cloud deployment: Deploy the model on Google Cloud AI Platform, AWS SageMaker or Azure ML to provide scalable, managed inference with auto-scaling.
Export to TensorFlow Lite or ONNX and deploy to edge devices, such as Raspberry Pi or NVIDIA Jetson, for real-time applications (e.g., embedded cameras).
