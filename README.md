
# SunWise

SunWise is an application used to predict estimation time of skin resistance to UV exposure. Skin types are divided into 6 types based on the research of a renowned dermatologist, Fitz Patrick. The FastAI model will detect the image by giving the output of the skin type and the Decision Tree Model will give the output of the estimated resistance time based on skin type and UV light. 

## Model

 - model_fastai.pkl: Model built with FastAI and Resnet18 for Image detection.
 - decision_tree_model.pkl : Decision Tree model used to predict the estimated time of skin resistance to UV light.

## Deployment

This application is deployed using Cloud Run and uses 2 GigaByte/2Gi memory


