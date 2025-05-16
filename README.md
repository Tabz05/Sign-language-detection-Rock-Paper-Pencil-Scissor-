To run the web app, run the following command:
python app.py

To train a custom rock-paper-pencil-scissor real-time classification model:

1. Collect images. Run the following command:
python collect_imgs.py

2. Then, create the dataset by running the following command:
python create_dataset.py

3. Train the random forest model by running the following command:
python train_classifier.py

4. Run inference by running the following command:
python inference_classifier.py

