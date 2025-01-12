# QHQ-CS250-D2
Chapter 1	Introduction
1.1	Background
In today’s world, people listen to music anytime and anywhere using apps or media players. However, many music players cannot understand or match the emotions of their users. The QHQ Music Player project was created to solve this problem. QHQ Music Player makes listening more personal and enjoyable by recommending songs that fit the user’s mood.
1.2	Project Objectives
There are three primary systems used to build the player.
1.2.1	Emotion recognition system
  This system uses an expression recognition model (FER2013) to analyze the user's facial expressions. By processing an uploaded image or real-time camera input, the system identifies the user's emotional state. The recognized emotion is then used as input for recommending suitable playlists.
1.2.2	Recommended Playlist System
  The recommended playlist system analyzes the user's recognized emotion and provides personalized song suggestions that match their mood.
1.2.3	Feedback system
In addition to the "reduce recommendation " button available for songs being played, the feedback system also provides an option for users to share their thoughts about the entire application. This feature allows users to report issues, suggest improvements, or provide general feedback about the player. The feedback is sent directly to the administrator, ensuring continuous improvement and better user satisfaction. This functionality will be presented in the video.

1.3	Report Structure
This report has the following sections:
Chapter 2: System development
Chapter 3: Testing
Chapter 4: Display QHQ Music Player system





 
Chapter 2	System development
2.1	Emotion recognition system
  We used FER2013 (Facial Expression Recognition 2013),that a publicly available dataset for facial expression recognition in our development.The FER2013 project used convolutional neural network (CNN) to deal with facial expression recognition task. CNNs are a popular deep-learning approach utilized in a wide range of applications. 
  The model supports python-3.8.10, the prerequisite is to have the plugins keras, numpy, sklearn, pandas, opencv-python installed.
1.	FER2013 data Volume and format:
Number of samples: A total of 35,887 grayscale images.
Image size: Each image is 48*48 pixels.
Format: The data is stored in CSV file format and contains two main fields:
emotion: A label indicating the type of expression (0-5)
pixels: A string indicating the pixel values of the image separated by Spaces
2.	Expression Categories:
 FER2013 contains emotion categories:
0) Angry  1) Fear  2) Happy  3) Sad  4) Surprise  5) Neutral
 
3.	Data partitioning:
Training set: 28,709 images.
Validation set: 3,589 images.
Test set: 3,589 images.
This Model - 66.369% accuracy
 

2.1.1	Data preprocessing
  The preprocessing.py does the work to get the pixel data and labels and to save the data when it's done.
#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))
X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
y = pd.get_dummies(data['emotion']).as_matrix()

#storing them using numpy
np.save('fdataX', X)
np.save('flabels', y)
print("Preprocessing Done")
print("Number of Features: "+str(len(X[0])))
print("Number of Labels: "+ str(len(y[0])))
print("Number of examples in dataset:"+str(len(X)))
print("X,y stored in fdataX.npy and flabels.npy respectively")

2.1.2	Facial emotion Recognition and Prediction
  The fertestcustom.py is used to detect faces and recognize their emotions from a given image file. It first gets the image file path from the command line arguments and sets the path of the model file and the Haar cascade file. The script checks for the existence of these files and then loads the pre-trained deep learning model. Next, it converts the image to grayscale format and uses Haar cascade to detect faces in the image. For each detected face, the script resizes it to the size required by the model, normalizes it, and uses the model to predict the emotion. Finally, it draws a rectangle on the original image marking the face location and displays the predicted emotion label. Eventually, the script displays the emotion-tagged image and waits for the user to press a key. Exception handling is used throughout to ensure that any errors are caught and handled in a timely manner.

    # Loading the model
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    print("Loaded model from disk")

    # Convert to grayscale image
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)

    # Check that the face cascade file exists
    assert os.path.exists(face_cascade_path), "Face cascade file not found!"

    # Detecting faces
    face = cv2.CascadeClassifier(face_cascade_path)
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected faces: {len(faces)}")

    # Processing each face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Predicting emotion
        yhat = loaded_model.predict(cropped_img)
        emotion = labels[int(np.argmax(yhat))]
        cv2.putText(full_size_image, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: " + emotion)
    cv2.imshow('Emotion', full_size_image)
    cv2.waitKey()
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

 
 

2.1.3	Convolutional Neural Network (CNN) model
# Creating models
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # 6 types of emotions
])

# Compilation model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Viewing the model structure
model.summary()

 

2.1.4	Training the model
# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))


2.1.5	Forecasting the model
# Forecasting
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Evaluation accuracy
accuracy = np.sum(predicted_labels == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
 
2.2	Recommended Playlist System
The music recommendation playlist generation system leverages Spotify's track data to curate playlists that align with users' emotional states, focusing primarily on valence as the leading factor. Valence, which measures the positivity or negativity of a track, plays a crucial role in reflecting the emotional tone of the music, from happy and cheerful to sad or melancholic. 

 

	This system categorizes music into six emotional states—Happy, Sad, Angry, Neutral, Surprise, and Fear—by analyzing key track attributes. While valence dictates the overall mood direction, tempo, energy, acousticness, and liveness act as secondary factors that refine the recommendation process, ensuring that the playlists not only match emotional tone but also the intensity, style, and performance characteristics. Each emotion would have ranges for these parameters that best reflect typical characteristics associated with that emotion. This system uses the following range:

Happy
- Valence: 0.7–1 (Positive emotions and feelings)  
- Tempo: 120–170 (Fast tempo, generally upbeat)  
- Energy: 0.6–1 (High energy, energetic and lively)  
- Acousticness: 0.1–0.4 (Moderate to low)  
- Liveness: 0.2–0.5 (Moderate)  

Sad 
- Valence: 0.0–0.4 (Negative emotions and melancholic feelings)  
- Tempo: 60–90 (Slow tempo, usually mellow)  
- Energy: 0.1–0.3 (Low, calm, and introspective)  
- Acousticness: 0.4–0.8 (Higher)  
- Liveness: 0.1–0.3 (Low)  

Angry  
- Valence: 0.0–0.3 (Negative emotions and frustration)  
- Tempo: 130–200 (Fast, intense, and aggressive)  
- Energy: 0.8–1 (Very high, aggressive or intense)  
- Acousticness: 0.1–0.3 (Low)  
- Liveness: 0.2–0.5 (Moderate)  

Neutral  
- Valence: 0.4–0.6 (Balanced, non-extreme emotions)  
- Tempo: 90–120 (Medium)  
- Energy: 0.3–0.6 (Medium, steady)  
- Acousticness: 0.2–0.5 (Moderate)  
- Liveness: 0.1–0.3 (Low)  

Surprise  
- Valence: 0.4–0.8 (Mixed, unpredictable)  
- Tempo: 100–160 (Variable, fluctuating)  
- Energy: 0.5–0.8 (Moderate to high)  
- Acousticness: 0.1–0.4 (Low to moderate)  
- Liveness: 0.1–0.4 (Moderate)  

Fear
- Valence: 0.0–0.3 (Negative emotions, unsettling)  
- Tempo: 70–110 (Slow to medium tempo)  
- Energy: 0.2–0.5 (Low to moderate)  
- Acousticness: 0.2–0.6 (Higher acousticness, eerie sounds)  
- Liveness: 0.1–0.3 (Low liveness, isolated sounds)  

For instance, a playlist for happiness will feature high-valence tracks with fast tempos and energetic beats, while sad playlists lean towards lower valence and slower tempos with more acoustic textures. 

Spotify API:
	The Spotify Web API allows developers to access Spotify’s music catalog and user data programmatically. It follows REST principles and communicates over HTTPS. Spotify also offers a Web Playback SDK to control music playback directly in the browser, allowing seamless integration of audio streaming within the app.
To connect with the Spotify API, the app receives a Client ID and Client Secret to identify the app during API calls. After the user logs in, Spotify sends an authorization code to a redirect URI specified during app registration. Once authorized, the app can fetch moods or genres from Spotify’s database and control playback (pause, skip, or resume tracks).
When music data is required, Spotify provides music data as such:
 

Manual Upload:
Admin can manually upload music files or metadata directly to the web app’s data. The form includes audio file (e.g. mp3), title, artist, album. Since manually uploaded music does not compromise with Spotify’s data, it require a manual match to a mood. 
 

Feedback System:
Users can provide feedback on individual songs by liking or disliking them. Liking a song increases the likelihood of receiving similar recommendations in the future. Disliking a song decreases the chance of similar tracks appearing in future playlists. Feedback helps refine the recommendation algorithm over time, personalizing playlists based on user preferences. Users can also submit written feedback to report inaccurate mood-based recommendations or suggest improvements.

2.3	Front-end
Our front-end architecture uses the open-source framework java_music, which provides a foundation for building music applications. It allowed us to quickly integrate features like playlist management and music playback, so we can facous on developing our unique emotion-based recommendation system. 
2.3.1	Emotion recognition system
 I defined a RESTful controller implemented with Spring Boot. It is designed to receive an image uploaded by the user and recognize the emotion in the image. This controller calls the Python script to process the image and returns the recognition result to the user.
The core code of this system is as follows, and I provide a simple explanation for it: The system use the REST API for recognizing emotions from an image. It is mapped to the /recognize-emotion endpoint, which accepts a POST request with an uploaded image file. First, the image is saved as a temporary file on the server. Then, a Python script is called to analyze the image and detect the emotion. The script returns an output that includes the detected emotion, such as "Emotion: Happy". The method extract the emotion from the script's output. For example, it retrieves "Happy" from "Emotion: Happy". The emotion is then added to a response map. If any error occurs, such as during file handling or script execution, the method catches the error. It logs the issue and returns an error response with details. Finally, the method returns a JSON response. It contains either the detected emotion or error information.  
2.3.2	Recommended playlist Music API
   It connects to the Spotify Web API to find and return playlists or songs that match the emotion,which can recommend music based on a keyword that represents the user's emotion (e.g., "Happy").  Here's a simple explanation of how it works:
This code has two main parts:
getAccessToken Method:
This method gets a token from Spotify. It sends a request to Spotify's token URL using the client ID. Spotify returns an access token, which is needed to call other APIs.
recommendSongs Method:
This method recommends songs based on an emotion. It first gets the access token by calling the getAccessToken method. Then, it uses the emotion keyword and the token to call the SpotifySearchService.searchSong method. This method communicates with Spotify’s API to find songs or playlists that match the emotion.If the process works, it sends back the song data in a JSON response with an HTTP 200 status. If something goes wrong, such as the token failing or the API call not working, it catches the error and sends back an error message with an HTTP 500 status.
 

Chapter 3	Testing
We conducted a detailed test of whether the interface between the emotionRecognition system and the music recommendation system was docking. (EmotionRecognitionController.java)
 
3.1	white-box testing
3.1.1	Statement Coverage
TC1: Upload a valid image file.Expect to save the temporary file successfully and call the Python script.
3.1.2	Branch Coverage
TC2: Upload an image file that does not contain emotion text and expect that no emotion value will be found on regular matching.
TC3: Upload a valid image file and expect to find the emotion value at the time of regular matching.

3.1.3	Condition Coverage
TC4: Upload a valid image file.Expect matcher.find() to return true.
TC5: Upload an image file with no emotion text, expect matcher.find() to return false.
3.1.4	Branch/Condition Coverage
TC6: Conditions that combine TC2 and TC4.
TC7: Conditions that combine TC3 and TC5.
3.1.5	Condition Combination Coverage
TC8: Upload a valid image file, expect matcher.find() to return true and extract the emotion values correctly.
TC9: Upload a valid image file, expect matcher.find() to return false and do not extract emotion values.
TC10: Upload an image file with no emotion text, expect matcher.find() to return false and do not extract emotion values.
3.1.6	Path Coverage
Path 1: Normal flow, including creating temporary files, transferring images, successfully running Python scripts, pruning results, successfully matching regex, setting sentiment, returning response.
Path 2: Normal flow, but the regex match fails and a response without emotion needs to be returned.
Path 3: The exception flow, which includes creating temporary files, transferring images, failing to run Python scripts, catching exceptions, printing stack traces, setting error states and messages, and returning error responses.
Start
 |
 |   Try Block [Path 1, Path 2, Path 3]
 |   |
 |   |   Create Temp File [Path 1.1, Path 2.1]
 |   |   |
 |   |   |   Transfer Image to Temp File [Path 1.1.1, Path 2.1.1]
 |   |   |
 |   |   |   Run Python Script [Path 1.1.2, Path 2.1.2]
 |   |   |   |
 |   |   |   |   If Python Script Success [Path 1.1.2.1]
 |   |   |   |   |
 |   |   |   |   |   Trim Result [Path 1.1.2.1.1]
 |   |   |   |   |
 |   |   |   |   |   Regex Match [Path 1.1.2.1.2]
 |   |   |   |       |
 |   |   |   |       |   If Match Found [Path 1.1.2.1.2.1]
 |   |   |   |       |   |
 |   |   |   |       |   Set Emotion [Path 1.1.2.1.2.1.1]
 |   |   |   |       |
 |   |   |   |       |   If Match Not Found [Path 1.1.2.1.2.2]
 |   |   |   |
 |   |   |   Return Response [Path 1.1.3, Path 2.1.3]
 |   |
 |   |   Catch Exception Block [Path 3.1]
 |       |
 |       |   Print Stack Trace [Path 3.1.1]
 |       |
 |       |   Set Error Status and Message [Path 3.1.2]
 |       |
 |       |   Return Error Response [Path 3.1.3]
 |
End


3.1.7	Test case design
TC1: Upload a valid image file, expect to save the temporary file successfully and call the Python script, the regular expression matches successfully.
 
TC2: A valid image file is uploaded, but the output of the Python script does not contain emotion text and the regular expression matching is expected to fail.
 
TC3: Upload an invalid image file (for example, in TXT format), expecting an exception to be thrown when saving a temporary file or calling a Python script.
 

3.2	black-box testing
We designed a table to show equivalence class partitioning in black-box testing. This table will be used to describe the input conditions and equivalence classes of an emotion recognition system.
 
 
Chapter 4	Display QHQ Music Player system
User Login System:
	Users can log in to existing accounts, create new accounts using email, or access the system without logging in.
  

Home Page:
	Displays a list of available music that users can play. Music can be sorted by "Hottest" or "Newest"
	Users can select and play music directly from the list.
	Users can also upload local music to the system.
 
Automatic Recommendation (Facial Analysis):
	When generating music recommendations, the system requests camera access to perform facial mood analysis.
 
	Upon verification, the system analyzes the user's facial expression to determine their mood.
	If no face is detected, an error message is displayed.
 
	A playlist is generated based on the user's mood by integrating the Spotify API and uploaded music.

Manual Recommendation (Mood Selection):
	Users can manually select their mood from six options: Happy, Sad, Angry, Neutral, Surprise, or Fear.
	The system generates a playlist that matches the selected mood.
 
User Profile:
	Users can update personal information and account settings, including profile picture, ID, phone number, email, and personal description.
 
	Password changes are also supported.
 
Music Controls:
	While music is playing, users can:
	Play, pause, skip to the next song, return to the previous song, or dislike (which decreases similar recommendations).
	Download music and adjust playback speed.
 
 
References
[1]	LeCun, Bengio, Y., & Hinton, G. (2015). Deep learning. Nature (London), 521(7553), 436–444. https://doi.org/10.1038/nature14539.
[2]	B. Chen and J. Zhang, “Tuple density: A new metric for combinatorial test suites,” in Proceeding of the 33rd International Conference on Software Engineering (ICSE’11), 2011, pp. 876–879.
[3]	A. Arcuri and L. Briand, “Formal analysis of the probability of interaction fault detection using random testing,” IEEE Transactions on Software Engineering, vol. 38, no. 5, pp. 1088–1099, 2012.
[4]	G. J. Myers, The Art of Software Testing. John Wiley & Sons: New York, 2004. 
[5]	“Keras: The python deep learning library,” 2020, https://keras.io3.
[6]	“scikit-learn, machine learning in python,” 2020, https://scikit-learn.org/stable/.
[7]	 H. Zhang, K. Zhang and N. Bryan-Kinns, "Exploiting the emotional preference of music for music recommendation in daily activities," 2020 13th International Symposium on Computational Intelligence and Design (ISCID), Hangzhou, China,2020, pp. 350-353, doi: 10.1109/ISCID51228.2020.00085. 
[8]	M. M. Joseph, D. Treessa Varghese, L. Sadath and V. P. Mishra, "Emotion Based Music Recommendation System," 2023 International Conference on Computational Intelligence and Knowledge Economy (ICCIKE), Dubai, United Arab Emirates, 2023, pp. 505-510, doi: 10.1109/ICCIKE58312.2023.10131874. 
[9]	Mr Chakrapani D S, Sidrath Iram, Suchitra R Bhat Agni, Supritha L, Leelavathi S, MUSIC RECOMMENDATION BASED ON FACIAL EMOTION RECOGNITION, 2023 
[10]	Martinez, Aleix M., and Shichuan Du. "A Model of the Perception of Facial Expressions of Emotion by Humans: Research Overview and Perspectives." Journal of Vision, vol. 12, no. 2, 2012, pp. 130. https://doi.org/10.1167/12.2.25. 
[11]	Tian, Ying-Li, et al. "Facial Expression Recognition Using Support Vector Machines." International Conference on Pattern Recognition, vol. 4, 2000, pp. 448-451. https://doi.org/10.1109/ICPR.2000.903017. 9 
[12]	M. M. Joseph, D. Treessa Varghese, L. Sadath and V. P. Mishra, "Emotion Based Music Recommendation System," 2023 International Conference on Computational Intelligence and Knowledge Economy (ICCIKE), Dubai, United Arab Emirates, 2023, pp. 505-510, doi: 10.1109/ICCIKE58312.2023.10131874.          
[13]	Mr Chakrapani D S, Sidrath Iram, Suchitra R Bhat Agni, Supritha L, Leelavathi S, MUSIC RECOMMENDATION BASED ON FACIAL EMOTION RECOGNITION, 2023 . 
[14]	Cano, Pedro, et al. "Content-based Music Recommendation." Proceedings of the ACM Workshop on Recommender Systems, ACM, 2005, pp. 5156. https://doi.org/10.1145/1109210.1109220. 
[15]	Schedl, Markus, et al. "Current Challenges and Visions in Music Recommender Systems Research." International Journal of Multimedia Information Retrieval, vol. 6, no. 2, 2017, pp. 95-116. https://doi.org/10.1007/s13735-017-0115-2. \
[16]	Chen, Liang, et al. "Towards Mood-Driven Music Recommendation." Proceedings of the 4th International Workshop on Human-Centric Multimedia Analysis, ACM, 2013, pp. 7-12. https://doi.org/10.1145/2512142.2512151. 
[17]	Yang, Yi-Hsuan, and Homer H. Chen. "Machine Recognition of Music Emotion: A Review." ACM Transactions on Intelligent Systems and Technology (TIST), vol. 3, no. 3, 2012, pp. 1-30. https://doi.org/10.1145/2168752.2168754.
[18]	geeeeeeeek. (n.d.). java_music [GitHub repository]. Retrieved January 8, 2025, from https://github.com/geeeeeeeek/java_music.
