from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Attention
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score
from keras.optimizers import Adam

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random string
global df, model, imputer, scaler, user_behaviors,result
users = {}
def load_profile_data():
    global profile_df
    profile_df = pd.read_csv("C:\\Users\\Shirisha Reddy\\Downloads\\PrivilegeInsider_Attention\\profiles dataset.csv")
    p = profile_df.copy()
    p['functional_unit'] = p['functional_unit'].str.split(' - ').str[1]
    p['department'] = p['department'].str.split(' - ').str[1]
    p['team'] = p['team'].str.split(' - ').str[1]
    return p

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        u = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(u, axis=1)
        output = inputs * a
        return output

def load_models():
    global model, imputer, scaler
    df = pd.read_csv("C:\\Users\\Shirisha Reddy\\Downloads\\PrivilegeInsider_Attention\\dataset.csv")

    categorical_columns = ['user']
    for col in categorical_columns:
        df[col] = pd.factorize(df[col])[0] + 1
    df['Behaviour'] = df['Behaviour'].replace({'Normal': 0, 'Abnormal': 1})

    X = df.iloc[:, 0:16].values
    y = df.iloc[:, 17].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(AttentionLayer())
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(32))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, epochs=600, batch_size=64, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)
    # print(y_pred)

    y_pred_binary = (y_pred > 0.5).astype(int)
    class_labels = np.where(y_pred_binary == 0, 'Normal', 'Attack')
    loss, Accuracy = model.evaluate(X_test, y_test)
    Precision = precision_score(y_test, y_pred_binary)
    Recall= recall_score(y_test, y_pred_binary)
    Fscore= f1_score(y_test, y_pred_binary)
    conf_matrix= confusion_matrix(y_test, y_pred_binary)
    ac= accuracy_score(y_test, y_pred_binary)
    print(ac)
    
    print("Accuracy: " , round(ac,2)*100)
    # print("Accuracy: " , round(Accuracy,2))
    print("Precision: " , round(Precision,2)*100)
    print("Recall: " , round(Recall,2)*100)
    print("Fscore: " , round(Fscore,2)*100)
    print("conf_matrix: " , conf_matrix)
@app.route('/', methods=['GET', 'POST'])
def home():
    if 'username' in session:
        return redirect(url_for('upload'))
    else:
        return redirect(url_for('login'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html', error='')
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            return render_template('signup.html', error='Username already exists')

        users[username] = {'username': username, 'password': generate_password_hash(password)}
        session['username'] = username
        return redirect(url_for('login'))

    return render_template('signup.html', error='')
user_behaviors = {}

# Function to upload file and get predictions
def upload_file(test_df):
    global df, model, imputer, scaler, user_behaviors, result
    
    t = test_df.copy()  # Make a copy of the DataFrame to avoid modifying the original
    categorical_columns = ['user']
    for col in categorical_columns:
        test_df[col] = pd.factorize(test_df[col])[0] + 1
            
    test_features = test_df.iloc[:, :16].values
    test_features = imputer.transform(test_features)
    test_features = scaler.transform(test_features)
    test_features = test_features.reshape((test_features.shape[0], 1, test_features.shape[1]))

    result_list = []

    for i in range(len(test_df)):
        user_data = test_features[i:i+1]
        user = t['user'].iloc[i]
        
        if user in user_behaviors:
            predicted_label = user_behaviors[user]  # Get the prediction from the dictionary
        else:
            user_prediction = model.predict(user_data)
            predicted_label = 'Attack' if user_prediction > 0.5 else 'Normal'
            user_behaviors[user] = predicted_label  # Store the prediction in the dictionary

        result_list.append({'user': user, 'Behaviour': predicted_label})

    result_df = pd.DataFrame(result_list)
    p = load_profile_data()
        
    result = pd.merge(result_df, p, on='user', how='left')
    return result

# Route for uploading files
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')

        file_path = request.files['file']

        if file_path.filename == '':
            return render_template('upload.html', error='No selected file')

        try:
            test_df = pd.read_csv(file_path)
            result = upload_file(test_df)
            return render_template('results.html', results=result.to_dict('records'))
        except Exception as e:
            # Handle errors
            return render_template('upload.html', error=str(e))
    else:
        return render_template('upload.html')
@app.route('/user/<user_id>')
def user_details(user_id):
    global result
    profile_df = load_profile_data()
    user_profile = profile_df[profile_df['user'] == user_id].to_dict('records')[0]
    
    
    # Get user behavior if available
    user_b = result[result['user'] == user_id]['Behaviour'].iloc[0]
    return render_template('user_details.html', user_profile=user_profile, user_b=user_b)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    load_models()
    app.run(debug=True)